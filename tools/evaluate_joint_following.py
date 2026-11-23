"""Compare generated body keypoints with the exact retargeted controls of a run."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np


def vector(points, start, end):
    a, b = points[[start, end]]
    if not np.isfinite([a, b]).all() or np.linalg.norm(b - a) < 1:
        return None
    return b - a


def joint_measures(points):
    values = {}
    for side, shoulder, wrist, hip, knee, ankle in (
        ("right", 2, 4, 8, 9, 10),
        ("left", 5, 7, 11, 12, 13),
    ):
        arm = vector(points, shoulder, wrist)
        thigh = vector(points, knee, hip)
        shin = vector(points, knee, ankle)
        values[f"{side}_arm_direction_deg"] = (
            float(np.degrees(np.arctan2(arm[1], arm[0]))) if arm is not None else None
        )
        angle = None
        if thigh is not None and shin is not None:
            cosine = np.dot(thigh, shin) / (np.linalg.norm(thigh) * np.linalg.norm(shin))
            angle = float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
        values[f"{side}_knee_angle_deg"] = angle
    return values


def pixel_coordinates(array, count, width, height):
    points = np.array(array, dtype=np.float64)
    if points.ndim != 3 or points.shape[0] != count or points.shape[1] < 18 or points.shape[2] != 2:
        raise ValueError(f"Expected {count} frames of normalized OpenPose body points, got {points.shape}")
    points = points[:, :18]
    visible = np.isfinite(points).all(axis=-1) & ((points >= 0) & (points <= 1)).all(axis=-1)
    points[~visible] = np.nan
    return points * [width, height]


def limb_pck(target, observed, width, height):
    """PCK over visible target joints, using 5% of the output image diagonal."""
    threshold = float(0.05 * np.hypot(width, height))
    visible = np.isfinite(target).all(axis=-1)
    detected = np.isfinite(observed).all(axis=-1) & visible
    correct = detected & (np.linalg.norm(observed - target, axis=-1) < threshold)
    groups = {}
    # OpenPose body order, shoulders through ankles on both sides.
    for name, indices in {
        "limbs_12": list(range(2, 14)), "wrists": [4, 7], "knees": [9, 12],
        "right_wrist": [4], "left_wrist": [7], "right_knee": [9], "left_knee": [12],
    }.items():
        total = int(visible[:, indices].sum())
        found = int(detected[:, indices].sum())
        hits = int(correct[:, indices].sum())
        groups[name] = {
            "correct": hits, "target_visible": total, "detected": found,
            "pck": hits / total if total else None,
            "detection_coverage": found / total if total else None,
        }
    return {"threshold_px": threshold, "groups": groups}


def overhead_counts(target, observed):
    # Score every target overhead frame. A missing generated joint counts as a miss.
    def above(points):
        y = points[:, [0, 4, 7], 1]
        return (y[:, 1] < y[:, 0]) & (y[:, 2] < y[:, 0])

    required = above(target)
    detected = np.isfinite(observed[:, [0, 4, 7]]).all(axis=(1, 2))
    return {"target_frames": int(required.sum()), "matched_frames": int((required & above(observed)).sum()),
            "detected_frames": int((required & detected).sum())}


def detect_frames(run_dir, count, width, height):
    from PIL import Image

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from DWPose.retarget import NoPersonDetected, detect_reference_pose
    from DWPose.dwpose_utils.wholebody import Wholebody

    folder = run_dir / "animated_images"
    files = [folder / f"frame_{i}.png" for i in range(count)]
    if set(folder.glob("frame_*.png")) != set(files):
        raise ValueError(f"Expected contiguous frame_0.png through frame_{count - 1}.png in {folder}")
    detector = Wholebody(providers=["CPUExecutionProvider"], require_person_box=True)
    result = np.full((count, 18, 2), -1, np.float32)
    for i, path in enumerate(files):
        with Image.open(path) as image:
            if image.size != (width, height):
                raise ValueError(f"Unexpected size {image.size}: {path}")
            try:
                result[i] = detect_reference_pose(image, detector)
            except NoPersonDetected:
                continue
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--generated-pose", type=Path, help="Optional externally checked [frames,18+,2] normalized OpenPose points in output order")
    parser.add_argument("--pck-only", action="store_true", help="Only report limb PCK@0.05 (image diagonal) and detection coverage")
    args = parser.parse_args()
    settings = json.loads((args.run_dir / "inference_settings.json").read_text(encoding="utf-8"))
    count, width, height = (settings[k] for k in ("num_frames", "width", "height"))
    if count < 1 or settings["frame_end"] - settings["frame_start"] != count:
        raise ValueError("Frame count disagrees with the source range in inference_settings.json")
    target = np.load(args.run_dir / "retargeted_poses_dense133.npy", allow_pickle=False)
    if args.generated_pose:
        observed = np.load(args.generated_pose, allow_pickle=False)
    else:
        observed = detect_frames(args.run_dir, count, width, height)
    target = pixel_coordinates(target, count, width, height)
    observed = pixel_coordinates(observed, count, width, height)
    report = {
        "run": str(args.run_dir),
        "source_frames": [settings["frame_start"], settings["frame_end"] - 1],
        "generated_pose_source": str(args.generated_pose) if args.generated_pose else "DWPose CPU, person box required, joint score >0.3",
        "pck_0.05_diagonal": limb_pck(target, observed, width, height),
        "both_wrists_above_nose": overhead_counts(target, observed),
    }
    if args.pck_only:
        print(json.dumps(report, indent=2, allow_nan=False))
        return
    errors = {}
    frames = []
    for i, (expected, actual) in enumerate(zip(target, observed)):
        target_angles = joint_measures(expected)
        observed_angles = joint_measures(actual)
        frame = {"source_frame": settings["frame_start"] + i, "target": target_angles, "observed": observed_angles}
        difference = {}
        for key, a in target_angles.items():
            b = observed_angles[key]
            difference[key] = None if a is None or b is None else abs((a - b + 180) % 360 - 180)
        for name, joint in (("right_wrist", 4), ("left_wrist", 7), ("right_knee", 9), ("left_knee", 12)):
            distance = np.linalg.norm(expected[joint] - actual[joint])
            difference[name + "_px"] = float(distance) if np.isfinite(distance) else None
        frame["absolute_error"] = difference
        frames.append(frame)
        for key, error in difference.items():
            errors.setdefault(key, [])
            if error is not None:
                errors[key].append(error)
    summary = {
        key: {"valid_frames": len(values), "total_frames": count, "mean_absolute_error": float(np.mean(values)) if values else None}
        for key, values in errors.items()
    }
    report.update(summary=summary, frames=frames)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
