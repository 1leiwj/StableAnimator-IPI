"""Retarget normalized DWPose keypoints to the proportions of a reference image."""

import numpy as np


# OpenPose body order at the start of the DWPose whole-body array.
BODY_BONES = (
    (1, 0), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
    (0, 14), (0, 15), (14, 16), (15, 17),
)
BODY_PARENTS = {child: parent for parent, child in BODY_BONES}
MIRROR = {2: 5, 5: 2, 3: 6, 6: 3, 4: 7, 7: 4,
          8: 11, 11: 8, 9: 12, 12: 9, 10: 13, 13: 10}
# Whole-body DWPose orders the left and right feet and hands separately.
DETAIL_GROUPS = (
    (18, 21, (13, 12, 11)), (21, 24, (10, 9, 8)),
    (24, 92, (0, 1)),
    (92, 113, (7, 6, 5)), (113, 133, (4, 3, 2)),
)


def valid_points(points):
    return np.isfinite(points).all(axis=-1) & (points != -1).all(axis=-1)


class NoPersonDetected(ValueError):
    """No person box was available for pose detection."""


def detect_reference_pose(image, detector=None):
    import cv2

    from DWPose.dwpose_utils.wholebody import Wholebody

    pixels = np.asarray(image.convert("RGB").resize((768, 1024)))
    if detector is None:
        detector = Wholebody()
    candidates, scores = detector(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
    if not len(candidates):
        raise NoPersonDetected("DWPose could not detect a person in the image.")

    def body_area(index):
        body = candidates[index, :18]
        visible = (scores[index, :18] > 0.3) & np.isfinite(body).all(axis=-1)
        return np.ptp(body[visible], axis=0).prod() if visible.sum() >= 3 else 0

    index = max(range(len(candidates)), key=body_area)
    body = candidates[index, :18].astype(np.float32) / np.array([768, 1024], dtype=np.float32)
    body[~(scores[index, :18] > 0.3) | ~np.isfinite(body).all(axis=-1)] = -1
    return body


def _with_shoulder_neck(points, visible):
    """Use the shoulder midpoint internally when DWPose's neck is missing."""
    use_midpoint = ~visible[..., 1] & visible[..., 2] & visible[..., 5]
    working = points.copy()
    midpoint = (working[..., 2, :] + working[..., 5, :]) / 2
    working[..., 1, :] = np.where(use_midpoint[..., None], midpoint, working[..., 1, :])
    return working


def match_ankle_height(mapped, source):
    """Translate each frame vertically to reproduce the driver's lower-ankle displacement."""
    ankles = [10, 13]
    reliable = valid_points(source[:, ankles]).all(axis=1) & valid_points(mapped[:, ankles]).all(axis=1)
    frames = np.flatnonzero(reliable)
    if not frames.size:
        raise ValueError("Ankle-height alignment requires at least one frame with both ankles detected.")
    source_y = source[frames][:, ankles, 1].max(axis=1)
    target_y = mapped[frames][:, ankles, 1].max(axis=1)
    # Match frame-to-frame displacement in normalized image height. This preserves
    # a genuine jump and leaves the stance height of the first reliable frame intact.
    adjustment = (source_y - source_y[0]) - (target_y - target_y[0])
    shift = np.interp(np.arange(len(source)), frames, adjustment).astype(np.float32)
    visible = valid_points(mapped)
    mapped[..., 1] += np.where(visible, shift[:, None], 0)


def retarget_pose(source, reference, width, height, root_motion_scale=1.0, ankle_height=False):
    source = np.asarray(source)
    reference = np.asarray(reference)
    if source.ndim != 3 or source.shape[1:] != (133, 2) or not len(source):
        raise ValueError(f"Expected a nonempty [frames, 133, 2] DWPose array, got {source.shape}.")
    if reference.ndim != 2 or reference.shape[0] < 18 or reference.shape[1] != 2:
        raise ValueError(f"Expected a reference with at least 18 (x, y) joints, got {reference.shape}.")
    for dimension in (width, height):
        if not isinstance(dimension, (int, np.integer)) or isinstance(dimension, (bool, np.bool_)) or dimension <= 0:
            raise ValueError(f"Expected positive integer canvas dimensions, got {width}x{height}.")

    for label, points in (("Driving pose", source), ("Reference pose", reference)):
        if points.dtype.kind not in "iuf":
            raise ValueError(f"{label} must contain numeric coordinates.")
        good = valid_points(points)
        if not good.any() or np.abs(points[good]).max() > 1.5:
            raise ValueError(f"{label} must contain normalized DWPose coordinates (x/W, y/H).")

    original_visible = valid_points(source)
    if not np.isfinite(root_motion_scale) or root_motion_scale < 0:
        raise ValueError("root_motion_scale must be finite and nonnegative.")
    original_reference_visible = valid_points(reference)
    if original_reference_visible[:18].sum() < 5:
        raise ValueError("DWPose found fewer than five reliable reference body joints.")

    source = _with_shoulder_neck(source.astype(np.float32, copy=False), original_visible)
    reference = _with_shoulder_neck(reference.astype(np.float32, copy=False), original_reference_visible)
    visible = valid_points(source)
    reference_visible = valid_points(reference)
    anchor_frames = np.flatnonzero(visible[:, 1])
    if reference_visible[1] and anchor_frames.size:
        roots = source[:, 1]
        offset = reference[1] - source[anchor_frames[0], 1]
    else:
        hips = [8, 11]
        anchor_frames = np.flatnonzero(visible[:, hips].all(axis=1))
        if not reference_visible[hips].all() or not anchor_frames.size:
            raise ValueError("Reference and driving pose need a shared neck, shoulder pair, or hip pair anchor.")
        offset = reference[hips].mean(axis=0) - source[anchor_frames[0], hips].mean(axis=0)
        roots = source[:, hips].mean(axis=1)

    root_shift = np.zeros((len(source), 2), dtype=np.float32)
    if root_motion_scale != 1:
        # Interpolate only the translation; missing body and hand points stay missing.
        for axis in range(2):
            trajectory = np.interp(np.arange(len(source)), anchor_frames, roots[anchor_frames, axis])
            root_shift[:, axis] = (root_motion_scale - 1) * (trajectory - roots[anchor_frames[0], axis])

    size = np.array([width, height], dtype=np.float32)
    scales = {}
    for parent, child in BODY_BONES:
        pairs = visible[:, parent] & visible[:, child]
        if not pairs.any():
            continue
        lengths = np.linalg.norm((source[pairs, child] - source[pairs, parent]) * size, axis=-1)
        driving_length = np.median(lengths)
        if driving_length <= 1:
            continue
        mirrored = (MIRROR.get(parent, parent), MIRROR.get(child, child))
        for start, end in ((parent, child), mirrored):
            if not (reference_visible[start] and reference_visible[end]):
                continue
            target_length = np.linalg.norm((reference[end] - reference[start]) * size)
            if target_length > 1:
                scales[parent, child] = np.clip(target_length / driving_length, 0.4, 1.5)
                break

    mapped = np.full(source.shape, -1, dtype=np.float32)
    for frame, points in enumerate(source):
        good = visible[frame]
        result = mapped[frame]
        result[good] = points[good] + offset
        for parent, child in BODY_BONES:
            if not good[child]:
                continue
            if good[parent]:
                result[child] = result[parent] + (points[child] - points[parent]) * scales.get((parent, child), 1)
            else:
                # Translate visible endpoints with their nearest visible ancestor
                # when an intermediate joint is undetected.
                ancestor = BODY_PARENTS.get(parent)
                while ancestor is not None and not good[ancestor]:
                    ancestor = BODY_PARENTS.get(ancestor)
                if ancestor is not None:
                    result[child] = points[child] + result[ancestor] - points[ancestor]

        # Keep side-specific details attached to their own limb, even when limbs cross.
        for start, end, joints in DETAIL_GROUPS:
            anchor = next((joint for joint in joints if good[joint]), None)
            if anchor is not None:
                indices = np.flatnonzero(good[start:end]) + start
                result[indices] = points[indices] + result[anchor] - points[anchor]

        if root_motion_scale != 1:
            result[good] += root_shift[frame]
    if ankle_height:
        match_ankle_height(mapped, source)
    in_frame = ((mapped >= 0) & (mapped <= 1)).all(axis=-1)
    mapped[~(in_frame & original_visible)] = -1
    return mapped


def render_body_poses(poses, width, height):
    from PIL import Image
    from DWPose.dwpose_utils.util import draw_bodypose

    frames = []
    for points in poses[:, :18]:
        subset = np.where(valid_points(points), np.arange(18), -1)[None]
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        frames.append(Image.fromarray(draw_bodypose(canvas, points, subset)))
    return frames
