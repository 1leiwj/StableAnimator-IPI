"""Check the dense poses used by the IPI training dataset before allocating GPUs."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np


def inspect_clip(folder):
    path = folder / "poses_dense133.npy"
    if not path.is_file():
        path = folder / "poses_dense.npy"
    if not path.is_file():
        raise ValueError("missing dense pose file")
    count = sum(p.is_file() and p.suffix in (".jpg", ".png") for p in (folder / "images").iterdir())
    if not count:
        raise ValueError("no training frames")
    a = np.load(path, allow_pickle=False, mmap_mode="r")
    if a.ndim != 3 or a.shape[:2] != (count, 133) or a.shape[2] < 2 or a.dtype.kind not in "fiu":
        raise ValueError(f"expected [{count}, 133, 2+] numeric poses, got {a.shape} {a.dtype}")
    body = a[:, :18, :2]
    valid = np.isfinite(body).all(-1) & (body != -1).all(-1) & (body != 0).any(-1)
    if not (valid.sum(axis=1) >= 3).any():
        raise ValueError("no frame with three valid body joints")
    visible = valid[:, [0, 2, 5, 4, 7]].all(axis=1)
    y = body[:, :, 1]
    overhead = visible & (y[:, 4] < y[:, 0]) & (y[:, 7] < y[:, 0])
    lowered = visible & (y[:, 4] > y[:, 2]) & (y[:, 7] > y[:, 5])
    # Count a low-to-high transition only when the high frame comes later.
    return Counter(clips=1, frames=count, arm_check_frames=int(visible.sum()),
                   both_wrists_above_nose_frames=int(overhead.sum()),
                   clips_with_low_then_high=int((np.maximum.accumulate(lowered)[:-1] & overhead[1:]).any()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, nargs="?", default=Path("animation_data/video_dec_path.txt"))
    args = parser.parse_args()
    folders = list(dict.fromkeys(line.strip() for line in args.manifest.read_text(encoding="utf-8-sig").splitlines()
                                 if line.strip()))
    totals, errors = Counter(), Counter()
    for folder in folders:
        try:
            totals.update(inspect_clip(Path(folder)))
        except (ValueError, OSError, EOFError) as exc:
            reason = str(exc)
            if sum(errors.values()) < 8:
                print(f"Invalid: {folder}: {reason}")
            errors[reason] += 1
    print(f"Valid clips: {totals['clips']}/{len(folders)}; invalid: {sum(errors.values())}")
    print(f"Frames with nose, shoulders and wrists visible: {totals['arm_check_frames']}/{totals['frames']}")
    print(f"Both wrists above nose: {totals['both_wrists_above_nose_frames']} frames; "
          f"clips with low then high: {totals['clips_with_low_then_high']}")
    if errors or not folders:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
