"""Compare saved IPI parameters between two numbered checkpoints of the same run."""
import argparse
from pathlib import Path

import torch


def compare(before, after, keys):
    diff_sq = base_sq = 0.0
    changed = 0
    for key in keys:
        a, b = before[key], after[key]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(f"Shape or dtype mismatch for {key}; use two checkpoints from the same run.")
        if not torch.equal(a, b):
            changed += 1
        a, b = a.float(), b.float()
        diff_sq += (b - a).square().sum().item()
        base_sq += a.square().sum().item()
    return changed, (diff_sq / base_sq) ** 0.5 if base_sq else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("before", type=int)
    parser.add_argument("after", type=int)
    args = parser.parse_args()
    if args.before >= args.after:
        parser.error("Expected two increasing checkpoint steps.")
    for model, groups in (
        ("ipi_adapter", {"adapter": None}),
        ("unet", {"gates": "ipi_temporal_gate", "projections": "ipi_temporal_projection.weight"}),
        ("pose_net", {"frozen PoseNet": None}),
    ):
        paths = [args.root / f"checkpoint-{s}" / f"{model}-{s}.pth" for s in (args.before, args.after)]
        before, after = [torch.load(p, map_location="cpu", mmap=True, weights_only=True) for p in paths]
        for group, suffix in groups.items():
            keys = sorted(k for k in before if suffix is None or k.endswith(suffix))
            if not keys or set(keys) != {k for k in after if suffix is None or k.endswith(suffix)}:
                raise ValueError(f"Missing or incompatible {group} keys in {paths}.")
            changed, relative = compare(before, after, keys)
            ratio = f"{relative:.3e}" if relative is not None else "undefined (initial norm zero)"
            print(f"{group}: {changed}/{len(keys)} tensors changed; relative L2 update = {ratio}", flush=True)
        del before, after


if __name__ == "__main__":
    main()
