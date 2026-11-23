#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

pilot=./checkpoints/Animation_ipi_limb_pilot_from_20000
stride="${TRAIN_FRAME_STRIDE:-4}"
[[ "$stride" =~ ^[1-9][0-9]*$ ]] || { echo "TRAIN_FRAME_STRIDE must be a positive integer." >&2; exit 2; }
suffix=""
if [[ "$stride" != 4 ]]; then suffix="_stride${stride}"; fi
weights="./checkpoints/Animation_ipi_joint_motion_from_limb1000${suffix}"
step="${CHECKPOINT_STEP:-1000}"
results="${EXPERIMENT_ROOT:-./basic_infer/10002_ipi_joint_motion${suffix}_step${step}}"
case "${1:-check}" in
    check)
        python tools/check_ipi_joint_motion.py
        ;;
    train)
        python tools/check_ipi_joint_motion.py
        env -u RESUME_FROM_CHECKPOINT IPI_ONLY_FINETUNE=1 IPI_TEMPORAL_V2=1 TARGET_FRAME_POSE=1 \
            IPI_JOINT_MOTION=1 IPI_LIMB_MASK_PROB=0.25 INIT_CHECKPOINT_ROOT="$pilot" INIT_CHECKPOINT_STEP=1000 \
            TRAIN_FRAME_STRIDE="$stride" \
            MAX_TRAIN_STEPS=1000 LR_WARMUP_STEPS=33 TRAIN_OUTPUT_DIR="$weights" bash command_finetune_diaosu.sh
        ;;
    eval)
        COMPARE_JOINT_BRANCH=1 bash tools/run_10002_final_pairs.sh "$weights" "$step" "$results"
        bash tools/run_ipi_joint_motion.sh score
        ;;
    score)
        for segment in raise hold; do
            for mode in global_only on; do
                run="$results/$segment/$mode"
                python tools/evaluate_joint_following.py "$run" --pck-only > "$run/joint_scores.json.tmp"
                mv -- "$run/joint_scores.json.tmp" "$run/joint_scores.json"
                python - "$run/joint_scores.json" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
scores = json.loads(path.read_text())
groups = scores["pck_0.05_diagonal"]["groups"]
overhead = scores["both_wrists_above_nose"]
print(f"{path.parent}: overhead {overhead['matched_frames']}/{overhead['target_frames']}, "
      f"visible {overhead['detected_frames']}/{overhead['target_frames']}, "
      f"left wrist PCK {groups['left_wrist']['pck']}, right wrist PCK {groups['right_wrist']['pck']}")
PY
            done
        done
        ;;
    *) echo "Usage: bash $0 {check|train|eval|score}" >&2; exit 2 ;;
esac
