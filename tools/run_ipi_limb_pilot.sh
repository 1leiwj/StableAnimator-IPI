#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

base=./checkpoints/Animation_ipi_temporal_v2_from_19500_base_29500_adapter_fresh20k
pilot=./checkpoints/Animation_ipi_limb_pilot_from_20000
case "${1:-check}" in
    check)
        python tools/check_ipi_training_poses.py
        ;;
    weights)
        python tools/check_ipi_updates.py "$base" "${BEFORE_STEP:-19000}" 20000
        ;;
    train)
        python tools/check_ipi_training_poses.py
        env -u RESUME_FROM_CHECKPOINT IPI_ONLY_FINETUNE=1 IPI_TEMPORAL_V2=1 TARGET_FRAME_POSE=1 \
            INIT_CHECKPOINT_ROOT="$base" INIT_CHECKPOINT_STEP=20000 IPI_LIMB_MASK_PROB=0.25 \
            LR_WARMUP_STEPS=33 MAX_TRAIN_STEPS=1000 TRAIN_OUTPUT_DIR="$pilot" \
            bash command_finetune_diaosu.sh
        ;;
    eval)
        bash tools/run_10002_final_pairs.sh "$pilot" 1000 ./basic_infer/10002_ipi_limb_pilot_step1000
        ;;
    *) echo "Usage: bash $0 {check|weights|train|eval}" >&2; exit 2 ;;
esac
