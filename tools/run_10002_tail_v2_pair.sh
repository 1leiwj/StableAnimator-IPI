#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

step="${CHECKPOINT_STEP:-6000}"
[[ "$step" =~ ^[1-9][0-9]*$ ]] || { echo "Choose a completed numbered checkpoint." >&2; exit 2; }
weights=./checkpoints/Animation_ipi_temporal_v2_from_19500_base_29500_adapter_fresh20k
root="${EXPERIMENT_ROOT:-./basic_infer/10002_tail_v2_from_19500_29500_step${step}_512}"
for model in unet pose_net face_encoder ipi_adapter; do
    file="$weights/checkpoint-$step/$model-$step.pth"
    [[ -s "$file" ]] || { echo "Missing checkpoint file: $file" >&2; exit 1; }
done
for mode in off on; do
    [[ ! -e "$root/$mode" ]] || { echo "Output already exists: $root/$mode" >&2; exit 1; }
done
for mode in off on; do
    off=0
    if [[ "$mode" == off ]]; then off=1; fi
    env CHECKPOINT_ROOT="$weights" CHECKPOINT_STEP="$step" IPI_TEMPORAL_V2=1 NO_IPI="$off" \
        REFERENCE_IMAGE=./basic_infer/ces/30555.jpg DRIVING_DIR=./animation_data/ceshi/10002 \
        WIDTH=512 HEIGHT=512 RETARGET=1 POSE_ROOT_MOTION_SCALE=0 MATCH_ANKLE_HEIGHT=1 \
        POSE_COORDINATE_SPACE=normalized FRAME_START=360 FRAME_END=380 SOURCE_TIMELINE=1 \
        POSE_SCALE=3 IPI_SCALE=1 MOTION_BUCKET_ID=300 TILE_SIZE=20 FRAMES_OVERLAP=6 \
        NUM_INFERENCE_STEPS=20 GUIDANCE_SCALE=3 MIN_GUIDANCE_SCALE=2 MAX_GUIDANCE_SCALE=3.5 \
        INDEPENDENT_NOISE=0 SEED=23123134 NOISE_AUG_STRENGTH=0 FPS_TIME_ID=6 \
        REFERENCE_LATENT_SCALE=1 REFERENCE_CLIP_SCALE=1 IS_STATUE=0 CLIP_BATCH_SIZE=32 \
        DECODE_CHUNK_SIZE=4 SKIP_GIF=1 OUTPUT_DIR="$root/$mode" bash command_ipi_infer.sh
done
cmp "$root/off/retargeted_poses_dense133.npy" "$root/on/retargeted_poses_dense133.npy"