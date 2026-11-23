#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

root="${EXPERIMENT_ROOT:-./basic_infer/10002_tail_31000}"
cases=("$@")
if (( $# == 0 )); then cases=(off300 off80 on300); fi
for name in "${cases[@]}"; do
    case "$name" in off300|off80|on300) ;; *) echo "Unknown case: $name" >&2; exit 2 ;; esac
    if [[ -e "$root/$name" ]]; then
        echo "Output exists: $root/$name. Select remaining cases or change EXPERIMENT_ROOT." >&2
        exit 1
    fi
done
for name in "${cases[@]}"; do
    off=1
    bucket=300
    case "$name" in off80) bucket=80 ;; on300) off=0 ;; esac
    env CHECKPOINT_ROOT=./checkpoints/Animation_ipi_7ganjing CHECKPOINT_STEP=31000 \
        IPI_TEMPORAL_V2=0 NO_IPI="$off" IPI_SCALE=1 POSE_SCALE=3 \
        RETARGET=1 POSE_ROOT_MOTION_SCALE=0 MATCH_ANKLE_HEIGHT=1 \
        DRIVING_DIR=./animation_data/ceshi/10002 REFERENCE_IMAGE=./basic_infer/ces/30555.jpg \
        POSE_COORDINATE_SPACE=normalized WIDTH=512 HEIGHT=512 SOURCE_TIMELINE=1 \
        FRAME_START=360 FRAME_END=380 TILE_SIZE=20 FRAMES_OVERLAP=6 NUM_INFERENCE_STEPS=20 \
        MOTION_BUCKET_ID="$bucket" FPS_TIME_ID=6 NOISE_AUG_STRENGTH=0 INDEPENDENT_NOISE=0 \
        GUIDANCE_SCALE=3 MIN_GUIDANCE_SCALE=2 MAX_GUIDANCE_SCALE=3.5 \
        REFERENCE_LATENT_SCALE=1 REFERENCE_CLIP_SCALE=1 IS_STATUE=0 SEED=23123134 \
        DECODE_CHUNK_SIZE=4 SKIP_GIF=1 OUTPUT_DIR="$root/$name" bash command_ipi_infer.sh
done