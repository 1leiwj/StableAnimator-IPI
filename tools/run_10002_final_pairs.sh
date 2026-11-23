#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

step=20000
weights=./checkpoints/Animation_ipi_temporal_v2_from_19500_base_29500_adapter_fresh20k
root="${EXPERIMENT_ROOT:-./basic_infer/10002_v2_from_19500_29500_step20000_512}"
if (( $# == 3 )); then
    weights="$1"
    step="$2"
    root="$3"
elif (( $# != 0 )); then
    echo "Usage: bash $0 [checkpoint_root checkpoint_step output_root]" >&2
    exit 2
fi
for model in pose_net face_encoder unet ipi_adapter; do
    weight="$weights/checkpoint-$step/$model-$step.pth"
    [[ -s "$weight" ]] || { echo "Missing checkpoint file: $weight" >&2; exit 1; }
done

if [[ -e "$root/raise" || -e "$root/hold" ]]; then
    echo "Output exists under $root. Set EXPERIMENT_ROOT to a fresh directory." >&2
    exit 1
fi

case "${COMPARE_JOINT_BRANCH:-0}" in
    0) control=off ;;
    1) control=global_only ;;
    *) echo "COMPARE_JOINT_BRANCH must be 0 or 1" >&2; exit 2 ;;
esac
for segment in raise hold; do
    case "$segment" in
        raise) start=307; end=327 ;;
        hold)  start=360; end=380 ;;
    esac
    for mode in "$control" on; do
        no_ipi=0
        joint_mode=auto
        case "$mode" in
            off) no_ipi=1 ;;
            global_only) joint_mode=off ;;
            on) if [[ "$control" == global_only ]]; then joint_mode=on; fi ;;
        esac
        echo "$segment / IPI $mode / source frames [$start, $end)"
        env CHECKPOINT_ROOT="$weights" CHECKPOINT_STEP="$step" \
            IPI_TEMPORAL_V2=1 NO_IPI="$no_ipi" IPI_JOINT_MODE="$joint_mode" IPI_SCALE=1 POSE_SCALE=3 \
            DRIVING_DIR=./animation_data/ceshi/10002 REFERENCE_IMAGE=./basic_infer/ces/30555.jpg \
            RETARGET=1 POSE_ROOT_MOTION_SCALE=0 MATCH_ANKLE_HEIGHT=1 POSE_COORDINATE_SPACE=normalized \
            FRAME_START="$start" FRAME_END="$end" SOURCE_TIMELINE=1 WIDTH=512 HEIGHT=512 \
            TILE_SIZE=20 FRAMES_OVERLAP=6 NUM_INFERENCE_STEPS=20 MOTION_BUCKET_ID=300 \
            GUIDANCE_SCALE=3 MIN_GUIDANCE_SCALE=2 MAX_GUIDANCE_SCALE=3.5 \
            FPS_TIME_ID=6 NOISE_AUG_STRENGTH=0 INDEPENDENT_NOISE=0 SEED=23123134 \
            REFERENCE_LATENT_SCALE=1 REFERENCE_CLIP_SCALE=1 IS_STATUE=0 \
            CLIP_BATCH_SIZE=32 DECODE_CHUNK_SIZE=4 SKIP_GIF=1 OUTPUT_DIR="$root/$segment/$mode" \
            bash command_ipi_infer.sh
    done
    cmp "$root/$segment/$control/retargeted_poses_dense133.npy" "$root/$segment/on/retargeted_poses_dense133.npy"
done
echo "Finished both comparisons in $root"
