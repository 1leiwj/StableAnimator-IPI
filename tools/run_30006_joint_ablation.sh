#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Select one or more conditions by name. All start with the historical short
# clip's settings, including its slice-local CFG ramp and periodic noise.
all_cases=(base pose0 pose45 steps40 ref075 root0 ipi_on)
cases=("$@")
if (( $# == 0 )); then cases=("${all_cases[@]}"); fi
step="${CHECKPOINT_STEP:-6000}"
checkpoint_root="${CHECKPOINT_ROOT:-./checkpoints/Animation_ipi_temporal_v2_from_31000_fresh20k}"
[[ "$step" == last || "$step" =~ ^[0-9]+$ ]] || { echo "CHECKPOINT_STEP must be a step number or last" >&2; exit 2; }
checkpoint_dir="$checkpoint_root/checkpoint-$step"
root="${EXPERIMENT_ROOT:-./basic_infer/30006_joint_ablation_$step}"
weights=(pose_net face_encoder unet)
driver="${DRIVING_DIR:-./animation_data/cec/30006}"
if [[ -z "${DRIVING_DIR:-}" && ! -f "$driver/poses_dense133.npy" && -f "$driver/30006/poses_dense133.npy" ]]; then
    driver="$driver/30006"
fi
for name in "${cases[@]}"; do
    case "$name" in base|pose0|pose45|steps40|ref075|root0|ipi_on) ;;
        *) echo "Unknown condition $name. Choose from: ${all_cases[*]}" >&2; exit 2 ;;
    esac
    if [[ "$name" == ipi_on ]]; then weights+=(ipi_adapter); fi
    if [[ -e "$root/$name" ]]; then
        echo "Output exists: $root/$name. Select missing cases or set a new EXPERIMENT_ROOT." >&2
        exit 1
    fi
done
for model in "${weights[@]}"; do
    if [[ ! -s "$checkpoint_dir/$model-$step.pth" ]]; then
        echo "Missing weight: $checkpoint_dir/$model-$step.pth. Select an existing CHECKPOINT_STEP in $checkpoint_root." >&2
        exit 1
    fi
done
echo "Using $checkpoint_dir for all selected conditions"

for name in "${cases[@]}"; do
    pose=3; steps=20; reference=1; root_motion=1; off=1
    case "$name" in
        pose0) pose=0 ;;
        pose45) pose=4.5 ;;
        steps40) steps=40 ;;
        ref075) reference=0.75 ;;
        root0) root_motion=0 ;;
        ipi_on) off=0 ;;
    esac
    echo "Running $name"
    env CHECKPOINT_ROOT="$checkpoint_root" \
        CHECKPOINT_STEP="$step" IPI_TEMPORAL_V2=1 IPI_SCALE=1 NO_IPI="$off" \
        REFERENCE_IMAGE="${REFERENCE_IMAGE:-./basic_infer/ces/30555.jpg}" DRIVING_DIR="$driver" \
        WIDTH=510 HEIGHT=510 FRAME_START=121 FRAME_END=141 SOURCE_TIMELINE=0 \
        TILE_SIZE=20 FRAMES_OVERLAP=6 INDEPENDENT_NOISE=0 SEED=23123134 \
        GUIDANCE_SCALE=3 MIN_GUIDANCE_SCALE=2 MAX_GUIDANCE_SCALE=3.5 \
        NOISE_AUG_STRENGTH=0 MOTION_BUCKET_ID=300 FPS_TIME_ID=6 \
        DECODE_CHUNK_SIZE=4 CLIP_BATCH_SIZE=32 SKIP_GIF=1 IS_STATUE=0 \
        RETARGET=1 POSE_COORDINATE_SPACE=normalized POSE_ROOT_MOTION_SCALE="$root_motion" \
        POSE_SCALE="$pose" NUM_INFERENCE_STEPS="$steps" REFERENCE_LATENT_SCALE="$reference" \
        REFERENCE_CLIP_SCALE=1 OUTPUT_DIR="$root/$name" bash command_ipi_infer.sh
done
control=
if [[ -f "$root/base/retargeted_poses_dense133.npy" ]]; then control="$root/base/retargeted_poses_dense133.npy"; fi
for name in "${cases[@]}"; do
    if [[ "$name" == root0 ]]; then continue; fi
    file="$root/$name/retargeted_poses_dense133.npy"
    if [[ -z "$control" ]]; then control="$file"; fi
    cmp "$control" "$file"
done
