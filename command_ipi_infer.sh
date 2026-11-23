#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

for setting in IPI_TEMPORAL_V2 RETARGET NO_IPI INDEPENDENT_NOISE SKIP_GIF IS_STATUE SOURCE_TIMELINE MATCH_ANKLE_HEIGHT; do
    value="${!setting-}"
    if [[ -n "$value" && "$value" != 0 && "$value" != 1 ]]; then
        echo "$setting must be 0 or 1." >&2
        exit 1
    fi
done

output_dir="${OUTPUT_DIR:-./basic_infer/aa}"
checkpoint_step="${CHECKPOINT_STEP:-24000}"
checkpoint_root="./checkpoints/Animation_ipi_7ganjing"
default_tile_size=20
default_ipi_scale=6
driving_dir="${DRIVING_DIR:-./basic_infer/ces/30008}"
inference_args=()
if [[ "${MATCH_ANKLE_HEIGHT:-0}" == 1 ]]; then
    inference_args+=(--match_ankle_height)
fi
# Preserve the historical ramp for a bare launch. An explicit GUIDANCE_SCALE
# requests a constant; either endpoint can also be overridden separately.
if [[ -z "${GUIDANCE_SCALE:-}" ]]; then
    inference_args+=(--min_guidance_scale="${MIN_GUIDANCE_SCALE:-2}" --max_guidance_scale="${MAX_GUIDANCE_SCALE:-3.5}")
else
    inference_args+=(--min_guidance_scale="${MIN_GUIDANCE_SCALE:-$GUIDANCE_SCALE}" --max_guidance_scale="${MAX_GUIDANCE_SCALE:-$GUIDANCE_SCALE}")
fi
if [[ "${IPI_TEMPORAL_V2:-0}" == 1 ]]; then
    inference_args+=(--ipi_temporal_v2)
    checkpoint_root="./checkpoints/Animation_ipi_temporal_v2"
    default_tile_size=22
    default_ipi_scale=1
fi
checkpoint_dir="${CHECKPOINT_ROOT:-$checkpoint_root}/checkpoint-${checkpoint_step}"
if [[ "${RETARGET:-1}" == 1 ]]; then
    inference_args+=(--retarget_pose)
fi
if [[ "${NO_IPI:-0}" == 1 ]]; then
    inference_args+=(--no_ipi)
fi
if [[ "${INDEPENDENT_NOISE:-0}" == 1 ]]; then
    inference_args+=(--independent_noise)
fi
if [[ "${SOURCE_TIMELINE:-0}" == 1 ]]; then
    inference_args+=(--source_timeline)
fi
if [[ -n "${FRAME_END:-}" ]]; then
    inference_args+=(--frame_end="$FRAME_END")
fi
if [[ "${SKIP_GIF:-0}" == 1 ]]; then
    inference_args+=(--skip_gif)
fi
if [[ "${IS_STATUE:-0}" == 1 ]]; then
    inference_args+=(--is_statue)
fi

command -v ffmpeg >/dev/null
if [[ -e "$output_dir/a-6-3.mp4" ]]; then
    echo "Output video already exists in $output_dir. Set a new OUTPUT_DIR." >&2
    exit 1
fi
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" python inference_latent_ar.py \
 --pretrained_model_name_or_path="./checkpoints/stable-video-diffusion-img2vid-xt" \
 --output_dir="$output_dir" \
 --validation_control_folder="$driving_dir/poses" \
 --validation_image="${REFERENCE_IMAGE:-./basic_infer/ces/30008.jpg}" \
 --driving_video_folder="$driving_dir/images" \
 --pose_data_path="$driving_dir/poses_dense133.npy" \
 --pose_coordinate_space="${POSE_COORDINATE_SPACE:-normalized}" \
 --width="${WIDTH:-512}" \
 --height="${HEIGHT:-768}" \
 --pose_scale="${POSE_SCALE:-3}" \
 --pose_root_motion_scale="${POSE_ROOT_MOTION_SCALE:-1}" \
 --reference_latent_scale="${REFERENCE_LATENT_SCALE:-1}" \
 --reference_clip_scale="${REFERENCE_CLIP_SCALE:-1}" \
 --ipi_scale="${IPI_SCALE:-$default_ipi_scale}" \
 --guidance_scale="${GUIDANCE_SCALE:-3}" \
 --motion_bucket_id="${MOTION_BUCKET_ID:-300}" \
 --fps_time_id="${FPS_TIME_ID:-6}" \
 --seed="${SEED:-23123134}" \
 --noise_aug_strength="${NOISE_AUG_STRENGTH:-0.00}" \
 --num_inference_steps="${NUM_INFERENCE_STEPS:-60}" \
 --posenet_model_name_or_path="$checkpoint_dir/pose_net-$checkpoint_step.pth" \
 --face_encoder_model_name_or_path="$checkpoint_dir/face_encoder-$checkpoint_step.pth" \
 --unet_model_name_or_path="$checkpoint_dir/unet-$checkpoint_step.pth" \
 --ipi_model_name_or_path="$checkpoint_dir/ipi_adapter-$checkpoint_step.pth" \
 --tile_size="${TILE_SIZE:-$default_tile_size}" \
 --frames_overlap="${FRAMES_OVERLAP:-6}" \
 --decode_chunk_size="${DECODE_CHUNK_SIZE:-4}" \
 --clip_batch_size="${CLIP_BATCH_SIZE:-32}" \
 --frame_start="${FRAME_START:-0}" \
 --gradient_checkpointing \
 "${inference_args[@]}"
temporary_video=$(mktemp "$output_dir/.encode-XXXXXXXX.mp4")
trap 'rm -f -- "$temporary_video"' EXIT
ffmpeg -nostdin -y -framerate 24 -i "$output_dir/animated_images/frame_%d.png" -c:v libx264 -pix_fmt yuv420p "$temporary_video"
mv -- "$temporary_video" "$output_dir/a-6-3.mp4"
