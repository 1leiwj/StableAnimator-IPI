set -e
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

output_dir="./checkpoints/Animation_ipi_7ganjing"
warmup_steps=1000
training_args=()
case "${TARGET_FRAME_POSE:-0}" in
    0) ;;
    1) training_args+=(--target_frame_pose) ;;
    *) echo "TARGET_FRAME_POSE must be 0 or 1." >&2; exit 2 ;;
esac
base_step=19500
adapter_step=29500
base_dir="/gaojiawei/wjl/StableAnimator/checkpoints/checkpoint-$base_step"
adapter_dir="/gaojiawei/wjl/StableAnimator/checkpoints/checkpoint-$adapter_step"
ipi_only="${IPI_ONLY_FINETUNE:-0}"
if [[ "${IPI_TEMPORAL_V2:-0}" == 1 || "$ipi_only" == 1 ]]; then
    output_dir="./checkpoints/Animation_ipi_temporal_v2"
    warmup_steps=100
    training_args+=(--ipi_temporal_v2 --num_frames=22)
    if [[ "$ipi_only" == 1 ]]; then
        base_step="${INIT_CHECKPOINT_STEP:-31000}"
        adapter_step="$base_step"
        base_dir="${INIT_CHECKPOINT_ROOT:-./checkpoints/Animation_ipi_7ganjing}/checkpoint-$base_step"
        adapter_dir="$base_dir"
        output_dir="./checkpoints/Animation_ipi_temporal_v2_from_$base_step"
        training_args+=(--ipi_only_finetune)
        for model in pose_net face_encoder unet ipi_adapter; do
            weight="$base_dir/$model-$base_step.pth"
            if [[ ! -s "$weight" ]]; then
                echo "Missing initial checkpoint file: $weight" >&2
                exit 1
            fi
        done
    fi
    if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
        training_args+=(--resume_from_checkpoint="$RESUME_FROM_CHECKPOINT")
        if [[ -n "${RESUME_LAST_STEP:-}" ]]; then
            training_args+=(--resume_last_step="$RESUME_LAST_STEP")
        fi
    elif compgen -G "${TRAIN_OUTPUT_DIR:-$output_dir}/checkpoint-*" > /dev/null; then
        echo "V2 checkpoints already exist. Set TRAIN_OUTPUT_DIR for a new run or RESUME_FROM_CHECKPOINT=latest to resume." >&2
        exit 1
    fi
else
    training_args+=(--resume_from_checkpoint=latest)
fi
if [[ -n "${MAX_TRAIN_STEPS:-}" ]]; then
    training_args+=(--max_train_steps="$MAX_TRAIN_STEPS")
fi

CUDA_VISIBLE_DEVICES=3,4,5 accelerate launch  \
 --use_deepspeed \
 --zero_stage=2 \
 train_ipi_ongyong.py \
 --pretrained_model_name_or_path="./checkpoints/stable-video-diffusion-img2vid-xt" \
 --finetune_mode=True \
 --posenet_model_finetune_path="$base_dir/pose_net-$base_step.pth" \
 --face_encoder_finetune_path="$base_dir/face_encoder-$base_step.pth" \
 --unet_model_finetune_path="$base_dir/unet-$base_step.pth" \
 --ipi_model_finetune_path="$adapter_dir/ipi_adapter-$adapter_step.pth" \
 --output_dir="${TRAIN_OUTPUT_DIR:-$output_dir}" \
 --data_root_path="./animation_data" \
 --rec_data_path="./animation_data/video_dec_path.txt" \
 --vec_data_path="./animation_data/video_dec_path.txt" \
 --clay_data_path="./animation_data/video_dec_path.txt" \
 --validation_image_folder="/gaojiawei/wjl/StableAnimator/basic_infer/ces/30008/images" \
 --validation_control_folder="/gaojiawei/wjl/StableAnimator/basic_infer/ces/30008/poses" \
 --validation_image="/gaojiawei/wjl/StableAnimator/basic_infer/ces/30008.jpg" \
 --pose_data_path="/gaojiawei/wjl/StableAnimator/basic_infer/ces/30008/poses_dense133.npy" \
 --use_8bit_adam \
 --num_workers=12 \
 --lr_warmup_steps="${LR_WARMUP_STEPS:-$warmup_steps}" \
 --seed=42 \
 --sample_n_frames=22 \
 --learning_rate=4e-6 \
 --per_gpu_batch_size=1 \
 --num_train_epochs=100 \
 --mixed_precision="bf16" \
 --gradient_accumulation_steps=1 \
 --checkpointing_steps=500 \
 --validation_steps=500 \
 --gradient_checkpointing \
 --checkpoints_total_limit=20 \
 "${training_args[@]}"
