CUDA_VISIBLE_DEVICES=1 python inference_basic.py \
 --pretrained_model_name_or_path="./checkpoints/stable-video-diffusion-img2vid-xt" \
 --output_dir="./basic_infer/data/1" \
 --validation_control_folder="/gaojiawei/wjl/StableAnimator/basic_infer/ces/30555/poses" \
 --validation_image="/gaojiawei/wjl/StableAnimator/basic_infer/ces/30555.jpg" \
 --width=512 \
 --height=768 \
 --guidance_scale=3 \
 --num_inference_steps=60 \
 --posenet_model_name_or_path="/gaojiawei/wjl/StableAnimator/checkpoints/Animation_ipi12/checkpoint-39500/pose_net-39500.pth" \
 --face_encoder_model_name_or_path="/gaojiawei/wjl/StableAnimator/checkpoints/Animation_ipi12/checkpoint-39500/face_encoder-39500.pth" \
 --unet_model_name_or_path="/gaojiawei/wjl/StableAnimator/checkpoints/Animation_ipi12/checkpoint-39500/unet-39500.pth" \
 --tile_size=32 \
 --overlap=4 \
 --noise_aug_strength=0.01 \
 --frames_overlap=4 \
 --decode_chunk_size=4 \
 --gradient_checkpointing
ffmpeg -framerate 27 -i /gaojiawei/wjl/StableAnimator/basic_infer/data/1/animated_images/frame_%d.png -c:v libx264 -pix_fmt yuv420p /gaojiawei/wjl/StableAnimator/basic_infer/data/1/new-6.mp4