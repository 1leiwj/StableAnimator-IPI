
BASE_ARGS="--pretrained_model_name_or_path=./checkpoints/stable-video-diffusion-img2vid-xt \
 --validation_control_folder="/gaojiawei/wjl/StableAnimator/basic_infer/ces/30008/poses" \
 --validation_image="/gaojiawei/wjl/StableAnimator/basic_infer/ces/30008.jpg" \
 --driving_video_folder="/gaojiawei/wjl/StableAnimator/basic_infer/ces/30008/images" \
 --pose_data_path="/gaojiawei/wjl/StableAnimator/basic_infer/ces/30008/poses_dense133.npy" \
 --width=512 \
 --height=768 \
 --guidance_scale=3.5 \
 --motion_bucket_id=300 \
 --num_inference_steps=20 \
 --posenet_model_name_or_path="/gaojiawei/wjl/StableAnimator/checkpoints/Animation_ipi_7ganjing/checkpoint-32000/pose_net-32000.pth" \
 --face_encoder_model_name_or_path="/gaojiawei/wjl/StableAnimator/checkpoints/Animation_ipi_7ganjing/checkpoint-32000/face_encoder-32000.pth" \
 --unet_model_name_or_path="/gaojiawei/wjl/StableAnimator/checkpoints/Animation_ipi_7ganjing/checkpoint-32000/unet-32000.pth" \
 --ipi_model_name_or_path="/gaojiawei/wjl/StableAnimator/checkpoints/Animation_ipi_7ganjing/checkpoint-32000/ipi_adapter-32000.pth" \
 --tile_size=22 \
 --overlap=6 \
 --noise_aug_strength=0.001 \
 --frames_overlap=6 \
 --decode_chunk_size=6 \
 --gradient_checkpointing"

echo "=========================================="
echo "执行 Group A: 纯 IPI (作为 Baseline 对照)"
echo "=========================================="

CUDA_VISIBLE_DEVICES=1 python inference_latent_ar.py \
 $BASE_ARGS \
 --output_dir="./basic_infer/opganjing32000/GroupA_Pure_IPI"
 
ffmpeg -y -framerate 24 -i ./basic_infer/opganjing32000/GroupA_Pure_IPI/animated_images/frame_%d.png -c:v libx264 -pix_fmt yuv420p ./basic_infer/opganjing32000/GroupA_Pure_IPI_result.mp4

echo "=========================================="
echo "执行 Group B: IPI + 全程 CLIP-HJB 优化 (0~25步)"
echo "=========================================="
CUDA_VISIBLE_DEVICES=1 python inference_ipi_op_clip.py \
 $BASE_ARGS \
 --output_dir="./basic_infer/opganjing32000/GroupB_Full_CLIP_HJB" \
 --num_optimization_iter=3 \
 --start_refine_step=0 \
 --end_refine_step=25
 
ffmpeg -y -framerate 24 -i ./basic_infer/opganjing32000/GroupB_Full_CLIP_HJB/animated_images/frame_%d.png -c:v libx264 -pix_fmt yuv420p ./basic_infer/opganjing32000/GroupB_Full_CLIP_HJB_result.mp4


echo "=========================================="
echo "执行 Group C: IPI + 前段截断 CLIP-HJB 优化 (0~15步) [最佳方案]"
echo "=========================================="
CUDA_VISIBLE_DEVICES=1 python inference_ipi_op_clip.py \
 $BASE_ARGS \
 --output_dir="./basic_infer/opganjing32000/GroupC_Gated_CLIP_HJB" \
 --num_optimization_iter=3 \
 --start_refine_step=0 \
 --end_refine_step=15
 
ffmpeg -y -framerate 24 -i ./basic_infer/opganjing32000/GroupC_Gated_CLIP_HJB/animated_images/frame_%d.png -c:v libx264 -pix_fmt yuv420p ./basic_infer/opganjing32000/GroupC_Gated_CLIP_HJB_result.mp4

echo "所有消融实验测试完成！结果保存在 ./basic_infer/opganjing32000/ 目录下。"
