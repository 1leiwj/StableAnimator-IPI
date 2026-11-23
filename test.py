import argparse
import os
import cv2
import numpy as np
from PIL import Image
from diffusers.models.attention_processor import XFormersAttnProcessor
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import torch
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler

from animation.modules.attention_processor import AnimationAttnProcessor
from animation.modules.attention_processor_normalized import AnimationIDAttnNormalizedProcessor
from animation.modules.face_model import FaceModel
from animation.modules.id_encoder import FusionFaceId
from animation.modules.pose_net import PoseNet
from animation.modules.unet import UNetSpatioTemporalConditionModel
from animation.pipelines.inference_pipeline_animation import InferenceAnimationPipeline
import random

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)

def load_images_from_folder(folder, width, height):
    images = []
    files = os.listdir(folder)
    png_files = [f for f in files if f.endswith('.png')]
    png_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    for filename in png_files:
        img = Image.open(os.path.join(folder, filename)).convert('RGB')
        img = img.resize((width, height))
        images.append(img)
    return images

def save_frames_as_png(frames, output_path):
    pil_frames = [Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame for frame in frames]
    num_frames = len(pil_frames)
    for i in range(num_frames):
        pil_frame = pil_frames[i]
        save_path = os.path.join(output_path, f'frame_{i}.png')
        pil_frame.save(save_path)

def save_frames_as_mp4(frames, output_mp4_path, fps):
    print("Starting saving the frames as mp4")
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_mp4_path, fourcc, fps, (width, height))
    for frame in frames:
        frame_bgr = frame if frame.shape[2] == 3 else cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()

def export_to_gif(frames, output_gif_path, fps):
    pil_frames = [Image.fromarray(frame) if isinstance(
        frame, np.ndarray) else frame for frame in frames]
    pil_frames[0].save(output_gif_path.replace('.mp4', '.gif'),
                       format='GIF',
                       append_images=pil_frames[1:],
                       save_all=True,
                       duration=125,
                       loop=0)

def parse_args():
    parser = argparse.ArgumentParser(description="Script for animation inference with long video support")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--validation_image", type=str, required=True)
    parser.add_argument("--validation_control_folder", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--posenet_model_name_or_path", type=str, default=None)
    parser.add_argument("--face_encoder_model_name_or_path", type=str, default=None)
    parser.add_argument("--unet_model_name_or_path", type=str, default=None)
    parser.add_argument("--tile_size", type=int, default=16)
    parser.add_argument("--overlap", type=int, default=4)
    parser.add_argument("--noise_aug_strength", type=float, default=0.0)
    parser.add_argument("--frames_overlap", type=int, default=4)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--decode_chunk_size", type=int, default=None)
    parser.add_argument("--time_chunk_size", type=int, default=32, help="时间维度分块大小（默认32帧）")
    return parser.parse_args()

def get_dynamic_tile_size(height, width, max_tiles=16):
    min_dim = min(height, width)
    tile_size = max(8, min_dim // int(max_tiles **0.5))
    tile_size -= tile_size % 8  # 确保为8的倍数
    return tile_size

if __name__ == "__main__":
    args = parse_args()

    # 动态调整空间分块大小
    if args.tile_size == 16:
        args.tile_size = get_dynamic_tile_size(args.height, args.width)
        print(f"动态空间分块大小: {args.tile_size}")
    
    # 固定随机种子
    seed = 23123134
    seed_everything(seed)
    generator = torch.Generator(device='cuda').manual_seed(seed)

    # 加载模型组件
    feature_extractor = CLIPImageProcessor.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="feature_extractor", revision=args.revision
    )
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="image_encoder", revision=args.revision
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
    )
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", low_cpu_mem_usage=True
    )
    pose_net = PoseNet(noise_latent_channels=unet.config.block_out_channels[0])
    face_encoder = FusionFaceId(
        cross_attention_dim=1024,
        id_embeddings_dim=512,
        clip_embeddings_dim=1024,
        num_tokens=4,
    )
    face_model = FaceModel()

    # 配置UNet注意力处理器
    lora_rank = 128
    attn_procs = {}
    unet_svd = unet.state_dict()
    for name in unet.attn_processors.keys():
        if "transformer_blocks" in name and "temporal_transformer_blocks" not in name:
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            
            if cross_attention_dim is None:
                attn_procs[name] = AnimationAttnProcessor(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank
                )
            else:
                layer_name = name.split(".processor")[0]
                weights = {
                    "id_to_k.weight": unet_svd[layer_name + ".to_k.weight"],
                    "id_to_v.weight": unet_svd[layer_name + ".to_v.weight"],
                }
                attn_procs[name] = AnimationIDAttnNormalizedProcessor(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank
                )
                attn_procs[name].load_state_dict(weights, strict=False)
        elif "temporal_transformer_blocks" in name:
            attn_procs[name] = XFormersAttnProcessor()
    unet.set_attn_processor(attn_procs)

    # 加载预训练权重
    if args.posenet_model_name_or_path and args.face_encoder_model_name_or_path and args.unet_model_name_or_path:
        print("加载预训练权重...")
        if args.posenet_model_name_or_path.endswith(".pth"):
            pose_net.load_state_dict(torch.load(args.posenet_model_name_or_path, map_location="cpu"), strict=True)
        if args.face_encoder_model_name_or_path.endswith(".pth"):
            face_encoder.load_state_dict(torch.load(args.face_encoder_model_name_or_path, map_location="cpu"), strict=True)
        if args.unet_model_name_or_path.endswith(".pth"):
            unet.load_state_dict(torch.load(args.unet_model_name_or_path, map_location="cpu"), strict=True)

    # 冻结模型参数
    torch.cuda.empty_cache()
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    pose_net.requires_grad_(False)
    face_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # 初始化推理管道
    weight_dtype = torch.float16
    pipeline = InferenceAnimationPipeline(
        vae=vae,
        image_encoder=image_encoder,
        unet=unet,
        scheduler=noise_scheduler,
        feature_extractor=feature_extractor,
        pose_net=pose_net,
        face_encoder=face_encoder,
    ).to(device='cuda', dtype=weight_dtype)

    # 准备输入数据
    os.makedirs(args.output_dir, exist_ok=True)
    validation_image = Image.open(args.validation_image).convert('RGB')
    validation_control_images = load_images_from_folder(
        args.validation_control_folder, width=args.width, height=args.height
    )
    num_total_frames = len(validation_control_images)
    print(f"总帧数: {num_total_frames}")

    # 提取人脸特征
    face_model.face_helper.clean_all()
    validation_face = cv2.imread(args.validation_image)
    validation_image_bgr = cv2.cvtColor(validation_face, cv2.COLOR_RGB2BGR)
    validation_image_face_info = face_model.app.get(validation_image_bgr)
    
    if len(validation_image_face_info) > 0:
        validation_image_face_info = sorted(
            validation_image_face_info, 
            key=lambda x: (x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1])
        )[-1]
        validation_image_id_ante_embedding = validation_image_face_info['embedding']
    else:
        validation_image_id_ante_embedding = None

    if validation_image_id_ante_embedding is None:
        face_model.face_helper.read_image(validation_image_bgr)
        face_model.face_helper.get_face_landmarks_5(only_center_face=True)
        face_model.face_helper.align_warp_face()
        if len(face_model.face_helper.cropped_faces) == 0:
            validation_image_id_ante_embedding = np.zeros((512,))
        else:
            validation_image_align_face = face_model.face_helper.cropped_faces[0]
            validation_image_id_ante_embedding = face_model.handler_ante.get_feat(validation_image_align_face)

    # 时间分块处理
    time_chunk_size = args.time_chunk_size
    time_overlap = max(2, time_chunk_size // 8)  # 重叠帧数为块大小的1/8（至少2帧）
    time_chunks = []
    
    # 拆分控制帧为时间块
    for i in range(0, num_total_frames, time_chunk_size - time_overlap):
        end = min(i + time_chunk_size, num_total_frames)
        time_chunks.append(validation_control_images[i:end])

    # 逐块推理并拼接
    full_video_frames = []
    for chunk_idx, control_chunk in enumerate(time_chunks):
        print(f"推理时间块 {chunk_idx + 1}/{len(time_chunks)}，帧数: {len(control_chunk)}")
        
        # 动态调整当前块的解码分块大小
        adaptive_decode_chunk = args.decode_chunk_size or max(2, len(control_chunk) // 2)
        
        # 推理当前块
        chunk_frames = pipeline(
            image=validation_image,
            image_pose=control_chunk,
            height=args.height,
            width=args.width,
            num_frames=len(control_chunk),
            tile_size=args.tile_size,
            tile_overlap=args.frames_overlap,
            decode_chunk_size=adaptive_decode_chunk,
            motion_bucket_id=127.,
            fps=7,
            min_guidance_scale=args.guidance_scale,
            max_guidance_scale=args.guidance_scale,
            noise_aug_strength=args.noise_aug_strength,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
            output_type="pil",
            validation_image_id_ante_embedding=validation_image_id_ante_embedding,
        ).frames[0]

        # 拼接帧（去除重叠部分）
        if chunk_idx == 0:
            full_video_frames.extend(chunk_frames)
        else:
            full_video_frames.extend(chunk_frames[time_overlap:])

    # 保存结果
    video_frames = [np.array(frame) for frame in full_video_frames]
    out_file = os.path.join(args.output_dir, "animation_video.mp4")
    png_out_file = os.path.join(args.output_dir, "animated_images")
    os.makedirs(png_out_file, exist_ok=True)
    
    export_to_gif(video_frames, out_file, 8)
    save_frames_as_png(video_frames, png_out_file)
    save_frames_as_mp4(video_frames, out_file, 8)  # 补充MP4保存
    print(f"结果保存至: {args.output_dir}")