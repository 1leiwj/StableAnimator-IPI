#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
inference_ipi2.py
=================
以原版 inference_basic.py 为基础，最小化地添加 IPI Adapter 支持。
只做三件额外的事：
  1. 加载 IPI Adapter 权重并挂载到 UNet
  2. 加载骨架 npy → dwpose_raw_tensor
  3. 提取驱动视频 CLIP 特征 → clip_image_embeds
  4. 以上两者传入 pipeline

是否启用雕像模式（切断 FaceEncoder）用 --is_statue 控制，逻辑极简。
"""

import argparse
import os
import cv2
import numpy as np
from PIL import Image
from diffusers.models.attention_processor import XFormersAttnProcessor
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import torch
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
import random

from animation.modules.attention_processor import AnimationAttnProcessor
from animation.modules.attention_processor_normalized import AnimationIDAttnNormalizedProcessor
from animation.modules.face_model import FaceModel
from animation.modules.id_encoder import FusionFaceId
from animation.modules.pose_net import PoseNet
from animation.modules.unet_ipi import UNetSpatioTemporalConditionModel
from animation.pipelines.inference_pipeline_animation_ipi import InferenceAnimationPipeline
from animation.modules.ipi_adapter import MotionAdapter


# ──────────────────────────────────────────────
# 工具函数（与原版保持一致）
# ──────────────────────────────────────────────

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)


def load_images_from_folder(folder, width, height):
    files = os.listdir(folder)
    png_files = [f for f in files if f.endswith('.png')]
    png_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    images = []
    for filename in png_files:
        img = Image.open(os.path.join(folder, filename)).convert('RGB')
        img = img.resize((width, height))
        images.append(img)
    return images


def save_frames_as_png(frames, output_path):
    pil_frames = [Image.fromarray(f) if isinstance(f, np.ndarray) else f for f in frames]
    for i, frame in enumerate(pil_frames):
        frame.save(os.path.join(output_path, f'frame_{i}.png'))


def export_to_gif(frames, output_gif_path, fps):
    pil_frames = [Image.fromarray(f) if isinstance(f, np.ndarray) else f for f in frames]
    pil_frames[0].save(
        output_gif_path.replace('.mp4', '.gif'),
        format='GIF',
        append_images=pil_frames[1:],
        save_all=True,
        duration=125,
        loop=0,
    )


# ──────────────────────────────────────────────
# 参数解析
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()

    # ── 基础参数（与原版相同） ──
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--validation_image", type=str, default=None)
    parser.add_argument("--validation_control_folder", type=str, default=None)
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
    parser.add_argument("--motion_bucket_id", type=int, default=127)

    # ── IPI 新增参数 ──
    parser.add_argument(
        "--ipi_model_name_or_path", type=str, default=None,
        help="IPI Adapter 权重路径（.pth）",
    )
    parser.add_argument(
        "--pose_data_path", type=str, default=None,
        help="骨架文件路径，必须是 .npy 格式，且 shape=[T, 133, 2]（133点全身骨架）",
    )
    parser.add_argument(
        "--driving_video_folder", type=str, default=None,
        help="驱动视频帧文件夹（RGB 原图，用于提取 CLIP 特征）",
    )
    parser.add_argument(
        "--is_statue", action="store_true",
        help="雕像模式：切断 FaceEncoder（强制全 0 输出）",
    )

    # ── 消融实验开关 ──
    parser.add_argument(
        "--no_ipi", action="store_true",
        help="【消融 A】关闭 IPI：dwpose_raw 和 clip_image_embeds 均置 None，只靠 PoseNet 驱动",
    )
    parser.add_argument(
        "--no_posenet", action="store_true",
        help="【消融 B】关闭 PoseNet：将 PoseNet 输出强制清零，只靠 IPI 驱动",
    )
    parser.add_argument(
        "--no_face_encoder", action="store_true",
        help="【消融 C】关闭 FaceEncoder：将 FaceEncoder 输出强制替换为全 0，不附带其他雕像逻辑",
    )

    return parser.parse_args()


# ──────────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    seed = 23123134
    seed_everything(seed)
    generator = torch.Generator(device='cuda').manual_seed(seed)

    # ── 1. 加载基础模型（与原版完全一致）──
    feature_extractor = CLIPImageProcessor.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="feature_extractor", revision=args.revision)
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="image_encoder", revision=args.revision)
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)

    # 使用 IPI 版 UNet（比原版多了 ipi_adapter 挂载点）
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True,
    )

    # ── 2. 挂载并加载 IPI Adapter ──
    ipi_adapter = MotionAdapter(
        cross_attention_dim=getattr(unet.config, "cross_attention_dim", 1024),
        clip_dim=1024,
        num_tokens=8,
    )
    unet.ipi_adapter = ipi_adapter

    if args.ipi_model_name_or_path and os.path.exists(args.ipi_model_name_or_path):
        print(f"✅ 加载 IPI Adapter 权重: {args.ipi_model_name_or_path}")
        ipi_state = torch.load(args.ipi_model_name_or_path, map_location="cpu")
        unet.ipi_adapter.load_state_dict(ipi_state, strict=True)
    else:
        print("⚠️ 未提供 IPI Adapter 路径，将使用零初始化权重（IPI 无效果）")

    # ── 3. PoseNet / FaceEncoder（与原版一致）──
    pose_net = PoseNet(noise_latent_channels=unet.config.block_out_channels[0])
    face_encoder = FusionFaceId(
        cross_attention_dim=1024,
        id_embeddings_dim=512,
        clip_embeddings_dim=1024,
        num_tokens=4,
    )
    face_model = FaceModel()

    # ── 4. 注意力处理器设置（与原版完全一致）──
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
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank)
            else:
                layer_name = name.split(".processor")[0]
                weights = {
                    "id_to_k.weight": unet_svd[layer_name + ".to_k.weight"],
                    "id_to_v.weight": unet_svd[layer_name + ".to_v.weight"],
                }
                attn_procs[name] = AnimationIDAttnNormalizedProcessor(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank)
                attn_procs[name].load_state_dict(weights, strict=False)
        elif "temporal_transformer_blocks" in name:
            attn_procs[name] = XFormersAttnProcessor()

    unet.set_attn_processor(attn_procs)

    # ── 5. 加载训练好的权重（与原版逻辑一致，只是 UNet 允许 strict=False）──
    if args.posenet_model_name_or_path and args.face_encoder_model_name_or_path and args.unet_model_name_or_path:
        print("Loading posenet / face_encoder / unet weights...")

        assert args.posenet_model_name_or_path.endswith(".pth"), "posenet path must be .pth"
        pose_net.load_state_dict(
            torch.load(args.posenet_model_name_or_path, map_location="cpu"), strict=True)

        assert args.face_encoder_model_name_or_path.endswith(".pth"), "face_encoder path must be .pth"
        face_encoder.load_state_dict(
            torch.load(args.face_encoder_model_name_or_path, map_location="cpu"), strict=True)

        assert args.unet_model_name_or_path.endswith(".pth"), "unet path must be .pth"
        unet_state = torch.load(args.unet_model_name_or_path, map_location="cpu")
        missing, unexpected = unet.load_state_dict(unet_state, strict=False)
        print(f"  UNet loaded: missing={len(missing)}, unexpected={len(unexpected)}")

    torch.cuda.empty_cache()
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    pose_net.requires_grad_(False)
    face_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    weight_dtype = torch.bfloat16

    # ── 6. 雕像模式：切断 FaceEncoder ──
    if args.is_statue:
        print("🗿 雕像模式：FaceEncoder 输出将被强制替换为全 0")
        import types
        def _statue_forward(self, faceid_embeds, image_embed, *a, **kw):
            return torch.zeros(
                image_embed.shape[0], 4, image_embed.shape[-1],
                device=image_embed.device, dtype=image_embed.dtype)
        face_encoder.forward = types.MethodType(_statue_forward, face_encoder)

    # ── 消融 B：关闭 PoseNet（输出清零）──
    if args.no_posenet:
        print("🔬 [消融 B] --no_posenet：PoseNet 输出将被强制清零，只靠 IPI 驱动")
        import types
        _orig_posenet_fwd = pose_net.forward
        def _zeroed_posenet_forward(self, x):
            out = _orig_posenet_fwd(x)
            return torch.zeros_like(out)
        pose_net.forward = types.MethodType(_zeroed_posenet_forward, pose_net)

    # ── 消融 C：单独关闭 FaceEncoder（不附带其他雕像逻辑）──
    if args.no_face_encoder:
        print("🔬 [消融 C] --no_face_encoder：FaceEncoder 输出将被强制替换为全 0")
        import types
        def _zeroed_face_encoder_fwd(self, faceid_embeds, image_embed, *a, **kw):
            return torch.zeros(
                image_embed.shape[0], 4, image_embed.shape[-1],
                device=image_embed.device, dtype=image_embed.dtype)
        face_encoder.forward = types.MethodType(_zeroed_face_encoder_fwd, face_encoder)

    # ── 7. 构建 Pipeline ──
    pipeline = InferenceAnimationPipeline(
        vae=vae,
        image_encoder=image_encoder,
        unet=unet,
        scheduler=noise_scheduler,
        feature_extractor=feature_extractor,
        pose_net=pose_net,
        face_encoder=face_encoder,
    ).to(device='cuda', dtype=weight_dtype)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 8. 准备输入数据 ──
    validation_image = Image.open(args.validation_image).convert('RGB')
    validation_control_images = load_images_from_folder(
        args.validation_control_folder, width=args.width, height=args.height)
    num_frames = len(validation_control_images)

    # 人脸 embedding（雕像模式下全 0，真人走正常检测流程）
    if args.is_statue:
        validation_image_id_ante_embedding = np.zeros((512,))
    else:
        face_model.face_helper.clean_all()
        val_face_bgr = cv2.imread(args.validation_image)
        val_face_info = face_model.app.get(val_face_bgr)
        if len(val_face_info) > 0:
            val_face_info = sorted(val_face_info,
                key=lambda x: (x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
            validation_image_id_ante_embedding = val_face_info['embedding']
        else:
            face_model.face_helper.read_image(val_face_bgr)
            face_model.face_helper.get_face_landmarks_5(only_center_face=True)
            face_model.face_helper.align_warp_face()
            if len(face_model.face_helper.cropped_faces) == 0:
                validation_image_id_ante_embedding = np.zeros((512,))
            else:
                validation_image_id_ante_embedding = face_model.handler_ante.get_feat(
                    face_model.face_helper.cropped_faces[0])

    # ── 9. IPI 数据准备（骨架 npy + 驱动视频 CLIP）──
    dwpose_raw_tensor = None
    clip_image_embeds = None

    if args.pose_data_path and os.path.exists(args.pose_data_path):
        dwpose_raw_np = np.load(args.pose_data_path)          # [T, 133, 2]
        assert dwpose_raw_np.ndim == 3 and dwpose_raw_np.shape[1:] == (133, 2), \
            f"pose_data shape error: {dwpose_raw_np.shape}"
        dwpose_raw_np = dwpose_raw_np[:num_frames]

        # ✅ [修复] 与数据集 normalize_pose 函数完全对齐
        # 找到有效点 (非 -1)
        valid_mask_2d = (dwpose_raw_np[..., 0] != -1) & (dwpose_raw_np[..., 1] != -1)
        valid_vals = dwpose_raw_np[valid_mask_2d]
        if len(valid_vals) > 0:
            if valid_vals.max() > 1.5:
                # 像素坐标（如 0~512）→ 先除以尺寸再映射到 [-1, 1]
                W, H = args.width, args.height
                dwpose_raw_np[valid_mask_2d, 0] = (dwpose_raw_np[valid_mask_2d, 0] / W) * 2.0 - 1.0
                dwpose_raw_np[valid_mask_2d, 1] = (dwpose_raw_np[valid_mask_2d, 1] / H) * 2.0 - 1.0
            else:
                # 已是 [0, 1] 坐标→ 直接映射到 [-1, 1]
                dwpose_raw_np[valid_mask_2d] = dwpose_raw_np[valid_mask_2d] * 2.0 - 1.0
        # 无效点保持 -1

        dwpose_raw_tensor = torch.from_numpy(dwpose_raw_np).float() \
            .view(1, num_frames, -1).to(device='cuda', dtype=weight_dtype)
        print(f"✅ 骨架数据加载完成: shape={dwpose_raw_np.shape}, "
              f"min={dwpose_raw_np.min():.3f}, max={dwpose_raw_np.max():.3f}")
    else:
        print("⚠️ 未提供 pose_data_path，IPI 骨架信号将为 None")

    if args.driving_video_folder and os.path.exists(args.driving_video_folder):
        driving_pil = load_images_from_folder(
            args.driving_video_folder, width=args.width, height=args.height)
        clip_inputs = feature_extractor(images=driving_pil, return_tensors="pt")
        driving_pixels = clip_inputs.pixel_values.to(device='cuda', dtype=weight_dtype)

        # 雕像模式：对驱动视频做灰度+模糊，防止真人外观泄漏
        if args.is_statue:
            import torchvision.transforms as T
            driving_pixels = T.GaussianBlur(kernel_size=(3, 3), sigma=(1.0, 1.0))(T.Grayscale(3)(driving_pixels))

        with torch.no_grad():
            clip_image_embeds = image_encoder(driving_pixels).image_embeds.unsqueeze(0)
        print(f"✅ CLIP 特征提取完成: shape={clip_image_embeds.shape}")
    else:
        print("⚠️ 未提供 driving_video_folder，IPI CLIP 信号将为 None")

    # ── 10. 推理（与原版 pipeline 调用方式保持一致，只多传两个 IPI 参数）──
    video_frames = pipeline(
        image=validation_image,
        image_pose=validation_control_images,
        height=args.height,
        width=args.width,
        num_frames=num_frames,
        tile_size=args.tile_size,
        tile_overlap=args.frames_overlap,
        decode_chunk_size=args.decode_chunk_size,
        motion_bucket_id=float(args.motion_bucket_id),
        fps=7,
        min_guidance_scale=args.guidance_scale,
        max_guidance_scale=args.guidance_scale,
        noise_aug_strength=args.noise_aug_strength,
        num_inference_steps=args.num_inference_steps,
        generator=generator,
        output_type="pil",
        validation_image_id_ante_embedding=validation_image_id_ante_embedding,
        is_statue=args.is_statue,
        # IPI 两路信号
        # 消融 A（--no_ipi）：两路全部切断，只靠 PoseNet
        # 消融 B（--no_posenet）：PoseNet 输出已被清零，IPI 正常传入
        dwpose_raw=None if args.no_ipi else dwpose_raw_tensor,
        clip_image_embeds=None if args.no_ipi else clip_image_embeds,
    ).frames[0]

    # ── 11. 保存结果（与原版一致）──
    out_file = os.path.join(args.output_dir, "animation_video.mp4")
    for i in range(num_frames):
        video_frames[i] = np.array(video_frames[i])

    png_out_dir = os.path.join(args.output_dir, "animated_images")
    os.makedirs(png_out_dir, exist_ok=True)
    export_to_gif(video_frames, out_file, 8)
    save_frames_as_png(video_frames, png_out_dir)
    print(f"✅ 推理完成！结果保存在: {args.output_dir}")
