#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import sys
import cv2
import numpy as np
from PIL import Image
from diffusers.models.attention_processor import XFormersAttnProcessor
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import torch
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler

# 🚀 [魏总专属路径补丁] 修复 DWPose 内部引用死锁
# 因为 DWPose 内部使用了 import pose.script.util，必须让 Python 知道 DWPose/preprocess 是根目录之一
current_dir = os.path.dirname(os.path.abspath(__file__))
dwpose_path = os.path.join(current_dir, "DWPose/preprocess")
if dwpose_path not in sys.path:
    sys.path.insert(0, dwpose_path)

from animation.modules.attention_processor import AnimationAttnProcessor
from animation.modules.attention_processor_normalized import AnimationIDAttnNormalizedProcessor
from animation.modules.face_model import FaceModel
from animation.modules.id_encoder import FusionFaceId
from animation.modules.pose_net import PoseNet
from animation.modules.unet_ipi import UNetSpatioTemporalConditionModel
from animation.pipelines.inference_pipeline_animation_ipi import InferenceAnimationPipeline
import random

from animation.modules.ipi_adapter import MotionAdapter

def normalize_pose(pose_data, width, height):
    """
    智能归一化：自动识别绝对像素坐标和 [0,1] 坐标，统一映射到 [-1, 1]
    与 training dataset 逻辑保持 100% 一致。
    """
    pose_data = pose_data.copy()
    
    # 找到有效点 (非-1)
    valid_mask = (pose_data[..., 0] != -1) & (pose_data[..., 1] != -1)
    
    valid_poses = pose_data[valid_mask]
    if len(valid_poses) > 0:
        # 如果最大值大于 1.5，说明输入的是绝对像素坐标
        if valid_poses.max() > 1.5:
            pose_data[valid_mask, 0] = (pose_data[valid_mask, 0] / width) * 2.0 - 1.0
            pose_data[valid_mask, 1] = (pose_data[valid_mask, 1] / height) * 2.0 - 1.0
        # 如果最大值在 0~1.5 之间，说明是 [0, 1] 归一化坐标
        else:
            pose_data[valid_mask, 0] = pose_data[valid_mask, 0] * 2.0 - 1.0
            pose_data[valid_mask, 1] = pose_data[valid_mask, 1] * 2.0 - 1.0
            
    return pose_data

def sanitize_pose_data(pose_data):
    pose_data = pose_data.copy()
    num_frames = pose_data.shape[0]
    num_kps = pose_data.shape[1]
    nan_inf_mask = np.isnan(pose_data) | np.isinf(pose_data)
    if nan_inf_mask.any(): pose_data[nan_inf_mask] = -1.0
    for f in range(num_frames):
        frame = pose_data[f]
        valid = (frame[..., 0] != -1) & (frame[..., 1] != -1) & ~((frame[..., 0] == 0) & (frame[..., 1] == 0))
        if valid.sum() < 3:
            if f > 0: pose_data[f] = pose_data[f - 1].copy()
            elif f < num_frames - 1: pose_data[f] = pose_data[f + 1].copy()
    for f in range(1, num_frames):
        for kp in range(num_kps):
            x, y = pose_data[f, kp]
            if x == -1 or y == -1 or (x == 0 and y == 0):
                prev_x, prev_y = pose_data[f - 1, kp]
                if prev_x != -1 and prev_y != -1:
                    pose_data[f, kp] = pose_data[f - 1, kp].copy()
    return pose_data

def compute_body_scales(ref_pose, drv_pose):
    def dist(p1, p2):
        if p1[0]==-1 or p2[0]==-1: return -1.0
        return np.linalg.norm(p1 - p2)
    def safe_dist(ref, drv, p_idx1, p_idx2):
        d_ref = dist(ref[p_idx1], ref[p_idx2])
        d_drv = dist(drv[p_idx1], drv[p_idx2])
        if d_ref > 0 and d_drv > 0.01: return d_ref / d_drv
        return 1.0

    scales = {}
    scales["shoulder"] = safe_dist(ref_pose, drv_pose, 2, 5)
    scales["body"] = safe_dist(ref_pose, drv_pose, 1, 8)
    
    s_arm_u_l, s_arm_u_r = safe_dist(ref_pose, drv_pose, 5, 6), safe_dist(ref_pose, drv_pose, 2, 3)
    scales["arm_upper"] = (s_arm_u_l + s_arm_u_r) / 2.0 if (s_arm_u_l != 1.0 and s_arm_u_r != 1.0) else max(s_arm_u_l, s_arm_u_r)

    s_arm_l_l, s_arm_l_r = safe_dist(ref_pose, drv_pose, 6, 7), safe_dist(ref_pose, drv_pose, 3, 4)
    scales["arm_lower"] = (s_arm_l_l + s_arm_l_r) / 2.0 if (s_arm_l_l != 1.0 and s_arm_l_r != 1.0) else max(s_arm_l_l, s_arm_l_r)

    s_leg_u_l, s_leg_u_r = safe_dist(ref_pose, drv_pose, 11, 12), safe_dist(ref_pose, drv_pose, 8, 9)
    scales["leg_upper"] = (s_leg_u_l + s_leg_u_r) / 2.0 if (s_leg_u_l != 1.0 and s_leg_u_r != 1.0) else max(s_leg_u_l, s_leg_u_r)

    s_leg_l_l, s_leg_l_r = safe_dist(ref_pose, drv_pose, 12, 13), safe_dist(ref_pose, drv_pose, 9, 10)
    scales["leg_lower"] = (s_leg_l_l + s_leg_l_r) / 2.0 if (s_leg_l_l != 1.0 and s_leg_l_r != 1.0) else max(s_leg_l_l, s_leg_l_r)
    return scales

def align_body_pose_single(pose, scales):
    pose = pose.copy()
    valid = (pose[:, 0] != -1) & (pose[:, 1] != -1)
    def scale_kps(kps_idx, center_idx, s):
        for k in kps_idx:
            if valid[k] and valid[center_idx]:
                pose[k] = pose[center_idx] + (pose[k] - pose[center_idx]) * s
    scale_kps([2, 5], 1, scales["shoulder"])
    scale_kps([8, 11], 1, scales["body"])
    scale_kps([6], 5, scales["arm_upper"])
    scale_kps([7], 6, scales["arm_lower"])
    scale_kps([3], 2, scales["arm_upper"])
    scale_kps([4], 3, scales["arm_lower"])
    scale_kps([12], 11, scales["leg_upper"])
    scale_kps([13], 12, scales["leg_lower"])
    scale_kps([9], 8, scales["leg_upper"])
    scale_kps([10], 9, scales["leg_lower"])
    scale_kps([0, 14, 15, 16, 17], 1, scales["body"])
    return pose
    
import math
def draw_bodypose_local(canvas, candidate, subset):
    H, W, C = canvas.shape
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
               [1, 16], [16, 18], [3, 17], [6, 18]]
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index: continue
            p1_idx = int(index[0])
            p2_idx = int(index[1])
            if candidate[p1_idx, 0] == -1 or candidate[p2_idx, 0] == -1: continue
            
            Y = np.array([candidate[p1_idx, 0], candidate[p2_idx, 0]])
            X = np.array([candidate[p1_idx, 1], candidate[p2_idx, 1]])
            if np.max(candidate) <= 1.5:
                Y = Y * W
                X = X * H
            mX, mY = np.mean(X), np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(4), int(length / 2)), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, colors[i])
    canvas = (canvas * 0.6).astype(np.uint8)
    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1 or candidate[index, 0] == -1: continue
            y, x = candidate[index, 0], candidate[index, 1]
            if np.max(candidate) <= 1.5: y, x = y * W, x * H
            cv2.circle(canvas, (int(y), int(x)), 4, colors[i], thickness=-1)
    return canvas

def render_pose_frames(dwpose_raw_frames, width, height):
    subset = np.arange(18).reshape(1, 18)
    rendered_tensors = []
    for i in range(len(dwpose_raw_frames)):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas = draw_bodypose_local(canvas, dwpose_raw_frames[i], subset)
        # [-1, 1] 并转为 PIL 以便 inference 直接使用
        import torchvision.transforms.functional as TF
        rendered_tensors.append(TF.to_pil_image((canvas).astype(np.uint8)))
    return rendered_tensors

def seed_everything(seed):
    # 固定 PyTorch CPU 随机数种子
    torch.manual_seed(seed)
    # 固定 PyTorch 所有 GPU 随机数种子（多 GPU 场景下确保一致性）
    torch.cuda.manual_seed_all(seed)
    # 固定 NumPy 随机数种子（对 seed 取模避免超出 NumPy 种子范围）
    np.random.seed(seed % (2**32))
    # 固定 Python 内置 random 模块的随机数种子
    random.seed(seed)

#从指定文件夹中加载 PNG 格式的图片，并进行统一处理后返回图片列表
def load_images_from_folder(folder, width, height):
    images = []
    files = os.listdir(folder)
    # 筛选出所有PNG格式的文件
    png_files = [f for f in files if f.endswith('.png')]
    # 对文件名进行排序（假设文件名格式为 "xxx_数字.png"，提取中间数字作为排序依据）
    png_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    for filename in png_files:
        # 打开图片并转换为RGB格式
        img = Image.open(os.path.join(folder, filename)).convert('RGB')
        # 将图片调整为指定的宽高
        img = img.resize((width, height))
        # 添加到图片列表中
        images.append(img)
    return images

def save_frames_as_png(frames, output_path):
    # 将帧转换为PIL Image格式（若输入是numpy数组则转换，否则直接使用）
    pil_frames = [Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame for frame in frames]
    # 获取帧的总数
    num_frames = len(pil_frames)
    # 遍历每帧并保存为PNG文件
    for i in range(num_frames):
        pil_frame = pil_frames[i]
        # 构建保存路径，格式为 "output_path/frame_序号.png"
        save_path = os.path.join(output_path, f'frame_{i}.png')
        # 保存图片
        pil_frame.save(save_path)

def save_frames_as_mp4(frames, output_mp4_path, fps):
    print("Starting saving the frames as mp4")
    # 获取帧的高度、宽度（假设所有帧尺寸一致，以第一帧为准）
    height, width, _ = frames[0].shape
    # 定义MP4编码器（'mp4v'是常用的MP4编码，注释中提到'H264'可获得更好质量）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'H264' for better quality
    # 创建视频写入对象，参数依次为：输出路径、编码器、帧率、视频尺寸
    out = cv2.VideoWriter(output_mp4_path, fourcc, fps, (width, height))
    # 遍历所有帧并写入视频
    for frame in frames:
        # 确保帧为BGR格式（OpenCV默认使用BGR，若输入是RGB则转换）
        frame_bgr = frame if frame.shape[2] == 3 else cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    # 释放视频写入资源
    out.release()


def export_to_gif(frames, output_gif_path, fps):
    """
    Export a list of frames to a GIF.

    Args:
    - frames (list): List of frames (as numpy arrays or PIL Image objects).
    - output_gif_path (str): Path to save the output GIF.
    - duration_ms (int): Duration of each frame in milliseconds.

    """
    # Convert numpy arrays to PIL Images if needed
    # 将numpy数组转换为PIL Image（若输入是数组则转换，否则直接使用）
    pil_frames = [Image.fromarray(frame) if isinstance(
        frame, np.ndarray) else frame for frame in frames]
    # 保存为GIF：以第一帧为基础，追加后续帧，设置格式和参数
    pil_frames[0].save(output_gif_path.replace('.mp4', '.gif'),
                       format='GIF',
                       append_images=pil_frames[1:],# 追加剩余帧
                       save_all=True,  # 保存所有帧
                       duration=125, # 每帧持续时间（毫秒） 
                       loop=0)  # 0表示无限循环

def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train Stable Diffusion XL for InstructPix2Pix."
    )
    parser.add_argument(
        "--ipi_model_name_or_path",
        type=str,
        default=None,
        help="预训练的 IPI Adapter 模型路径",
    )
    parser.add_argument(
        "--pose_data_path",
        type=str,
        default=None,
        help="Path to the .npy/.pkl file containing raw dwpose data (Frames, 18, 2) or Animate-X pkl.",
    )
    parser.add_argument(
        "--motion_bucket_id",
        type=int,
        default=80,  # 可选0.02
        required=False
    )
    parser.add_argument(
        "--pose_data_is_normalized",
        action="store_true",
        help="If set, the pose data is already in [-1, 1] and will not be normalized again.",
    )

    # 预训练模型路径（必填）
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True
    )

    # 验证图像路径（输入图像）
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        help=(
            "控制图像路径，用于推理时的条件输入。"
            "需与验证提示词匹配，支持单张图像或多张图像对应多个提示词。"
        ),
    )
    
    # 验证控制图像文件夹（动作/姿态序列）
    parser.add_argument(
        "--validation_control_folder",
        type=str,
        default=None,
        help="存储验证用控制图像（如姿态序列帧/骨架图）的文件夹路径",
    )
    parser.add_argument(
        "--driving_video_folder",
        type=str,
        default=None,
        help="存储驱动视频帧（原始真实视频帧）的文件夹路径，用于CLIP运动特征。可选，如果不提供则会用control_folder作为驱动。",
    )

    # 输出目录（必填）
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=True
    )

    # 生成图像高度
    parser.add_argument(
        "--height",
        type=int,
        default=768,
        required=False
    )

    # 生成图像宽度
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        required=False
    )

    # 引导尺度（控制文本提示对生成的影响强度）
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.5,  # 原来 2.0
        required=False
    )

    # 推理步数（扩散过程迭代次数）
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=30,  # 原来 25
        required=False
    )

    # 姿态估计模型路径
    parser.add_argument(
        "--posenet_model_name_or_path",
        type=str,
        default=None,
        help="预训练姿态网络模型路径",
    )
    
    # 人脸编码器模型路径
    parser.add_argument(
        "--face_encoder_model_name_or_path",
        type=str,
        default=None,
        help="预训练人脸编码模型路径",
    )
    
    # UNet模型路径
    parser.add_argument(
        "--unet_model_name_or_path",
        type=str,
        default=None,
        help="预训练UNet模型路径",
    )

    # 分块生成尺寸（用于大图像分块处理）
    parser.add_argument(
        "--tile_size",
        type=int,
        default=32,
        required=False
    )

    # 分块重叠区域大小
    parser.add_argument(
        "--overlap",
        type=int,
        default=4,
        required=False
    )

    # 噪声增强强度（控制输入图像的噪声添加量）
    parser.add_argument(
        "--noise_aug_strength",
        type=float,
        default=0.0,  # 可选0.02
        required=False
    )
    
    # 帧重叠数量（序列帧生成时的重叠帧数，增强连贯性）
    parser.add_argument(
        "--frames_overlap",
        type=int,
        default=6,  # 原来 4
        required=False
    )
    
    # 梯度检查点（节省内存，牺牲速度）
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="启用梯度检查点，以内存换取计算速度",
    )
    
    # 模型版本（HuggingFace模型库的版本标识）
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="HuggingFace模型库中预训练模型的版本号",
    )
    
    # 解码分块大小（控制VAE解码时的分块尺寸，避免OOM）
    parser.add_argument(
        "--decode_chunk_size",
        type=int,
        default=None,
        required=False
    )

    # 🚀 新增：IPI 消融开关
    parser.add_argument(
        "--no_ipi",
        action="store_true",
        help="如果加上此参数，将强制关闭 IPI 模块的特征注入（设为全 0），用于对比 IPI 是否起作用。",
    )
    # 🚀 消融实验开关 B：关闭 PoseNet，只保留 IPI
    parser.add_argument(
        "--no_posenet",
        action="store_true",
        help="【消融实验 B】强制关闭 PoseNet 输出（将 pose_latents 置为全 0），只靠 IPI Adapter 驱动动作。",
    )

    # 🚀 新增：雕像模式开关
    parser.add_argument(
        "--is_statue",
        action="store_true",
        help="如果是雕像推理，请加上此参数。会强制跳过人脸检测并切断真人先验。",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # torch.set_default_dtype(torch.float16)
    seed = 23123134
    # seed = 42
    # seed = 123
    seed_everything(seed)
    generator = torch.Generator(device='cuda').manual_seed(seed)

    feature_extractor = CLIPImageProcessor.from_pretrained(args.pretrained_model_name_or_path, subfolder="feature_extractor", revision=args.revision)
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="image_encoder", revision=args.revision
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        low_cpu_mem_usage=False,
        device_map=None,
        ignore_mismatched_sizes=True,  # 关键：忽略缺失的 ipi_adapter 权重
    )

    # -------------------------------------------------------
    # ===[IPI 独立加载拼装逻辑 (显式路径版)] ===
    # -------------------------------------------------------
    # 1. 实例化 IPI Adapter 并挂载到 UNet
    from animation.modules.ipi_adapter import MotionAdapter
    ipi_adapter = MotionAdapter(
        cross_attention_dim=getattr(unet.config, "cross_attention_dim", 1024),
        clip_dim=1024,                 
        num_tokens=8                   
    )
    unet.ipi_adapter = ipi_adapter
    ipi_adapter.to(unet.device, dtype=unet.dtype)

    # 2. 加载纯净版 UNet 权重
    if args.unet_model_name_or_path is not None:
        print(f"Loading trained UNet weights from {args.unet_model_name_or_path}")
        unet_state = torch.load(args.unet_model_name_or_path, map_location="cpu")
        missing, unexpected = unet.load_state_dict(unet_state, strict=False)
        print(f"UNet weights loaded: missing={len(missing)}, unexpected={len(unexpected)}")

    # 3. 显式加载 IPI Adapter 权重 (使用你脚本里传入的参数！)
    if args.ipi_model_name_or_path is not None and os.path.exists(args.ipi_model_name_or_path):
        print(f"Loading trained IPI Adapter weights from {args.ipi_model_name_or_path}")
        ipi_state = torch.load(args.ipi_model_name_or_path, map_location="cpu")
        unet.ipi_adapter.load_state_dict(ipi_state, strict=True)
        print("✅ IPI Adapter 独立权重拼接成功！控制链路打通！")
    else:
        print("⚠️ 警告: 未提供 IPI 路径或路径不存在，如果是旧模型，尝试从 UNet 字典里抢救 IPI 权重...")
        if args.unet_model_name_or_path is not None:
            unet.load_state_dict(unet_state, strict=False) # 兼容没分离权重的旧模型
    # -------------------------------------------------------
    pose_net = PoseNet(noise_latent_channels=unet.config.block_out_channels[0])
    # 🔬 消融实验 B：如果 --no_posenet，用 monkey-patch 把 PoseNet 输出强制清零
    if args.no_posenet:
        print("🔬 [消融实验 B] --no_posenet 已激活：PoseNet 输出将被强制清零，只靠 IPI 驱动！")
        import types
        def zeroed_posenet_forward(self, x):
            out = self._original_forward(x)
            return torch.zeros_like(out)
        pose_net._original_forward = pose_net.forward
        pose_net.forward = types.MethodType(zeroed_posenet_forward, pose_net)
    face_encoder = FusionFaceId(
        cross_attention_dim=1024,
        id_embeddings_dim=512,
        # clip_embeddings_dim=image_encoder.config.hidden_size,
        clip_embeddings_dim=1024,
        num_tokens=4, )
    face_model = FaceModel()

    lora_rank = 128
    attn_procs = {}
    unet_svd = unet.state_dict()

    #UNet 模型的注意力处理器（attn_processors）设置不同的实现类，以适配动画生成中的空间和时间注意力计算需求
    for name in unet.attn_processors.keys():  #遍历 UNet 模型中所有注意力处理器的名称（键值）   unet.attn_processors是一个字典，存储了模型中所有注意力层的处理器实例
        # 筛选出空间注意力层（仅包含transformer_blocks，不包含temporal_transformer_blocks），这些层主要处理单帧图像内部的空间关联
        if "transformer_blocks" in name and "temporal_transformer_blocks" not in name:
            # 判断 是否name.endswith("attn1.processor")：注意力处理器名称是否以 "attn1.processor" 结尾，是则None 否则unet.config.cross_attention_dim
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):#  若注意力层属于中间块（mid_block）
                hidden_size = unet.config.block_out_channels[-1]  #从 UNet 配置的输出通道列表中取最后一个值作为隐藏层维度（hidden_size）
            elif name.startswith("up_blocks"): #上采样
                block_id = int(name[len("up_blocks.")]) #名称中提取块索引（block_id，如down_blocks.1中的1）
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id] ##从 UNet 配置的输出通道列表中按索引取对应值作为hidden_size（下采样块通道数从小到大排列）
            elif name.startswith("down_blocks"):#下采样
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]#从 UNet 配置的输出通道列表中按索引取对应值作为hidden_size（下采样块通道数从小到大排列）
            #根据 cross_attention_dim 的值来选择使用不同类型的注意力处理器
            if cross_attention_dim is None:
                # print(f"This is AnimationAttnProcessor: {name}") # 自注意力处理器
                attn_procs[name] = AnimationAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank)
            else:
                # print(f"This is AnimationIDAttnProcessor: {name}")  # 交叉注意力处理器（带ID功能）
                layer_name = name.split(".processor")[0]
                weights = {#从预训练 UNet 的权重（unet_svd）中提取原注意力层的to_k和to_v权重，用于初始化新处理器的id_to_k和id_to_v
                    "id_to_k.weight": unet_svd[layer_name + ".to_k.weight"],
                    "id_to_v.weight": unet_svd[layer_name + ".to_v.weight"],
                }
                attn_procs[name] = AnimationIDAttnNormalizedProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank)
                attn_procs[name].load_state_dict(weights, strict=False)
        elif "temporal_transformer_blocks" in name:
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:#时间注意力层统一使用XFormersAttnProcessor，这是基于 xFormers 库的高效注意力实现，优化了时间序列上的计算性能。
                attn_procs[name] = XFormersAttnProcessor()
            else:
                attn_procs[name] = XFormersAttnProcessor()
    #将构建好的注意力处理器字典（attn_procs）应用到 UNet 模型中，替换原有的注意力处理器，完成模型注意力机制的定制化配置。
    unet.set_attn_processor(attn_procs)

    # 🚀 注意力处理器设置完毕后，重新加载一次 LoRA 相关权重（attn processors 的权重）
    # ipi_adapter 的权重已经在上面一次性加载正确，不需要重新加载整个 trained_state
    # 这里只需要重加载包含 LoRA 的注意力处理器权重（id_to_k / id_to_v 等）
    if args.unet_model_name_or_path is not None:
        print(f"Reloading attn processor weights (LoRA) after set_attn_processor...")
        trained_state = torch.load(args.unet_model_name_or_path, map_location="cpu")
        # 只过滤出注意力处理器相关的 key，避免覆盖 ipi_adapter 权重
        attn_state = {k: v for k, v in trained_state.items() if ".processor." in k}
        if attn_state:
            missing, unexpected = unet.load_state_dict(attn_state, strict=False)
            print(f"Attn processor reload: missing={len(missing)}, unexpected={len(unexpected)}")
        else:
            # 若没有单独的 processor key，则全量加载（兼容旧版 checkpoint）
            missing, unexpected = unet.load_state_dict(trained_state, strict=False)
            print(f"Full reload: missing={len(missing)}, unexpected={len(unexpected)}")

    # resume the previous checkpoint 从之前保存的 checkpoint（检查点）中恢复模型权重，确保推理时使用训练好的参数
    if args.posenet_model_name_or_path is not None and args.face_encoder_model_name_or_path is not None and args.unet_model_name_or_path is not None:
        print("Loading existing posenet weights, face_encoder weights.")
        if args.posenet_model_name_or_path.endswith(".pth"):
            pose_net_state_dict = torch.load(args.posenet_model_name_or_path, map_location="cpu")
            pose_net.load_state_dict(pose_net_state_dict, strict=True)#将加载的权重字典应用到 pose_net 模型：strict=True 表示要求权重字典中的键与模型的参数名完全匹配，不允许缺失或多余的参数，确保权重加载准确
        else:
            print("posenet weights loading fail")
            print(1/0)
        if args.face_encoder_model_name_or_path.endswith(".pth"):
            face_encoder_state_dict = torch.load(args.face_encoder_model_name_or_path, map_location="cpu")
            face_encoder.load_state_dict(face_encoder_state_dict, strict=True)
        else:
            print("face_encoder weights loading fail")
            print(1/0)

    torch.cuda.empty_cache()
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    pose_net.requires_grad_(False)
    face_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # weight_dtype = torch.float16
    # weight_dtype = torch.float32
    weight_dtype = torch.bfloat16

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

    fopath = args.validation_image
    validation_image = Image.open(args.validation_image).convert('RGB')
    validation_control_images = load_images_from_folder(args.validation_control_folder, width=args.width, height=args.height)

    num_frames = len(validation_control_images)  ## 获取姿态图像序列的长度（即生成视频的帧数）

    # -------------------------------------------------------
    # === [IPI Mod] 推理数据准备 ===
    # -------------------------------------------------------
    # 1. 准备 DWPose Raw Data
    if args.pose_data_path and os.path.exists(args.pose_data_path):
        if args.pose_data_path.endswith('.pkl'):
            import pickle
            with open(args.pose_data_path, 'rb') as f:
                pose_dict = pickle.load(f)
            print(f"Loaded pose pkl: {args.pose_data_path}, keys={list(pose_dict.keys())}")
            if 'bodies' in pose_dict:
                bodies = pose_dict['bodies']
                # animate-x 保存 bodies 是 [T, 18, 2] (0-1 normalized) 或 [T, 20,2]
                if isinstance(bodies, dict):
                    # 兼容可能是字典结构
                    bodies = bodies.get('candidate', None)
                if bodies is None:
                    raise ValueError('Cannot find pose bodies in pkl.')
                if bodies.ndim == 3 and bodies.shape[1] >= 18:
                    # 选择前 18 关键点
                    dwpose_raw_np = np.array(bodies[:, :18, :], dtype=np.float32)
                else:
                    raise ValueError(f"Unsupported bodies shape in pkl: {bodies.shape}")
            elif isinstance(pose_dict, np.ndarray):
                dwpose_raw_np = pose_dict
            else:
                raise ValueError('Unsupported pkl format for pose data.')

            # 这里根据值域判断：0-1 => 像素 -> 归一化；-1~1 或近似 => 直接使用
            if dwpose_raw_np.max() <= 1.01 and dwpose_raw_np.min() >= -0.01:
                print('Pose pkl values in [0,1], converting to pixel coordinates first.')
                dwpose_raw_np[..., 0] = dwpose_raw_np[..., 0] * args.width
                dwpose_raw_np[..., 1] = dwpose_raw_np[..., 1] * args.height
            elif dwpose_raw_np.max() <= 1.2 and dwpose_raw_np.min() >= -1.2:
                print('Pose pkl values appear in [-1,1], using directly without pixel conversion.')
            else:
                print('Warning: pkl pose coordinates outside expected range, assuming pixel coordinates.')
            # 对齐帧数
            if dwpose_raw_np.shape[0] < num_frames:
                raise ValueError(f"Pose pkl frame count {dwpose_raw_np.shape[0]} < control frames {num_frames}")
            dwpose_raw_np = dwpose_raw_np[:num_frames]
        else:
            full_pose_data = np.load(args.pose_data_path)
            print(f"Loaded pose npy: {args.pose_data_path}, shape={full_pose_data.shape}, min={full_pose_data.min():.4f}, max={full_pose_data.max():.4f}")
            if full_pose_data.ndim != 3 or full_pose_data.shape[1:] != (18, 2):
                raise ValueError(f"pose_data_path must be shape [frames, 18, 2], got {full_pose_data.shape}")
            if full_pose_data.shape[0] < num_frames:
                raise ValueError(f"Pose npy frame count {full_pose_data.shape[0]} < control frames {num_frames}")
            dwpose_raw_np = full_pose_data[:num_frames]
    else:
        print("Warning: No pose data provided, using zeros.")
        dwpose_raw_np = np.zeros((num_frames, 18, 2), dtype=np.float32)

    # -------------------------------------------------------------------
    # === [IPI Mod] 终极补丁：在线分部位重定位 + 实时骨架重绘 ===
    # -------------------------------------------------------------------
    print("\n🔧 正在执行动作重定位与中心补偿 (Part-based Retargeting)...")
    print("   [1/2] 正在提取参考图像原始骨架...")
    from DWPose.preprocess.pose.script.dwpose import DWposeDetector
    detector = DWposeDetector(
        det_config=f"{os.path.dirname(os.path.abspath(__file__))}/DWPose/preprocess/pose/config/yolox_l_8xb8-300e_coco.py",
        det_ckpt="/gaojiawei/wjl/StableAnimator/checkpoints/DWPose/yolox_l_8x8_300e_coco.pth",
        pose_config=f"{os.path.dirname(os.path.abspath(__file__))}/DWPose/preprocess/pose/config/dwpose-l_384x288.py",
        pose_ckpt="/gaojiawei/wjl/StableAnimator/checkpoints/DWPose/dw-ll_ucoco_384.pth",
        device=torch.device('cuda')
    )
    img_refer = cv2.imread(args.validation_image)
    result = detector(img_refer)
    pose_refer_dict = result[1] if isinstance(result, tuple) else result
    
    # 兼容没有检测到人的情况
    if pose_refer_dict['bodies']['candidate'].shape[0] == 0:
        print("   ⚠️ 警告：参考图像未检测出有效骨架，跳过重定位直接绘制。")
    else:
        ref_pose_raw = np.array(pose_refer_dict['bodies']['candidate'], dtype=np.float32)
        # 将原始像素坐标恢复给 sanitize_pose_data 进行处理
        ref_pose_clean = sanitize_pose_data(ref_pose_raw[np.newaxis, ...])[0]
        dwpose_raw_np = sanitize_pose_data(dwpose_raw_np)
        
        # ── 宽高比修正（与 pose_align.py 保持一致） ──────────────────────────
        # DWPose 输出 [0,1] 归一化坐标，x 方向实际物理长度 = 归一化值 × W/H
        # 不修正时用欧式距离算出的缩放比在非正方形图像上会偏差
        ref_ratio = args.width / args.height
        ref_pose_for_scale = ref_pose_clean.copy()
        ref_pose_for_scale[:, 0] = ref_pose_for_scale[:, 0] * ref_ratio   # x × W/H

        first_drv_pose_raw = dwpose_raw_np[0].copy()
        first_drv_pose_for_scale = first_drv_pose_raw.copy()
        first_drv_pose_for_scale[:, 0] = first_drv_pose_for_scale[:, 0] * ref_ratio  # x × W/H
        # ────────────────────────────────────────────────────────────────────

        # 用修正后的坐标计算各部位缩放比
        body_scales = compute_body_scales(ref_pose_for_scale, first_drv_pose_for_scale)
        # 中心点偏移仍在原始 [0,1] 归一化坐标系下计算（render 时会用到）
        center_offset = ref_pose_clean[1] - first_drv_pose_raw[1]

        for i in range(len(dwpose_raw_np)):
            dwpose_raw_np[i] = align_body_pose_single(dwpose_raw_np[i], body_scales)
            valid_mask = (dwpose_raw_np[i][:, 0] != -1) & (dwpose_raw_np[i][:, 1] != -1)
            dwpose_raw_np[i][valid_mask] += center_offset
            
        print("   [2/2] 分部位形变与中心捕捉完成！")
    
    # 释放 detector 以防 OOM
    del detector
    import gc
    gc.collect()

    print("\n🔧 正在进行方案 B 重绘：在线提取纯净彩色骨架...")
    validation_control_images = render_pose_frames(dwpose_raw_np.copy(), args.width, args.height)

    # 🚀 [统一归一化逻辑] 自动识别并映射到 [-1, 1]
    print("\n🔧 正在执行骨架坐标修正 (与训练集对齐)...")
    dwpose_raw_np = normalize_pose(dwpose_raw_np, args.width, args.height)
    print(f"✅ 修正完毕！有效点值域: min={dwpose_raw_np.min():.4f}, max={dwpose_raw_np.max():.4f}")
    
    # 🚀 [IPI Mod] 动作有效性检测
    if np.abs(dwpose_raw_np).max() < 1e-5:
        print("\n" + "!"*60)
        print("⚠️ [系统警告] 检测到 dwpose_raw 为全 0！")
        print("   这意味着没有动作输入，视频将注定不会产生动作。")
        print("   请检查 .npy/.pkl 文件是否包含有效的 DWPose 坐标。")
        print("!"*60 + "\n")
    else:
        # 检测帧间变化
        if dwpose_raw_np.shape[0] > 1:
            diff = np.abs(dwpose_raw_np[1:] - dwpose_raw_np[:-1]).mean()
            if diff < 1e-6:
                print("\n" + "!"*60)
                print(f"⚠️ [警告] 检测到帧间动作几乎为零 (diff={diff:.8f})")
                print("   骨架数据在所有帧中是完全【静止】的。")
                print("!"*60 + "\n")
    
    # [Frames, 18, 2] -> [1, Frames, 36] (增加 Batch 维度并 Flatten)
    dwpose_raw_tensor = torch.from_numpy(dwpose_raw_np).float().view(1, num_frames, -1).to(device='cuda', dtype=weight_dtype)

    # -------------------------------------------------------
    # 2. 准备 CLIP Image Embeddings (动态驱动特征)
    # -------------------------------------------------------
    if args.driving_video_folder and os.path.exists(args.driving_video_folder):
        print(f"🎬 读取驱动视频提取动态 CLIP: {args.driving_video_folder}")
        driving_pil_images = load_images_from_folder(args.driving_video_folder, width=args.width, height=args.height)
        
        # 🚀 [IPI Mod] 驱动视频质量检测
        if len(driving_pil_images) > 0:
            first_img_np = np.array(driving_pil_images[0])
            avg_brightness = first_img_np.mean()
            if avg_brightness < 10:  # 经验值：骨架图通常背景是全黑，亮度极低
                print("\n" + "!"*60)
                print(f"⚠️ [警告] 驱动视频平均亮度极低 ({avg_brightness:.2f})")
                print("   这看起来像是【骨架图】而不是【真实 RGB 视频帧】。")
                print("   IPI 模型在训练阶段使用的是真实图像特征，如果推理时给的是骨架图，")
                print("   会导致 CLIP 描述子完全错位，从而无法正确引导运动！")
                print("!"*60 + "\n")
    else:
        print("⚠️ 未提供 driving_video_folder，将使用 validation_control_folder 的图像（请注意这些必须是 RGB 原图）")
        driving_pil_images = validation_control_images

    clip_inputs = feature_extractor(images=driving_pil_images, return_tensors="pt")
    driving_clip_pixels = clip_inputs.pixel_values.to(device='cuda', dtype=weight_dtype)

    # ====================================================================
    # 🚀 [推理终极补丁：同步致盲！必须与训练时的视觉域保持绝对一致！]
    # ====================================================================
    if args.is_statue:
        print("🗿 推理致盲启动：对驱动视频执行 [灰度化 + 强模糊]，切断身份/材质泄漏！")
        import torchvision.transforms as T
        gray_transform = T.Grayscale(num_output_channels=3)
        blur_transform = T.GaussianBlur(kernel_size=(3, 3), sigma=(1.0, 1.0))
        driving_clip_pixels = blur_transform(gray_transform(driving_clip_pixels))

    with torch.no_grad():
        clip_image_embeds = image_encoder(driving_clip_pixels).image_embeds
        clip_image_embeds = clip_image_embeds.unsqueeze(0)   # [1, num_frames, 1024]

    # 🚀 雕像免检通道：如果是雕像，强制给全0特征，不调 InsightFace！
    if args.is_statue:
        print("🗿 雕像模式已开启：强制跳过人脸检测，提供全0特征！")
        validation_image_id_ante_embedding = np.zeros((512,))
        # 👇 新增：终极补丁！直接暴力劫持 face_encoder，让它只能吐出绝对的全 0 张量！
        def statue_forward(self, faceid_embeds, image_embed, *args, **kwargs):
            return torch.zeros(image_embed.shape[0], 4, image_embed.shape[-1], 
                               device=image_embed.device, dtype=image_embed.dtype)
        
        import types
        face_encoder.forward = types.MethodType(statue_forward, face_encoder)
        print("🔪 FaceEncoder 已被物理切断，UNet 将接收纯净的全 0 材质信号！")
    else:
        face_model.face_helper.clean_all() # 清空人脸处理工具的缓存
        validation_face = cv2.imread(args.validation_image) # 用 OpenCV 读取参考图像
        validation_image_bgr = cv2.cvtColor(validation_face, cv2.COLOR_RGB2BGR)
        validation_image_face_info = face_model.app.get(validation_image_bgr)  
        if len(validation_image_face_info) > 0:
            validation_image_face_info = sorted(validation_image_face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[-1]
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
                print('fail to detect face using insightface, extract embedding on align face')
                validation_image_id_ante_embedding = face_model.handler_ante.get_feat(validation_image_align_face)

    # generator = torch.Generator(device=accelerator.device).manual_seed(23123134)

    decode_chunk_size = args.decode_chunk_size # 从命令行参数获取解码分块大小
    video_frames = pipeline(                     # 调用pipeline生成视频帧
        image=validation_image,                  # 参考图像（生成视频的基础图像）               
        image_pose=validation_control_images,    # 姿态控制图像序列（指导人物动作）
        height=args.height,                      # 生成视频的高度
        width=args.width,                        # 生成视频的宽度
        num_frames=num_frames,                   # 视频帧数（由姿态控制图像数量决定）
        tile_size=args.tile_size,                # 分块生成时的块大小（用于大尺寸视频的内存优化）
        tile_overlap=args.frames_overlap,        # 分块间的重叠帧数（避免块间衔接生硬）
        decode_chunk_size=decode_chunk_size,     # VAE解码时的分块大小（避免解码大视频时OOM）
        
        # 🚀 如果是雕像，运动桶开到 140 逼它动；真人保持你的默认 100
        #motion_bucket_id=100. if args.is_statue else 100., 
        motion_bucket_id=args.motion_bucket_id,
        fps=7,                                   # 视频帧率（每秒7帧）
        min_guidance_scale=args.guidance_scale,  # 引导尺度最小值（控制参考图像的影响程度）
        max_guidance_scale=args.guidance_scale,  # 引导尺度最大值（与最小值相同，固定引导强度）
        noise_aug_strength=args.noise_aug_strength, # 噪声增强强度（增加生成多样性）
        num_inference_steps=args.num_inference_steps, # 扩散模型推理步数（步数越多，生成越精细）
        generator=generator,                        # 随机数生成器（控制生成结果的可复现性）
        output_type="pil",                           # 输出格式（PIL图像列表）
        validation_image_id_ante_embedding=validation_image_id_ante_embedding,  # 参考图像的人脸特征
        
        is_statue=args.is_statue,  # 🚀 将雕像开关传进底层 Pipeline！
        
        # === [IPI Mod] 传入 Pipeline ===
        dwpose_raw=None if args.no_ipi else dwpose_raw_tensor,
        clip_image_embeds=None if args.no_ipi else clip_image_embeds,
        # =============================
    ).frames[0]

    print(f"📸 video_frames type: {type(video_frames)}")
    if isinstance(video_frames, list):
        print(f"📸 video_frames length: {len(video_frames)}")
        if len(video_frames) > 0:
            first_frame = video_frames[0]
            print(f"📸 first frame type: {type(first_frame)}")
            if isinstance(first_frame, Image.Image):
                print(f"📸 first frame size: {first_frame.size}")
                print(f"📸 first frame mode: {first_frame.mode}")
                # 转numpy查看数值范围
                frame_np = np.array(first_frame)
                print(f"📸 frame numpy shape: {frame_np.shape}, dtype: {frame_np.dtype}")
                print(f"📸 frame min: {frame_np.min()}, max: {frame_np.max()}")

    out_file = os.path.join(
        args.output_dir,
        f"animation_video.mp4",
    )
    for i in range(num_frames):
        img = video_frames[i]
        video_frames[i] = np.array(img)

    png_out_file = os.path.join(args.output_dir, "animated_images")
    os.makedirs(png_out_file, exist_ok=True)
    export_to_gif(video_frames, out_file, 8)
    save_frames_as_png(video_frames, png_out_file)
    #save_frames_as_mp4(video_frames, png_out_file, 60)


# bash command_basic_infer.sh
