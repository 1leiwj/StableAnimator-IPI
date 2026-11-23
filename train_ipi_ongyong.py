#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_ipi_ongyong.py
通用 IPI 训练脚本（在 train.py + train_ipi_diaosu.py 基础上整合修复）

训练策略：与 train_ipi_diaosu.py 完全一致（三轨制 LR）
  IPI Adapter  → base_lr * 3.0
  UNet 时序层  → base_lr * 1.0
  UNet 空间层  → base_lr * 0.5
  PoseNet      → 冻结
  FaceEncoder  → 冻结

Bug 修复列表（相对 train_ipi_diaosu.py）：
  [FIX-1] CLIP Dropout 独立于 Pose Dropout，默认 50% 概率独立掉落
  [FIX-2] statue_repeat_factor 真正生效（ConcatDataset * dataset3_repeat）
  [FIX-3] PoseNet 冻结时调用 pose_net.eval()，防止 BN 统计量污染
  [FIX-4] 修复雕像 CLIP 致盲的双重灰度化 Bug（现在只灰度一次）
  [FIX-5] 材质一致性 Loss 系数由 0 改为 0.1（可通过 --consistency_loss_weight 调整）
  [FIX-6] is_statue 默认值改为 False（防止雕像模式永远开启的致命 Bug）
"""

import argparse
import logging
import math
import os
import random
import shutil
import warnings
import datetime
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import numpy as np
import PIL
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms as T
from einops import rearrange
from torch.utils.data import ConcatDataset
from tqdm.auto import tqdm

import transformers
import diffusers
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.attention_processor import XFormersAttnProcessor
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, load_image
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import randn_tensor
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed

from animation.dataset.animation_dataset_ipi import LargeScaleAnimationVideos
from animation.modules.attention_processor import AnimationAttnProcessor
from animation.modules.attention_processor_normalized import AnimationIDAttnNormalizedProcessor
from animation.modules.face_model import FaceModel
from animation.modules.id_encoder import FusionFaceId
from animation.modules.pose_net import PoseNet
from animation.modules.unet_ipi import UNetSpatioTemporalConditionModel
from animation.pipelines.inference_pipeline_animation_ipi import InferenceAnimationPipeline

check_min_version("0.24.0.dev0")
logger = get_logger(__name__, log_level="INFO")

# ─────────────────────────────────────────────────────────────────────────────
# 常量（与原版 train.py 一致）
# ─────────────────────────────────────────────────────────────────────────────
min_value   = 0.002
max_value   = 700
image_d     = 64
noise_d_low = 32
noise_d_high= 64
sigma_data  = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数（与原版完全一致，保持可维护性）
# ─────────────────────────────────────────────────────────────────────────────
def validate_and_convert_image(image, target_size=(256, 256)):
    if image is None:
        return None
    if isinstance(image, torch.Tensor):
        if image.ndim == 3 and image.shape[0] in [1, 3]:
            if image.shape[0] == 1:
                image = image.repeat(3, 1, 1)
            image = image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            image = Image.fromarray(image)
        else:
            return None
    elif isinstance(image, Image.Image):
        image = image.resize(target_size)
    else:
        return None
    return image


def create_image_grid(images, rows, cols, target_size=(256, 256)):
    valid = [validate_and_convert_image(img, target_size) for img in images]
    valid = [img for img in valid if img is not None]
    if not valid:
        return None
    w, h = target_size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, image in enumerate(valid):
        grid.paste(image, box=((i % cols) * w, (i // cols) * h))
    return grid


def save_combined_frames(batch_output, validation_images, validation_control_images, output_folder):
    flattened = [img for sublist in batch_output for img in sublist]
    combined = validation_images + validation_control_images + flattened
    cols = 3
    rows = (len(combined) + cols - 1) // cols
    grid = create_image_grid(combined, rows, cols)
    out_dir = os.path.join(output_folder, "validation_images")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if grid is not None:
        grid.save(os.path.join(out_dir, f"combined_frames_{ts}.png"))


def load_images_from_folder(folder):
    files = os.listdir(folder)
    png_files = [f for f in files if f.endswith('.png')]
    png_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    return [Image.open(os.path.join(folder, f)).convert('RGB') for f in png_files]


def numbered_checkpoints(output_dir):
    return sorted(
        (path for path in Path(output_dir).glob("checkpoint-*")
         if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()),
        key=lambda path: int(path.name.removeprefix("checkpoint-")),
    )


@contextmanager
def measure_mid_ipi(unet, frames, step, enabled):
    """Measure the gated residual before the mid-block time mixer on one training batch."""
    if not enabled:
        yield
        return
    mid = unet.mid_block.attentions[0]
    values = {}

    def record_projection(module, inputs, output):
        values["residual"] = output.detach() * mid.ipi_temporal_gate.detach().tanh()

    def record_temporal(module, inputs, output):
        residual = values["residual"].to(output.dtype).float().reshape(-1, frames, output.shape[-1])
        rms = output.detach().float().square().mean().sqrt().clamp_min(1e-12)
        relative = residual.square().mean().sqrt() / rms
        relative_change = residual.diff(dim=1).square().mean().sqrt() / rms if frames > 1 else rms.new_zeros(())
        logger.info(f"[IPI probe] step={step} mid_residual/temporal_RMS={relative.item():.3e} "
                    f"mid_frame_delta/temporal_RMS={relative_change.item():.3e}")

    projection = mid.ipi_temporal_projection.register_forward_hook(record_projection)
    temporal = mid.temporal_transformer_blocks[-1].register_forward_hook(record_temporal)
    try:
        yield
    finally:
        projection.remove()
        temporal.remove()


def stratified_uniform(shape, group=0, groups=1, dtype=None, device=None):
    if groups <= 0:
        raise ValueError(f"groups must be positive, got {groups}")
    if group < 0 or group >= groups:
        raise ValueError(f"group must be in [0, {groups})")
    n = shape[-1] * groups
    offsets = torch.arange(group, n, groups, dtype=dtype, device=device)
    u = torch.rand(shape, dtype=dtype, device=device)
    return (offsets + u) / n


def rand_cosine_interpolated(shape, image_d, noise_d_low, noise_d_high,
                              sigma_data=1., min_value=1e-3, max_value=1e3,
                              device='cpu', dtype=torch.float32):
    def logsnr_schedule_cosine(t, logsnr_min, logsnr_max):
        t_min = math.atan(math.exp(-0.5 * logsnr_max))
        t_max = math.atan(math.exp(-0.5 * logsnr_min))
        return -2 * torch.log(torch.tan(t_min + t * (t_max - t_min)))

    def logsnr_schedule_cosine_shifted(t, image_d, noise_d, logsnr_min, logsnr_max):
        shift = 2 * math.log(noise_d / image_d)
        return logsnr_schedule_cosine(t, logsnr_min - shift, logsnr_max - shift) + shift

    def logsnr_schedule_cosine_interpolated(t, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max):
        logsnr_low = logsnr_schedule_cosine_shifted(t, image_d, noise_d_low, logsnr_min, logsnr_max)
        logsnr_high = logsnr_schedule_cosine_shifted(t, image_d, noise_d_high, logsnr_min, logsnr_max)
        return torch.lerp(logsnr_low, logsnr_high, t)

    logsnr_min = -2 * math.log(min_value / sigma_data)
    logsnr_max = -2 * math.log(max_value / sigma_data)
    u = stratified_uniform(shape, group=0, groups=1, dtype=dtype, device=device)
    logsnr = logsnr_schedule_cosine_interpolated(u, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max)
    return torch.exp(-logsnr / 2) * sigma_data


def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])
    sigmas = (max((factors[0] - 1.0) / 2.0, 0.001), max((factors[1] - 1.0) / 2.0, 0.001))
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))
    if (ks[0] % 2) == 0: ks = ks[0] + 1, ks[1]
    if (ks[1] % 2) == 0: ks = ks[0], ks[1] + 1
    input = _gaussian_blur2d(input, ks, sigmas)
    return torch.nn.functional.interpolate(input, size=size, mode=interpolation, align_corners=align_corners)


def _compute_padding(kernel_size):
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]
    out_padding = 2 * len(kernel_size) * [0]
    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]
        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front
        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear
    return out_padding


def _filter2d(input, kernel):
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)
    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)
    height, width = tmp_kernel.shape[-2:]
    padding_shape = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))
    output = torch.nn.functional.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)
    return output.view(b, c, h, w)


def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])
    batch_size = sigma.shape[0]
    x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)
    if window_size % 2 == 0:
        x = x + 0.5
    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))
    return gauss / gauss.sum(-1, keepdim=True)


def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)
    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    return _filter2d(out_x, kernel_y[..., None])


def export_to_gif(frames, output_gif_path, fps):
    pil_frames = [Image.fromarray(frame) if isinstance(
        frame, np.ndarray) else frame for frame in frames]

    pil_frames[0].save(output_gif_path.replace('.mp4', '.gif'),
                       format='GIF',
                       append_images=pil_frames[1:],
                       save_all=True,
                       duration=125,
                       loop=0)


def tensor_to_vae_latent(t, vae, scale=True):
    t = t.to(vae.dtype)
    if len(t.shape) == 5:
        video_length = t.shape[1]
        t = rearrange(t, "b f c h w -> (b f) c h w")
        latents = vae.encode(t).latent_dist.sample()
        latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)
    elif len(t.shape) == 4:
        latents = vae.encode(t).latent_dist.sample()
    if scale:
        latents = latents * vae.config.scaling_factor
    return latents


def export_to_video(video_frames, output_video_path, fps):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, _ = video_frames[0].shape
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for frame in video_frames:
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(img)


# ─────────────────────────────────────────────────────────────────────────────
# 用于 DeepSpeed 多模型包装
# ─────────────────────────────────────────────────────────────────────────────
class DeepSpeedWrapperModel(nn.Module):
    def __init__(self, unet, pose_net):
        super().__init__()
        self.unet = unet
        self.pose_net = pose_net

    def forward(self, *args, **kwargs):
        return self.unet(*args, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 参数解析
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="通用 IPI 训练脚本（双轨：IPI Adapter + PoseNet）")

    # 模型路径
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--pretrain_unet", type=str, default=None)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--posenet_model_name_or_path", type=str, default=None)
    parser.add_argument("--face_encoder_model_name_or_path", type=str, default=None)
    parser.add_argument("--unet_model_name_or_path", type=str, default=None)

    # 微调续训路径
    parser.add_argument("--finetune_mode", type=bool, default=False)
    parser.add_argument("--posenet_model_finetune_path", type=str, default=None)
    parser.add_argument("--face_encoder_finetune_path", type=str, default=None)
    parser.add_argument("--unet_model_finetune_path", type=str, default=None)
    parser.add_argument("--ipi_model_finetune_path", type=str, default=None,
                        help="单独的 IPI Adapter .pth 路径（从上一次训练续训）")

    # 数据集路径（支持最多 3 个数据集混合）
    parser.add_argument("--data_root_path", type=str, default=None)
    parser.add_argument("--rec_data_path", type=str, default=None, help="数据集 1 的 txt 索引文件")
    parser.add_argument("--vec_data_path", type=str, default=None, help="数据集 2 的 txt 索引文件（可选）")
    parser.add_argument("--clay_data_path", type=str, default=None, help="数据集 3 的 txt 索引文件（可选）")
    parser.add_argument("--dataset3_repeat", type=int, default=1,
                        help="数据集 3 的过采样倍数（如雕像数据量少，设 3~5）")

    # 验证相关
    parser.add_argument("--validation_image", type=str, default=None)
    parser.add_argument("--validation_image_folder", type=str, default=None)
    parser.add_argument("--validation_control_folder", type=str, default=None)
    parser.add_argument("--pose_data_path", type=str, default=None,
                        help="验证用的 .npy 骨架文件 (Frames, 133/18, 2)")
    parser.add_argument("--validation_steps", type=int, default=500)
    parser.add_argument("--num_validation_images", type=int, default=1)

    # 训练超参数
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--per_gpu_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--ipi_temporal_v2", action="store_true", help="Use residual temporal IPI.")
    parser.add_argument("--ipi_joint_motion", type=int, choices=(0, 1), default=os.environ.get("IPI_JOINT_MOTION", "0"),
                        help="Fit only the localized IPI joint encoder and its zero-initialized readouts.")
    parser.add_argument(
        "--ipi_only_finetune", action="store_true",
        help="Freeze the legacy model and train only the IPI Adapter and residual IPI parameters.",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="基础学习率（时序层 / PoseNet）")
    parser.add_argument("--ipi_lr_scale", type=float, default=3.0,
                        help="IPI Adapter 学习率相对倍数（default: 3x，与 diaosu 一致）")
    parser.add_argument("--spatial_lr_scale", type=float, default=0.5,
                        help="UNet 空间层学习率相对倍数（default: 0.5x，与 diaosu 一致）")
    parser.add_argument("--consistency_loss_weight", type=float, default=0.1,
                        help="[FIX-5] 材质一致性 Loss 系数（diaosu 原版是 0，此处修复为 0.1）")
    parser.add_argument("--scale_lr", action="store_true", default=False)
    parser.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--conditioning_dropout_prob", type=float, default=0.1,
                        help="整体条件 Dropout 概率（图像 / Pose 等）")
    parser.add_argument("--clip_dropout_prob", type=float, default=0.5,
                        help="[FIX-2] CLIP 静态特征的独立 Dropout 概率（建议 0.5）")

    # 优化器
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--adam_beta1", type=float, default=0.95)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # 混合精度 / EMA / xformers
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--non_ema_revision", type=str, default=None)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")

    # 数据加载
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--sample_n_frames", type=int, default=14)
    parser.add_argument("--train_frame_stride", type=int, default=os.environ.get("TRAIN_FRAME_STRIDE", "4"),
                        help="Spacing in source frames within a training clip. Short videos use the available span.")
    parser.add_argument("--target_frame_pose", action="store_true",
                        help="Use the target RGB frame's own detected pose for spatial and IPI supervision.")
    parser.add_argument("--ipi_limb_mask_prob", type=float, default=os.environ.get("IPI_LIMB_MASK_PROB", "0"),
                        help="Probability of hiding one limb in the PoseNet drawing only (IPI-only pilot).")
    parser.add_argument("--num_frames", type=int, default=16)

    # checkpoint
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=2)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--resume_last_step", type=int, help="Completed steps for an older checkpoint-last without step metadata.")

    # 其他
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--log_trainable_parameters", action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--rank", type=int, default=128, help="LoRA rank")
    parser.add_argument("--train_noise_aug", type=float, default=0.04,
                        help="条件图像噪声增强强度（打破 SVD 惰性，建议 0.02~0.05）")

    args = parser.parse_args()
    if args.train_frame_stride < 1:
        parser.error("--train_frame_stride (TRAIN_FRAME_STRIDE) must be positive.")
    if args.ipi_joint_motion not in (0, 1) or (args.ipi_joint_motion and not (
            args.ipi_only_finetune and args.target_frame_pose)):
        parser.error("IPI_JOINT_MOTION must be 0 or 1; 1 requires IPI-only training with target-frame poses.")
    if not 0.0 <= args.ipi_limb_mask_prob <= 1.0:
        parser.error("--ipi_limb_mask_prob must be between 0 and 1.")
    if args.ipi_limb_mask_prob and not (args.target_frame_pose and args.ipi_only_finetune):
        parser.error("IPI limb masking requires --target_frame_pose and --ipi_only_finetune.")
    if args.ipi_only_finetune:
        if not (args.ipi_temporal_v2 and args.finetune_mode):
            parser.error("--ipi_only_finetune requires --ipi_temporal_v2 and --finetune_mode=True.")
        for argument in (
            "unet_model_finetune_path", "ipi_model_finetune_path",
            "posenet_model_finetune_path", "face_encoder_finetune_path",
        ):
            filename = getattr(args, argument)
            if not filename or not Path(filename).is_file():
                parser.error(f"--ipi_only_finetune requires an existing file for --{argument}.")
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision
    return args


# ─────────────────────────────────────────────────────────────────────────────
# 验证函数
# ─────────────────────────────────────────────────────────────────────────────
def log_validation(vae, image_encoder, unet, pose_net, face_encoder,
                   app, face_helper, handler_ante, scheduler, accelerator,
                   feature_extractor, width, height, torch_dtype,
                   validation_image_folder, validation_image,
                   validation_control_folder, output_dir, generator,
                   global_step, num_frames, pose_data_path=None,
                   num_validation_cases=1, ipi_temporal_v2=False):

    logger.info("Running validation...")
    val_unet        = accelerator.unwrap_model(unet)
    val_img_enc     = accelerator.unwrap_model(image_encoder)
    val_vae         = accelerator.unwrap_model(vae)
    val_pose_net    = accelerator.unwrap_model(pose_net)
    val_face_enc    = accelerator.unwrap_model(face_encoder)

    # 切换 eval 模式，确保 BN / Dropout 行为与推理一致
    val_unet.eval()
    if hasattr(val_unet, "ipi_adapter") and val_unet.ipi_adapter is not None:
        val_unet.ipi_adapter.eval()
    val_pose_net.eval()
    val_face_enc.eval()
    val_img_enc.eval()

    pipeline = InferenceAnimationPipeline(
        vae=val_vae,
        image_encoder=val_img_enc,
        unet=val_unet,
        scheduler=scheduler,
        feature_extractor=feature_extractor,
        pose_net=val_pose_net,
        face_encoder=val_face_enc,
    ).to(accelerator.device)

    validation_img  = Image.open(validation_image).convert("RGB").resize((width, height))
    control_images  = load_images_from_folder(validation_control_folder)
    n_frames        = min(len(control_images), num_frames)
    control_images  = [img.resize((width, height)) for img in control_images[:n_frames]]

    # 准备骨架张量（供 IPI 使用）
    dwpose_raw_tensor    = None
    clip_image_embeds_val= None
    if pose_data_path:
        dwpose_raw_np = np.load(pose_data_path, allow_pickle=False)
        if (dwpose_raw_np.ndim != 3 or dwpose_raw_np.shape[1:] != (133, 2)
                or len(dwpose_raw_np) < n_frames):
            raise ValueError(
                f"IPI validation needs at least {n_frames} frames of 133 (x, y) joints, "
                f"found {dwpose_raw_np.shape}."
            )
        dwpose_raw_np = dwpose_raw_np[:n_frames]
        # 归一化到 [-1, 1]
        valid_mask = (dwpose_raw_np[..., 0] != -1) & (dwpose_raw_np[..., 1] != -1)
        if valid_mask.any() and dwpose_raw_np[valid_mask].max() > 1.5:
            dwpose_raw_np[valid_mask, 0] = dwpose_raw_np[valid_mask, 0] / width  * 2.0 - 1.0
            dwpose_raw_np[valid_mask, 1] = dwpose_raw_np[valid_mask, 1] / height * 2.0 - 1.0
        elif valid_mask.any() and dwpose_raw_np[valid_mask].max() <= 1.05:
            dwpose_raw_np[valid_mask] = dwpose_raw_np[valid_mask] * 2.0 - 1.0
        dwpose_raw_tensor = (torch.from_numpy(dwpose_raw_np).float()
                             .view(1, n_frames, -1).to(device=accelerator.device, dtype=torch_dtype))

    if dwpose_raw_tensor is not None:
        if not validation_image_folder:
            raise ValueError("IPI validation requires --validation_image_folder with driving RGB frames.")
        driving_images = load_images_from_folder(validation_image_folder)
        if len(driving_images) < n_frames:
            raise ValueError(f"IPI validation needs {n_frames} RGB frames, found {len(driving_images)}.")
        driving_images = [image.resize((width, height)) for image in driving_images[:n_frames]]
        frames_tensor = feature_extractor(images=driving_images, return_tensors="pt").pixel_values
        frames_tensor = frames_tensor.to(accelerator.device, dtype=torch_dtype)
        with torch.no_grad():
            clip_image_embeds_val = val_img_enc(frames_tensor).image_embeds.unsqueeze(0)

    # 人脸 embedding 置零（通用验证，不区分真人/雕像）
    validation_image_id_ante_embedding = np.zeros((512,))

    with torch.no_grad():
        video_frames = pipeline(
            image=validation_img,
            image_pose=control_images,
            height=height,
            width=width,
            num_frames=n_frames,
            tile_size=n_frames,
            tile_overlap=6,
            decode_chunk_size=8,
            motion_bucket_id=127.0,
            fps=7,
            min_guidance_scale=3.0,
            max_guidance_scale=3.0,
            noise_aug_strength=0.02,
            num_inference_steps=25,
            generator=generator,
            output_type="pil",
            validation_image_id_ante_embedding=validation_image_id_ante_embedding,
            dwpose_raw=dwpose_raw_tensor,
            clip_image_embeds=clip_image_embeds_val,
            ipi_scale=1.0 if ipi_temporal_v2 else None,
            ipi_temporal_v2=ipi_temporal_v2,
        ).frames[0]

    # 保存视频帧为 GIF (对齐 diaosu.py)
    save_dir = os.path.join(output_dir, f"validation_step{global_step}")
    os.makedirs(save_dir, exist_ok=True)
    out_file = os.path.join(save_dir, "validation_output.mp4")
    
    for i in range(len(video_frames)):
        img = video_frames[i]
        video_frames[i] = np.array(img)
    export_to_gif(video_frames, out_file, 8)
    logger.info(f"Saved validation GIF to {out_file.replace('.mp4', '.gif')}")

    del pipeline
    torch.cuda.empty_cache()

    # ✅ 极其关键：验证结束后必须将 UNet 恢复为 train() 模式
    # 否则下一轮训练由于 self.training=False 会禁用 gradient_checkpointing 导致立刻 OOM！
    val_unet.train()
    if hasattr(val_unet, "ipi_adapter") and val_unet.ipi_adapter is not None:
        val_unet.ipi_adapter.train()


# ─────────────────────────────────────────────────────────────────────────────
# 主训练函数
# ─────────────────────────────────────────────────────────────────────────────
def main():
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    torch.multiprocessing.set_start_method('spawn')

    args = parse_args()

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
        log_with=args.report_to,
        kwargs_handlers=[ddp_kwargs],
    )

    generator = torch.Generator(device=accelerator.device).manual_seed(23123134)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # ─── 加载基础模型 ────────────────────────────────────────────────────────
    feature_extractor = CLIPImageProcessor.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="feature_extractor", revision=args.revision)
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="image_encoder", revision=args.revision)
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant="fp16")
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        args.pretrained_model_name_or_path if args.pretrain_unet is None else args.pretrain_unet,
        subfolder="unet",
        low_cpu_mem_usage=False,
        device_map=None,
        variant="fp16",
        ignore_mismatched_sizes=True,
    )

    # ─── 初始化 IPI Adapter ───────────────────────────────────────────────────
    from animation.modules.ipi_adapter import MotionAdapter
    ipi_adapter = MotionAdapter(
        cross_attention_dim=getattr(unet.config, "cross_attention_dim", 1024),
        clip_dim=1024,
        num_tokens=8,
    )
    # 零初始化输出层，保证训练初期 IPI 不干扰 UNet 的既有能力
    zero_init_count = 0
    for name, param in ipi_adapter.named_parameters():
        if "proj_out.weight" in name or "proj_out.bias" in name:
            torch.nn.init.zeros_(param)
            zero_init_count += 1
    if accelerator.is_main_process:
        logger.info(f"[IPI Init] 零初始化了 {zero_init_count} 个输出层参数")
    unet.ipi_adapter = ipi_adapter
    ipi_adapter.to(unet.device, dtype=unet.dtype)

    # ─── 辅助模块 ─────────────────────────────────────────────────────────────
    pose_net = PoseNet(noise_latent_channels=unet.config.block_out_channels[0])
    face_encoder = FusionFaceId(
        cross_attention_dim=1024,
        id_embeddings_dim=512,
        clip_embeddings_dim=1024,
        num_tokens=4,
    )
    face_model = FaceModel()

    # ─── 注册 Attention Processor ─────────────────────────────────────────────
    lora_rank = args.rank
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
                proc = AnimationIDAttnNormalizedProcessor(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank)
                proc.load_state_dict(weights, strict=False)
                attn_procs[name] = proc
        elif "temporal_transformer_blocks" in name:
            attn_procs[name] = unet.attn_processors[name]

    unet.set_attn_processor(attn_procs)

    # ─── 加载微调权重（续训模式）─────────────────────────────────────────────
    if (args.finetune_mode and args.posenet_model_finetune_path
            and args.face_encoder_finetune_path and args.unet_model_finetune_path):
        logger.info("[Finetune] 加载已有权重进行续训...")

        pose_net.load_state_dict(
            torch.load(args.posenet_model_finetune_path, map_location="cpu"), strict=True)
        face_encoder.load_state_dict(
            torch.load(args.face_encoder_finetune_path, map_location="cpu"), strict=True)

        animator_sd = torch.load(args.unet_model_finetune_path, map_location="cpu")
        if args.ipi_only_finetune:
            ipi_sd = torch.load(args.ipi_model_finetune_path, map_location="cpu")
            if args.ipi_joint_motion:
                missing_v2 = [k for k, _ in unet.named_parameters() if "ipi_temporal_" in k and k not in animator_sd]
                if missing_v2:
                    raise ValueError(f"Joint-motion training requires a complete v2 parent: {missing_v2[:4]}")
                if any(k.startswith("joint_motion.") for k in ipi_sd):
                    unet.ipi_adapter.enable_joint_motion(unet.config.block_out_channels)
            unet.ipi_adapter.load_state_dict(ipi_sd, strict=True)
            # Use the dedicated IPI export even if the base checkpoint embeds an adapter.
            animator_sd = {k: v for k, v in animator_sd.items() if not k.startswith("ipi_adapter.")}
            animator_sd.update({f"ipi_adapter.{key}": value for key, value in ipi_sd.items()})
            missing, unexpected = unet.load_state_dict(animator_sd, strict=False)
            missing = [
                key for key in missing
                if not key.endswith(("ipi_temporal_gate", "ipi_temporal_projection.weight"))
            ]
            if missing or unexpected:
                raise ValueError(
                    f"Base UNet checkpoint {args.unet_model_finetune_path} does not match this model. "
                    f"Missing keys: {missing}. Unexpected keys: {unexpected}."
                )
            logger.info("[IPI-only finetune] Loaded full base UNet and the dedicated IPI Adapter.")
            if args.ipi_joint_motion:
                unet.ipi_adapter.enable_joint_motion(unet.config.block_out_channels)
        else:
            # 只加载时序层权重，保护 SVD 空间画功
            temporal_sd = {k: v for k, v in animator_sd.items()
                           if "temporal" in k or "ipi_adapter" in k}
            missing, unexpected = unet.load_state_dict(temporal_sd, strict=False)
            logger.info(f"[Finetune UNet] 加载了 {len(temporal_sd)} 个时序层参数")

            if args.ipi_model_finetune_path and os.path.exists(args.ipi_model_finetune_path):
                ipi_sd = torch.load(args.ipi_model_finetune_path, map_location="cpu")
                unet.ipi_adapter.load_state_dict(ipi_sd, strict=True)
                logger.info("[Finetune IPI] IPI Adapter 权重加载成功")

    # ─── 冻结策略 ─────────────────────────────────────────────────────────────
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    pose_net.requires_grad_(False)   # PoseNet 完全冻结
    face_encoder.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    image_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    face_encoder.to(accelerator.device, dtype=weight_dtype)

    if args.use_ema:
        ema_unet = EMAModel(unet.parameters(),
                            model_cls=UNetSpatioTemporalConditionModel,
                            model_config=unet.config)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers not available")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.per_gpu_batch_size
            * accelerator.num_processes
        )

    # ─── 训练策略：与 train_ipi_diaosu.py 完全一致（三轨制 LR）────────────────
    # IPI Adapter  → base_lr * ipi_lr_scale (default 3x)
    # UNet 时序层  → base_lr * 1.0
    # UNet 空间层  → base_lr * spatial_lr_scale (default 0.5x)
    # PoseNet / FaceEncoder → 冻结
    base_lr    = args.learning_rate
    adapter_lr = args.learning_rate * args.ipi_lr_scale
    spatial_lr = args.learning_rate * args.spatial_lr_scale

    unet_temporal_params = []
    unet_spatial_params  = []
    ipi_params           = []

    for name, para in unet.named_parameters():
        if args.ipi_joint_motion:
            if name.startswith("ipi_adapter.joint_motion."):
                para.requires_grad_(True)
                if name.startswith("ipi_adapter.joint_motion.readouts."):
                    ipi_params.append(para)
                else:
                    unet_temporal_params.append(para)
            continue
        is_ipi_residual    = "ipi_temporal_" in name
        is_ipi             = "ipi_adapter" in name
        if (is_ipi_residual and not args.ipi_temporal_v2
                or args.ipi_only_finetune and not (is_ipi or is_ipi_residual)):
            para.requires_grad_(False)
            continue
        is_time_mixer      = "time_mixer" in name or "time_pos_embed" in name
        is_temporal        = "temporal" in name or is_time_mixer
        is_spatial         = "attentions" in name and not is_temporal

        if is_ipi:
            para.requires_grad = True
            ipi_params.append(para)
        elif is_temporal:
            para.requires_grad = True
            unet_temporal_params.append(para)
        elif is_spatial:
            para.requires_grad = True
            unet_spatial_params.append(para)
        else:
            para.requires_grad = False

    # PoseNet 冻结（原版已训好，无需再动）
    pose_net.requires_grad_(False)

    parameters_list = [
        {"params": ipi_params,           "lr": adapter_lr},  # IPI Adapter (3x)
        {"params": unet_temporal_params, "lr": base_lr},     # 时序层 + time_mixer (1x)
    ]
    if args.ipi_only_finetune:
        if not ipi_params or not unet_temporal_params:
            raise ValueError("IPI-only finetuning needs both adapter and residual temporal parameters.")
    else:
        parameters_list.append({"params": unet_spatial_params, "lr": spatial_lr})

    if accelerator.is_main_process:
        print("\n" + "="*60)
        print("🎯 IPI-only 学习率分配" if args.ipi_only_finetune else "🎯 三轨制学习率分配确认")
        adapter_label = "Joint motion readouts" if args.ipi_joint_motion else "IPI Adapter"
        print(f"-> [1. {adapter_label}] {sum(p.numel() for p in ipi_params)/1e6:.2f} M | LR: {adapter_lr:.2e} ({args.ipi_lr_scale:.1f}x)")
        temporal_label = "IPI 时序门控和投影" if args.ipi_only_finetune else "UNet 时序层"
        if args.ipi_joint_motion:
            temporal_label = "Joint motion encoder"
        print(f"-> [2. {temporal_label}] {sum(p.numel() for p in unet_temporal_params)/1e6:.2f} M | LR: {base_lr:.2e} (1.0x)")
        if not args.ipi_only_finetune:
            print(f"-> [3. UNet 空间层 ] {sum(p.numel() for p in unet_spatial_params)/1e6:.2f} M | LR: {spatial_lr:.2e} ({args.spatial_lr_scale:.1f}x)")
        print(f"-> [🔒 PoseNet     ] 已冻结")
        print("="*60 + "\n")

    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    frozen_params    = sum(p.numel() for p in unet.parameters() if not p.requires_grad)
    logger.info(f"✅ 可训练参数 (UNet): {trainable_params / 1e6:.2f} M")
    logger.info(f"🔒 冻结参数   (UNet): {frozen_params / 1e6:.2f} M")

    # ─── 优化器 ───────────────────────────────────────────────────────────────
    if args.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        parameters_list,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # ─── 数据集 ───────────────────────────────────────────────────────────────
    if accelerator.is_main_process:
        logger.info("Training clip sampling: %d frames, source frame stride %d",
                    args.sample_n_frames, args.train_frame_stride)

    def make_dataset(txt_path):
        return LargeScaleAnimationVideos(
            root_path=args.data_root_path,
            txt_path=txt_path,
            width=512,
            height=512,
            n_sample_frames=args.sample_n_frames,
            sample_frame_rate=args.train_frame_stride,
            app=face_model.app,
            handler_ante=face_model.handler_ante,
            face_helper=face_model.face_helper,
            clip_image_processor=feature_extractor,
            target_frame_pose=args.target_frame_pose,
            ipi_limb_mask_prob=args.ipi_limb_mask_prob,
        )

    assert args.rec_data_path is not None, "--rec_data_path 不能为空！"
    datasets   = [make_dataset(args.rec_data_path)]
    dataloaders= []

    if args.vec_data_path:
        datasets.append(make_dataset(args.vec_data_path))

    if args.clay_data_path:
        # [FIX-3] 真正的过采样：用 ConcatDataset 让数据集 3 重复 dataset3_repeat 次
        ds3 = make_dataset(args.clay_data_path)
        ds3 = ConcatDataset([ds3] * max(1, args.dataset3_repeat))
        datasets.append(ds3)

    for ds in datasets:
        dataloaders.append(torch.utils.data.DataLoader(
            ds,
            batch_size=args.per_gpu_batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            drop_last=True,
        ))

    total_dataloader_len = sum(len(dl) for dl in dataloaders)
    num_update_steps_per_epoch = math.ceil(total_dataloader_len / args.gradient_accumulation_steps)
    overrode_max_train_steps = False
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # ─── Accelerator 准备 ─────────────────────────────────────────────────────
    composite_model = DeepSpeedWrapperModel(unet=unet, pose_net=pose_net)
    prepare_args = [composite_model, optimizer, lr_scheduler] + dataloaders
    prepared = accelerator.prepare(*prepare_args)
    composite_model = prepared[0]
    optimizer       = prepared[1]
    lr_scheduler    = prepared[2]
    dataloaders     = list(prepared[3:])

    unet = composite_model.module.unet if hasattr(composite_model, 'module') else composite_model.unet
    pose_net = composite_model.module.pose_net if hasattr(composite_model, 'module') else composite_model.pose_net
    face_encoder = face_encoder.to(accelerator.device)

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # The prepared dataloaders are sharded across processes. Training epochs and
    # resume offsets must use the lengths this process actually iterates over.
    total_dataloader_len = sum(len(dl) for dl in dataloaders)
    num_update_steps_per_epoch = math.ceil(total_dataloader_len / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("StableAnimator_IPI", config=vars(args))

    # ─── 辅助函数 ─────────────────────────────────────────────────────────────
    def encode_image(pixel_values):
        pixel_values = _resize_with_antialiasing(pixel_values, (224, 224))
        pixel_values = (pixel_values + 1.0) / 2.0
        pixel_values = pixel_values.to(torch.float32)
        pixel_values = feature_extractor(
            images=pixel_values,
            do_normalize=True, do_center_crop=False,
            do_resize=False, do_rescale=False,
            return_tensors="pt",
        ).pixel_values.to(device=accelerator.device, dtype=image_encoder.dtype)
        embeddings = image_encoder(pixel_values).image_embeds
        return embeddings.unsqueeze(1)

    def _get_add_time_ids(fps, motion_bucket_id, noise_aug_strength, dtype, batch_size, device):
        add_time_ids = torch.tensor(
            [[fps, motion_bucket_id, noise_aug_strength]],
            dtype=dtype, device=device
        ).repeat(batch_size, 1)
        return add_time_ids

    # ─── Checkpoint 恢复 ──────────────────────────────────────────────────────
    global_step = 0
    first_epoch = 0
    resume_step = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "latest":
            checkpoints = numbered_checkpoints(args.output_dir)
            checkpoint_dir = checkpoints[-1] if checkpoints else None
        else:
            checkpoint_dir = Path(args.resume_from_checkpoint)
            if checkpoint_dir.parent == Path("."):
                checkpoint_dir = Path(args.output_dir) / checkpoint_dir
            step = checkpoint_dir.name.removeprefix("checkpoint-")
            if not checkpoint_dir.is_dir() or not checkpoint_dir.name.startswith("checkpoint-") or not (step.isdigit() or step == "last"):
                raise ValueError(f"Expected an existing checkpoint directory, got {checkpoint_dir}.")

        if checkpoint_dir is None:
            logger.info("No checkpoint found, starting fresh.")
            args.resume_from_checkpoint = None
        else:
            suffix = checkpoint_dir.name.removeprefix("checkpoint-")
            if suffix == "last":
                step_file = checkpoint_dir / "global_step.txt"
                global_step = int(step_file.read_text().strip()) if step_file.is_file() else args.resume_last_step
                if global_step is None or global_step < 0:
                    raise ValueError("checkpoint-last has no valid recorded step. Set --resume_last_step to the completed step from the log.")
            else:
                global_step = int(suffix)
            first_epoch = global_step // num_update_steps_per_epoch
            unet_ckpt = checkpoint_dir / f"unet-{suffix}.pth"
            if unet_ckpt.is_file():
                logger.warning(
                    f"Restoring model weights from {checkpoint_dir}. This checkpoint has no saved "
                    "optimizer or scheduler state. Training starts at the beginning of the current epoch."
                )
                unwrapped = accelerator.unwrap_model(composite_model)
                unet_state = torch.load(unet_ckpt, map_location="cpu")
                if args.ipi_temporal_v2:
                    missing_v2 = [
                        name for name, _ in unwrapped.unet.named_parameters()
                        if "ipi_temporal_" in name and name not in unet_state
                    ]
                    if missing_v2:
                        raise ValueError(f"Cannot resume IPI v2: {unet_ckpt} is missing {missing_v2[0]}.")
                ipi_state = torch.load(checkpoint_dir / f"ipi_adapter-{suffix}.pth", map_location="cpu")
                unwrapped.unet.ipi_adapter.load_state_dict(ipi_state, strict=True)
                unet_state.update({f"ipi_adapter.{key}": value for key, value in ipi_state.items()})
                missing, unexpected = unwrapped.unet.load_state_dict(unet_state, strict=False)
                if not args.ipi_temporal_v2:
                    missing = [key for key in missing if "ipi_temporal_" not in key]
                if missing or unexpected:
                    raise ValueError(
                        f"UNet checkpoint {unet_ckpt} does not match this model. "
                        f"Missing keys: {missing}. Unexpected keys: {unexpected}."
                    )
                unwrapped.pose_net.load_state_dict(
                    torch.load(checkpoint_dir / f"pose_net-{suffix}.pth", map_location="cpu"),
                    strict=True,
                )
                face_encoder.load_state_dict(
                    torch.load(checkpoint_dir / f"face_encoder-{suffix}.pth", map_location="cpu"),
                    strict=True,
                )
            else:
                logger.info(f"Resuming full training state from {checkpoint_dir}.")
                accelerator.load_state(str(checkpoint_dir))
                resume_step = global_step % num_update_steps_per_epoch * args.gradient_accumulation_steps

    # ─── 训练主循环 ───────────────────────────────────────────────────────────
    progress_bar = tqdm(range(global_step, args.max_train_steps),
                        disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        # [FIX-3] PoseNet 冻结时用 eval()，防止 BN 统计量在训练模式下被污染
        pose_net.eval()
        face_encoder.eval()
        unet.train()

        # 构建当前 epoch 的数据索引序列（保证多卡顺序一致）
        len_per_dl = [len(dl) for dl in dataloaders]
        index_list = []
        for dl_idx, length in enumerate(len_per_dl):
            index_list.extend([dl_idx] * length)

        g = torch.Generator()
        g.manual_seed((args.seed if args.seed else 42) + epoch)
        shuffled_indices = torch.randperm(len(index_list), generator=g).tolist()
        index_list = [index_list[i] for i in shuffled_indices]

        iters = [iter(dl) for dl in dataloaders]

        for step, dl_idx in enumerate(index_list):
            try:
                batch = next(iters[dl_idx])
            except StopIteration:
                iters[dl_idx] = iter(dataloaders[dl_idx])
                batch = next(iters[dl_idx])

            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(composite_model):
                with accelerator.autocast():
                    pixel_values = batch["pixel_values"].to(device=accelerator.device, dtype=weight_dtype,
                                                            non_blocking=True)
                    cond_pixel_values = batch["reference_image"].to(device=accelerator.device, dtype=weight_dtype,
                                                                    non_blocking=True)
                    latents = tensor_to_vae_latent(pixel_values, vae).to(weight_dtype)

                    encoder_hidden_states = encode_image(cond_pixel_values).to(weight_dtype)
                    image_embed = encoder_hidden_states.clone()

                    # 1. DWPose 原始关键点坐标（已在 Dataset 中归一化到 [-1, 1]）
                    dwpose_raw = batch["dwpose_raw"].to(device=accelerator.device, dtype=weight_dtype)

                    # 2. CLIP Image Embeddings（驱动视频帧的外观特征）
                    driving_clip_pixels = batch["driving_clip_pixel_values"].to(
                        device=accelerator.device, dtype=weight_dtype, non_blocking=True)
                    b_size, n_frames, c_c, h_c, w_c = driving_clip_pixels.shape
                    driving_clip_pixels = driving_clip_pixels.reshape(b_size * n_frames, c_c, h_c, w_c)

                    with torch.no_grad():
                        clip_image_embeds = image_encoder(driving_clip_pixels).image_embeds
                        clip_image_embeds = clip_image_embeds.reshape(b_size, n_frames, -1)

                    bsz = latents.shape[0]

                    # 3. 条件图像噪声增强（打破 SVD 的复制粘贴惰性）
                    cond_pixel_values = cond_pixel_values + args.train_noise_aug * randn_tensor(
                        cond_pixel_values.shape, generator=generator,
                        device=cond_pixel_values.device, dtype=cond_pixel_values.dtype)
                    conditional_latents = tensor_to_vae_latent(cond_pixel_values, vae, scale=False)

                    # 4. 扩散噪声 & 时间步
                    noise = torch.randn_like(latents)
                    sigmas = rand_cosine_interpolated(
                        shape=[bsz], image_d=image_d, noise_d_low=noise_d_low,
                        noise_d_high=noise_d_high, sigma_data=sigma_data,
                        min_value=min_value, max_value=max_value,
                    ).to(latents.device, dtype=weight_dtype)

                    sigmas_reshaped = sigmas.clone()
                    while len(sigmas_reshaped.shape) < len(latents.shape):
                        sigmas_reshaped = sigmas_reshaped.unsqueeze(-1)

                    noisy_latents = latents + noise * sigmas_reshaped
                    timesteps = torch.Tensor([0.25 * sigma.log() for sigma in sigmas]).to(
                        latents.device, dtype=weight_dtype)
                    inp_noisy_latents = noisy_latents / ((sigmas_reshaped ** 2 + 1) ** 0.5)

                    # 5. Motion Bucket（基于真实运动幅度动态分配）
                    with torch.no_grad():
                        pose_diff = (dwpose_raw[:, 1:] - dwpose_raw[:, :-1]).abs().mean().item()
                    low_bound, high_bound = 0.005, 0.050
                    motion_score = max(low_bound, min(high_bound, pose_diff))
                    motion_score_norm = (motion_score - low_bound) / (high_bound - low_bound + 1e-6)
                    dynamic_bucket = 80.0 + motion_score_norm * (160.0 - 80.0)

                    added_time_ids = _get_add_time_ids(
                        fps=14,
                        motion_bucket_id=dynamic_bucket,
                        noise_aug_strength=args.train_noise_aug,
                        dtype=encoder_hidden_states.dtype,
                        batch_size=bsz,
                        device=latents.device,
                    )

                    # 6. 条件 Dropout（CFG 训练核心）
                    if args.conditioning_dropout_prob > 0:
                        random_p = torch.rand(bsz, device=latents.device, generator=generator)

                        # 图像 embedding Dropout（全条件 -> 无条件）
                        prompt_mask = (random_p < 2 * args.conditioning_dropout_prob).reshape(bsz, 1, 1)
                        encoder_hidden_states = torch.where(
                            prompt_mask, torch.zeros_like(encoder_hidden_states), encoder_hidden_states)

                        image_mask = 1 - (
                            (random_p >= args.conditioning_dropout_prob).float()
                            * (random_p < 3 * args.conditioning_dropout_prob).float()
                        )
                        conditional_latents = image_mask.reshape(bsz, 1, 1, 1) * conditional_latents

                        # Pose Dropout（基于伪随机，避免多卡种子冲突死锁）
                        drop_mask_1d = torch.zeros(bsz, dtype=torch.bool, device=latents.device)
                        for b_idx in range(bsz):
                            pseudo_rand = ((global_step * bsz + b_idx) * 37 + (args.seed or 0)) % 100
                            if pseudo_rand < int(args.conditioning_dropout_prob * 100):
                                drop_mask_1d[b_idx] = True

                        drop_mask_pose = drop_mask_1d.view(bsz, *[1] * (len(latents.shape) - 1))
                        # DWPose 和 Pose Dropout 同步
                        drop_mask_seq = drop_mask_1d.view(bsz, *[1] * (dwpose_raw.dim() - 1))
                        dwpose_raw = torch.where(drop_mask_seq, torch.zeros_like(dwpose_raw), dwpose_raw)

                    # [FIX-1] CLIP 使用独立的高概率 Dropout（默认 50%）
                    # diaosu 原版 CLIP 和 Pose 共用同一个 mask，导致 CLIP 静态特征无法被充分 Dropout
                    # 修复后 CLIP 独立掉落，迫使网络不依赖静态外观来降 Loss
                    drop_clip = torch.rand(bsz, device=latents.device) < args.clip_dropout_prob
                    drop_clip_seq = drop_clip.view(bsz, *[1] * (clip_image_embeds.dim() - 1))
                    clip_image_embeds = torch.where(
                        drop_clip_seq, torch.zeros_like(clip_image_embeds), clip_image_embeds)

                    # 7. PoseNet 提取姿态潜变量
                    pose_pixels = batch["pose_pixels"].to(device=accelerator.device, dtype=weight_dtype,
                                                          non_blocking=True)
                    pose_latents = pose_net(pose_pixels).to(latents.dtype)

                    # Pose Dropout 同步到 pose_latents
                    if args.conditioning_dropout_prob > 0:
                        drop_mask_pose_lat = drop_mask_1d.view(bsz, *[1] * (pose_latents.dim() - 1))
                        pose_latents = torch.where(
                            drop_mask_pose_lat, torch.zeros_like(pose_latents), pose_latents)

                    # 8. 人脸 embedding
                    faceid_embeds = batch["faceid_embeds"].to(device=accelerator.device, dtype=weight_dtype,
                                                              non_blocking=True)
                    pseudo_rand_face = ((global_step * 73) + (args.seed or 0)) % 100
                    if pseudo_rand_face < 15:
                        # 15% 概率随机掐断人脸特征，逼迫模型依靠骨架运动
                        faceid_latents = torch.zeros(
                            image_embed.shape[0], 4, image_embed.shape[-1],
                            device=image_embed.device, dtype=image_embed.dtype)
                    else:
                        faceid_latents = face_encoder(faceid_embeds, image_embed)

                    # 9. 组装输入
                    conditional_latents = conditional_latents.unsqueeze(1).repeat(
                        1, noisy_latents.shape[1], 1, 1, 1)
                    inp_noisy_latents = torch.cat([inp_noisy_latents, conditional_latents], dim=2)
                    inp_noisy_latents = inp_noisy_latents.to(latents.dtype)
                    encoder_hidden_states = torch.cat(
                        [encoder_hidden_states, faceid_latents], dim=1).to(latents.dtype)

                    # 10. UNet 前向
                    probe = (not args.ipi_joint_motion and args.ipi_limb_mask_prob > 0
                             and accelerator.is_main_process and global_step % 250 == 0)
                    with measure_mid_ipi(unet, n_frames, global_step, enabled=probe):
                        model_pred = composite_model(
                            inp_noisy_latents, timesteps, encoder_hidden_states,
                            added_time_ids=added_time_ids,
                            pose_latents=pose_latents,
                            dwpose_raw=dwpose_raw,
                            clip_image_embeds=clip_image_embeds,
                            ipi_temporal_v2=args.ipi_temporal_v2,
                        ).sample

                    # 11. 损失计算
                    c_out = -sigmas_reshaped / ((sigmas_reshaped ** 2 + 1) ** 0.5)
                    c_skip = 1 / (sigmas_reshaped ** 2 + 1)
                    denoised_latents = model_pred * c_out + c_skip * noisy_latents
                    weighing = (1 + sigmas_reshaped ** 2) * (sigmas_reshaped ** -2.0)

                    # 面部加权 mask
                    tgt_face_masks = batch["tgt_face_masks"].to(device=accelerator.device, dtype=weight_dtype,
                                                                non_blocking=True)
                    tgt_face_masks = rearrange(tgt_face_masks, "b f c h w -> (b f) c h w")
                    tgt_face_masks = F.interpolate(tgt_face_masks,
                                                   size=(latents.size()[-2], latents.size()[-1]),
                                                   mode='nearest')
                    tgt_face_masks = rearrange(tgt_face_masks, "(b f) c h w -> b f c h w",
                                               f=args.sample_n_frames)
                    face_weight = 1.0 + tgt_face_masks

                    loss = torch.mean(
                        (weighing.float() * (denoised_latents.float() - latents.float()) ** 2
                         * face_weight).reshape(latents.shape[0], -1),
                        dim=1,
                    ).mean()

                    # [FIX-5] 材质一致性约束（diaosu 原版系数为 0 → 修复为可配置）
                    # 第一帧 denoised latent 应该尽量接近干净的第一帧真值
                    ref_latent = latents[:, 0:1].detach()
                    pred_first = denoised_latents[:, 0:1]
                    consistency_loss = F.l1_loss(pred_first, ref_latent)
                    loss = loss + args.consistency_loss_weight * consistency_loss

                    avg_loss = accelerator.gather(loss.repeat(args.per_gpu_batch_size)).mean()
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            # ─── 同步更新 ─────────────────────────────────────────────────────
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": avg_loss.item()}, step=global_step)
                if args.ipi_joint_motion and accelerator.is_main_process and (global_step == 50 or global_step % 500 == 0):
                    readouts = unet.ipi_adapter.joint_motion.readouts
                    rms = {c: head.weight.detach().float().square().mean().sqrt().item() for c, head in readouts.items()}
                    logger.info("Joint readout RMS at step %d (zero at fresh initialization): %s", global_step, rms)

                # ─── 保存 Checkpoint ──────────────────────────────────────────
                if global_step % args.checkpointing_steps == 0:
                    accelerator.wait_for_everyone()
                    full_composite_dict  = accelerator.get_state_dict(composite_model)
                    full_face_enc_dict   = accelerator.get_state_dict(face_encoder)

                    if accelerator.is_main_process:
                        # 清理旧 checkpoint
                        if args.checkpoints_total_limit is not None:
                            ckpts = numbered_checkpoints(args.output_dir)
                            if len(ckpts) >= args.checkpoints_total_limit:
                                for old in ckpts[:len(ckpts) - args.checkpoints_total_limit + 1]:
                                    shutil.rmtree(old)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_path, exist_ok=True)

                        full_unet_dict = {k.replace("unet.", ""): v
                                          for k, v in full_composite_dict.items()
                                          if k.startswith("unet.")}
                        full_pose_dict = {k.replace("pose_net.", ""): v
                                          for k, v in full_composite_dict.items()
                                          if k.startswith("pose_net.")}

                        # 单独保存 IPI Adapter
                        ipi_sd = {k.replace("ipi_adapter.", ""): v
                                  for k, v in full_unet_dict.items()
                                  if k.startswith("ipi_adapter.")}
                        torch.save(ipi_sd, os.path.join(save_path, f"ipi_adapter-{global_step}.pth"))

                        # 保存去除 ipi_adapter 的纯净 UNet（方便恢复原版推理）
                        clean_unet_sd = {k: v for k, v in full_unet_dict.items()
                                         if not k.startswith("ipi_adapter.")}
                        torch.save(clean_unet_sd, os.path.join(save_path, f"unet-{global_step}.pth"))
                        torch.save(full_pose_dict, os.path.join(save_path, f"pose_net-{global_step}.pth"))
                        torch.save(full_face_enc_dict, os.path.join(save_path, f"face_encoder-{global_step}.pth"))
                        logger.info(f"Checkpoint saved to {save_path}")

                    accelerator.wait_for_everyone()

                # ─── 验证 ────────────────────────────────────────────────────
                if accelerator.is_main_process and global_step % args.validation_steps == 0:
                    if args.validation_image and args.validation_control_folder:
                        log_validation(
                            vae=vae, image_encoder=image_encoder,
                            unet=unet, pose_net=pose_net,
                            face_encoder=face_encoder,
                            app=face_model.app,
                            face_helper=face_model.face_helper,
                            handler_ante=face_model.handler_ante,
                            scheduler=noise_scheduler,
                            accelerator=accelerator,
                            feature_extractor=feature_extractor,
                            width=512, height=768,
                            torch_dtype=weight_dtype,
                            validation_image_folder=args.validation_image_folder,
                            validation_image=args.validation_image,
                            validation_control_folder=args.validation_control_folder,
                            output_dir=args.output_dir,
                            generator=generator,
                            global_step=global_step,
                            num_frames=args.num_frames,
                            pose_data_path=args.pose_data_path,
                            ipi_temporal_v2=args.ipi_temporal_v2,
                        )

            accelerator.wait_for_everyone()

            current_lrs = lr_scheduler.get_last_lr()
            logs = {
                "loss":   f"{loss.detach().item():.4f}",
                "lr_readout" if args.ipi_joint_motion else "lr_ipi": f"{current_lrs[0]:.2e}",
                "lr_joint_encoder" if args.ipi_joint_motion else "lr_tmp": f"{current_lrs[1]:.2e}",
            }
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # ─── 最终保存 ─────────────────────────────────────────────────────────────
    accelerator.wait_for_everyone()
    full_composite_dict = accelerator.get_state_dict(composite_model)
    full_face_enc_dict  = accelerator.get_state_dict(face_encoder)

    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, "checkpoint-last")
        os.makedirs(save_path, exist_ok=True)

        full_unet_dict = {k.replace("unet.", ""): v
                          for k, v in full_composite_dict.items() if k.startswith("unet.")}
        full_pose_dict = {k.replace("pose_net.", ""): v
                          for k, v in full_composite_dict.items() if k.startswith("pose_net.")}

        ipi_sd = {k.replace("ipi_adapter.", ""): v
                  for k, v in full_unet_dict.items() if k.startswith("ipi_adapter.")}
        torch.save(ipi_sd, os.path.join(save_path, "ipi_adapter-last.pth"))

        clean_unet_sd = {k: v for k, v in full_unet_dict.items()
                         if not k.startswith("ipi_adapter.")}
        torch.save(clean_unet_sd, os.path.join(save_path, "unet-last.pth"))
        torch.save(full_pose_dict, os.path.join(save_path, "pose_net-last.pth"))
        torch.save(full_face_enc_dict, os.path.join(save_path, "face_encoder-last.pth"))
        (Path(save_path) / "global_step.txt").write_text(f"{global_step}\n", encoding="utf-8")
        logger.info(f"Final checkpoint saved to {save_path}")


if __name__ == "__main__":
    main()
