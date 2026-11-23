#!/usr/bin/env python
# -*- coding: utf-8 -*-
from torch.utils.data import ConcatDataset
import argparse
import random
import logging
import math
import os
os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import torchvision.transforms as T
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import shutil
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
import PIL
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from diffusers.models.attention_processor import XFormersAttnProcessor

from animation.dataset.animation_dataset_ipi import LargeScaleAnimationVideos  #数据集
from animation.modules.attention_processor import AnimationAttnProcessor   #注意力处理器
from animation.modules.attention_processor_normalized import AnimationIDAttnNormalizedProcessor  #归一化注意力处理器
from animation.modules.face_model import FaceModel  #面部模型
from animation.modules.id_encoder import FusionFaceId  #ID 编码器
from animation.modules.pose_net import PoseNet  #姿态网络
from animation.modules.unet_ipi import UNetSpatioTemporalConditionModel  #时空条件 UNet
from animation.pipelines.inference_pipeline_animation_ipi import InferenceAnimationPipeline
#from animation.pipelines.validation_pipeline_animation import ValidationAnimationPipeline  #验证流水线
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from einops import rearrange

import datetime
import diffusers
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available, load_image
from diffusers.utils.import_utils import is_xformers_available
import warnings
import torch.nn as nn
from diffusers.utils.torch_utils import randn_tensor

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0.dev0")

logger = get_logger(__name__, log_level="INFO")



#i should make a utility function file 统一处理输入的图像和尺寸
def validate_and_convert_image(image, target_size=(256, 256)):
    # 检查输入图像是否为None（空值）
    if image is None:
        print("Encountered a None image")
        return None
    # 处理PyTorch张量类型的图像（模型输出通常为此类型）
    if isinstance(image, torch.Tensor):
        # Convert PyTorch tensor to PIL Image将PyTorch张量转换为PIL Image
        # 检查张量维度是否为3（CxHxW格式：通道数x高度x宽度），且通道数为1（灰度）或3（RGB
        if image.ndim == 3 and image.shape[0] in [1, 3]:  # Check for CxHxW format
            # 若为单通道灰度图，复制为3通道（模拟RGB格式，避免后续处理报错）
            if image.shape[0] == 1:  # Convert single-channel grayscale to RGB
                image = image.repeat(3, 1, 1)# 沿着通道维度复制3次
            # 张量转图像步骤：
            # 1. 乘以255（将模型输出的[0,1]范围映射到像素值[0,255]）
            # 2. 裁剪到[0,255]区间（防止溢出）
            # 3. 转换为字节类型（8位整数，符合图像存储格式）
            # 4. 调整维度顺序（从CxHxW转为HxWxC，符合PIL图像格式）
            # 5. 移动到CPU并转为numpy数组
            # 6. 从numpy数组创建PIL Image对象
            image = image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            image = Image.fromarray(image)
        else: # 若张量维度或通道数不符合要求，打印错误信息并返回None
            print(f"Invalid image tensor shape: {image.shape}")
            return None
    # 处理PIL Image类型的图像（数据集中的图像通常为此类型
    elif isinstance(image, Image.Image):
        # Resize PIL Image # 注释：调整PIL图像的尺寸到目标大小（默认256x256）
        image = image.resize(target_size)
    else: # 处理其他不支持的图像类型
        print("Image is not a PIL Image or a PyTorch tensor")
        return None
    # 返回处理后的图像（PIL Image类型，尺寸为target_size）
    return image


# 定义创建图像网格的函数，参数包括：
# images：待组合的图像列表（可混合PIL Image或PyTorch张量）
# rows：网格的行数
# cols：网格的列数
# target_size：每个图像的目标尺寸（默认256x256）
def create_image_grid(images, rows, cols, target_size=(256, 256)):
    # 1. 预处理图像：调用工具函数将所有图像统一转换为PIL Image并调整到目标尺寸
    # 列表推导式遍历输入图像，对每个图像执行格式验证和转换
    valid_images = [validate_and_convert_image(img, target_size) for img in images]
    # 2. 过滤无效图像：移除转换失败的None值，确保后续处理的图像都是有效的
    valid_images = [img for img in valid_images if img is not None]
    # 3. 检查有效图像是否存在：若所有图像都无效，打印提示并返回None
    if not valid_images:
        print("No valid images to create a grid")
        return None
    # 4. 准备网格画布：
    # 从目标尺寸中提取宽(w)和高(h)
    w, h = target_size
    # 创建一个新的RGB模式图像作为网格画布，尺寸为（列数×单图宽，行数×单图高）
    grid = Image.new('RGB', size=(cols * w, rows * h))
    
    # 5. 拼接图像到网格：遍历有效图像列表，使用索引i确定每个图像在网格中的位置
    for i, image in enumerate(valid_images):
        # 计算图像在网格中的左上角坐标：
        # 列坐标 = (i % cols) × 单图宽度（取余得到所在列）
        # 行坐标 = (i // cols) × 单图高度（整除得到所在行）
        grid.paste(image, box=((i % cols) * w, (i // cols) * h))

    return grid

def save_combined_frames(batch_output, validation_images, validation_control_images,output_folder):#多个图像帧保存成一个网格图像并保存
    # Flatten batch_output, which is a list of lists of PIL Images
    #batch_output 是嵌套列表（例如：[[img1, img2], [img3, img4]]），通过列表推导式展平为一维列表（[img1, img2, img3, img4]）。
    flattened_batch_output = [img for sublist in batch_output for img in sublist]

    # Combine frames into a list without converting (since they are already PIL Images)将验证图像、控制图像、模型输出结果按顺序合并成一个列表，便于统一处理
    combined_frames = validation_images + validation_control_images + flattened_batch_output

    # Calculate rows and columns for the grid
    num_images = len(combined_frames)
    cols = 3  # adjust number of columns as needed
    rows = (num_images + cols - 1) // cols   ## 计算需要的行数（向上取整）
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  #使用当前时间（年月日 - 时分秒）作为文件名后缀，确保每次保存的文件不重名

    filename = f"combined_frames_{timestamp}.png" 
    # Create and save the grid image调用 create_image_grid 函数，将所有图像按 rows×cols 的网格排列
    grid = create_image_grid(combined_frames, rows, cols)  
    output_folder = os.path.join(output_folder, "validation_images") #在指定的输出目录下创建 validation_images 子目录，exist_ok=True 确保目录已存在时不报错。
    os.makedirs(output_folder, exist_ok=True)
    
    # Now define the full path for the file
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"combined_frames_{timestamp}.png"
    #拼接完整保存路径，若网格图像创建成功则保存，否则打印错误信息
    output_loc = os.path.join(output_folder, filename)
    
    if grid is not None:
        grid.save(output_loc)
    else:
        print("Failed to create image grid")

def load_images_from_folder(folder):#从指定文件夹中加载所有 PNG 图像，并按照文件名中的数字序号进行排序
    images = []

    files = os.listdir(folder)
    png_files = [f for f in files if f.endswith('.png')]
    png_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    for filename in png_files:
        img = Image.open(os.path.join(folder, filename)).convert('RGB')
        images.append(img)

    return images



# copy from https://github.com/crowsonkb/k-diffusion.git  从均匀分布中抽取分层样本
def stratified_uniform(shape, group=0, groups=1, dtype=None, device=None):
    """Draws stratified samples from a uniform distribution."""
    if groups <= 0: #确保分组数 groups 为正数，且当前分组 group 在有效范围内（[0, groups)），否则抛出错误。
        raise ValueError(f"groups must be positive, got {groups}")
    if group < 0 or group >= groups:
        raise ValueError(f"group must be in [0, {groups})")
    n = shape[-1] * groups  #shape[-1] 是样本的最后一个维度大小（即单组样本数），乘以总分组数 groups 后，得到整个采样区间的总划分粒度 n
    offsets = torch.arange(group, n, groups, dtype=dtype, device=device)  #torch.arange(start, end, step) 生成等差数列：从 group 开始，到 n 结束，步长为 groups。
    u = torch.rand(shape, dtype=dtype, device=device)   #生成形状为 shape 的标准均匀分布随机数（取值范围 [0,1)），作为每个样本在基础偏移位置上的微小扰动。
    return (offsets + u) / n


def rand_cosine_interpolated(shape, image_d, noise_d_low, noise_d_high, sigma_data=1., min_value=1e-3, max_value=1e3, device='cpu', dtype=torch.float32):
    """Draws samples from an interpolated cosine timestep distribution (from simple diffusion)."""
    def logsnr_schedule_cosine(t, logsnr_min, logsnr_max):#计算logsnr  通过反正切函数将 logsnr 映射到 [0,1] 区间的 t，再通过正切函数反转得到余弦分布的 logsnr
        t_min = math.atan(math.exp(-0.5 * logsnr_max))
        t_max = math.atan(math.exp(-0.5 * logsnr_min))
        return -2 * torch.log(torch.tan(t_min + t * (t_max - t_min)))

    def logsnr_schedule_cosine_shifted(t, image_d, noise_d, logsnr_min, logsnr_max):#余弦调度的 logsnr 进行偏移（基于 image_d 和 noise_d 的比例）：
        shift = 2 * math.log(noise_d / image_d)
        return logsnr_schedule_cosine(t, logsnr_min - shift, logsnr_max - shift) + shift

    def logsnr_schedule_cosine_interpolated(t, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max):
        #在 noise_d_low 和 noise_d_high 对应的 logsnr 之间插值：
        #用 torch.lerp 基于 t 进行线性插值，实现两个噪声维度的平滑过渡。
        logsnr_low = logsnr_schedule_cosine_shifted(
            t, image_d, noise_d_low, logsnr_min, logsnr_max)
        logsnr_high = logsnr_schedule_cosine_shifted(
            t, image_d, noise_d_high, logsnr_min, logsnr_max)
        return torch.lerp(logsnr_low, logsnr_high, t)
    #1.计算 logsnr_min 和 logsnr_max：
    logsnr_min = -2 * math.log(min_value / sigma_data)
    logsnr_max = -2 * math.log(max_value / sigma_data)
    #2.生成均匀分布的 u
    u = stratified_uniform(
        shape, group=0, groups=1, dtype=dtype, device=device
    )
    #3.调用 logsnr_schedule_cosine_interpolated 得到插值后的 logsnr。
    logsnr = logsnr_schedule_cosine_interpolated(
        u, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max)
    #4.将 logsnr 转换为 sigma 并返回：
    return torch.exp(-logsnr / 2) * sigma_data
    #若输入 shape=(2,1)，则返回形状为 (2,1) 的 sigma 张量，每个元素对应一个采样的噪声水平。

def rand_log_normal(shape, loc=0., scale=1., device='cpu', dtype=torch.float32):#从对数正态分布（log-normal distribution） 中生成样本。
    """Draws samples from an lognormal distribution."""
    # 生成[1e-7, 1-1e-7)区间的均匀分布随机数，避免ICDF在0或1处的数值问题
    u = torch.rand(shape, dtype=dtype, device=device) * (1 - 2e-7) + 1e-7
    return torch.distributions.Normal(loc, scale).icdf(u).exp()

min_value = 0.002
max_value = 700
image_d = 64
noise_d_low = 32
noise_d_high = 64
sigma_data = 0.5

def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]
    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)#使用计算好的核大小 ks 和标准差 sigmas 对输入图像进行高斯模糊，调用内部函数 _gaussian_blur2d 实现
    output = torch.nn.functional.interpolate(
        input, size=size, mode=interpolation, align_corners=align_corners)
    return output

def _compute_padding(kernel_size):
    if len(kernel_size) < 2:# kernel_size 的长度是否≥2（需包含高度和宽度），否则抛出异常
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size] # 例如：kernel_size=(3,3) → computed=[2,2]
    out_padding = 2 * len(kernel_size) * [0] # 长度为 2×维度数，例如2D核→长度4
    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)] # 从最后一个维度开始（如宽度）
        pad_front = computed_tmp // 2  # 前半部分填充
        pad_rear = computed_tmp - pad_front   # 后半部分填充
        # 按 (左,右,上,下) 顺序赋值（2D为例：i=0→宽度，i=1→高度）
        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding

def _filter2d(input, kernel):
    # prepare kernel
    b, c, h, w = input.shape# [批次, 通道数, 高, 宽]
    tmp_kernel = kernel[:, None, ...].to(
        device=input.device, dtype=input.dtype)
    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)  

    height, width = tmp_kernel.shape[-2:] # 内核的高和宽
    padding_shape: list[int] = _compute_padding([height, width])     #填充
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    output = torch.nn.functional.conv2d(
        input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)
    out = output.view(b, c, h, w)
    return out

def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])# 转换为张量，形状 (1, 1)
    batch_size = sigma.shape[0]# sigma 形状通常为 (batch_size, 1)
    x = (torch.arange(window_size, device=sigma.device,
         dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)
    if window_size % 2 == 0:
        x = x + 0.5  

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))
    return gauss / gauss.sum(-1, keepdim=True)

def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):#sigma 是元组（如 (2.0, 1.5)），将其转换为张量，形状为 (1, 2)
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1]) 
    bs = sigma.shape[0] 
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1)) 
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))  
    out_x = _filter2d(input, kernel_x[..., None, :])  
    out = _filter2d(out_x, kernel_y[..., None])       

    return out

def export_to_video(video_frames, output_video_path, fps):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")#指定视频的编码格式
    h, w, _ = video_frames[0].shape #第一帧图像中获取视频的高度 h 和宽度 w
    video_writer = cv2.VideoWriter( #创建视频写入器
        output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for i in range(len(video_frames)):#写入视频帧
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)

def export_to_gif(frames, output_gif_path, fps):
    pil_frames = [Image.fromarray(frame) if isinstance(
        frame, np.ndarray) else frame for frame in frames]

    pil_frames[0].save(output_gif_path.replace('.mp4', '.gif'), # 处理输出路径，确保后缀为 .gif
                       format='GIF',                              ## 指定格式为 GIF
                       append_images=pil_frames[1:],                # 追加后续所有帧
                       save_all=True,                    # 保存所有帧（而非仅第一帧）
                       duration=125,                  # 每帧持续时间（毫秒）
                       loop=0)                   # 0 表示无限循环播放

def tensor_to_vae_latent(t, vae, scale=True):
    t = t.to(vae.dtype)  #t：输入张量
    if len(t.shape) == 5: #5维张量
        video_length = t.shape[1]  # 获取视频的帧数（f）
        t = rearrange(t, "b f c h w -> (b f) c h w")
        latents = vae.encode(t).latent_dist.sample()
        latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)
    elif len(t.shape) == 4:
        latents = vae.encode(t).latent_dist.sample()
    if scale:
        latents = latents * vae.config.scaling_factor#缩放因子
    return latents


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train Stable Diffusion XL for InstructPix2Pix."
    )
    parser.add_argument(
        "--clay_data_path",
        type=str,
        default=None,
        help="Path to the clay dataset path",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default='ubc',
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=500,
        help=(
            "Run fine-tuning validation every X epochs."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--per_gpu_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=0,
        help="Conditioning dropout probability. Drops out the conditionings (image and edit prompt) used in training InstructPix2Pix.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs."
        ),
    )
    parser.add_argument(
        "--use_ema", action="store_true", help="Whether to use EMA model."
    )
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier."
        ),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.95,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16)."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=1,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--log_trainable_parameters",
        action="store_true",
        help="Whether to write the trainable parameters.",
    )
    parser.add_argument(
        "--pretrain_unet",
        type=str,
        default=None,
        help="use weight for unet block",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=256,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help=(
            "path to the dataset csv"
        ),
    )
    parser.add_argument(
        "--video_folder",
        type=str,
        default=None,
        help=(
            "path to the video folder"
        ),
    )
    parser.add_argument(
        "--condition_folder",
        type=str,
        default=None,
        help=(
            "path to the depth folder"
        ),
    )
    parser.add_argument(
        "--ipi_model_finetune_path",
        type=str,
        default=None,
        help="Path to the pretrained IPI adapter model",
    )
    parser.add_argument(
        "--motion_folder",
        type=str,
        default=None,
        help=(
            "path to the depth folder"
        ),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
        ),
    )
    parser.add_argument(
        "--validation_image_folder",
        type=str,
        default=None,
        help=(
            "参考视频的视频帧文件夹"
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        help=(
            "A set of paths to the controlnext conditioning image be evaluated every `--validation_steps`"
        ),
    )
    parser.add_argument(
        "--validation_control_folder",
        type=str,
        default=None,
        help=(
            "the validation control image poses"
        ),
    )
    parser.add_argument(
        "--sample_n_frames",
        type=int,
        default=14,
        help=(
            "the sample_n_frames"
        ),
    )

    parser.add_argument(
        "--ref_augment",
        action="store_true",
        help=(
            "use augment for the reference image"
        ),
    )
    parser.add_argument(
        "--train_stage",
        type=int,
        default=2,
        help=(
            "the training stage"
        ),
    )

    parser.add_argument(
        "--posenet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained posenet model",
    )
    parser.add_argument(
        "--face_encoder_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained face encoder model",
    )
    parser.add_argument(
        "--unet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained unet model",
    )

    parser.add_argument(
        "--data_root_path",
        type=str,
        default=None,
        help="Path to the data root path",
    )
    parser.add_argument(
        "--rec_data_path",
        type=str,
        default=None,
        help="Path to the rec data path",
    )
    parser.add_argument(
        "--vec_data_path",
        type=str,
        default=None,
        help="Path to the vec data path",
    )

    parser.add_argument(
        "--finetune_mode",
        type=bool,
        default=False,
        help="Enable or disable the finetune mode (True/False).",
    )
    parser.add_argument(
        "--posenet_model_finetune_path",
        type=str,
        default=None,
        help="Path to the pretrained posenet model",
    )
    parser.add_argument(
        "--face_encoder_finetune_path",
        type=str,
        default=None,
        help="Path to the pretrained face encoder",
    )
    parser.add_argument(
        "--unet_model_finetune_path",
        type=str,
        default=None,
        help="Path to the pretrained unet model",
    )
    parser.add_argument(
        "--pose_data_path",
        type=str,
        default=None,
        help="Path to the .npy file containing raw dwpose data (Frames, 18, 2).",
    )
    parser.add_argument(
        "--is_statue",
        action="store_true",
        default=True,
        help="Whether to enable statue-specific handling in validation.",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def download_image(url):
    original_image = (
        lambda image_url_or_path: load_image(image_url_or_path)
        if urlparse(image_url_or_path).scheme
        else PIL.Image.open(image_url_or_path).convert("RGB")
    )(url)
    return original_image


class DeepSpeedWrapperModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        for name, value in kwargs.items():
            assert isinstance(value, nn.Module)
            self.register_module(name, value)


def main():

    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', module='albumentations')
    torch.multiprocessing.set_start_method('spawn')  

    args = parse_args() 

    if args.non_ema_revision is not None:  
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir)
    
    from accelerate.utils import DistributedDataParallelKwargs
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(  
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
        log_with=args.report_to,
        kwargs_handlers=[ddp_kwargs],
    )

    generator = torch.Generator(
        device=accelerator.device).manual_seed(23123134)

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    print(args.pretrained_model_name_or_path)
    feature_extractor = CLIPImageProcessor.from_pretrained(args.pretrained_model_name_or_path, subfolder="feature_extractor", revision=args.revision)
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="image_encoder", revision=args.revision
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant="fp16")
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        args.pretrained_model_name_or_path if args.pretrain_unet is None else args.pretrain_unet,
        subfolder="unet",
        low_cpu_mem_usage=False,
        device_map=None,
        variant="fp16",
        ignore_mismatched_sizes=True  # 添加这行！
    )
    # -------------------------------------------------------
    # === [IPI Mod] 初始化 IPI Adapter ===
    # -------------------------------------------------------
    from animation.modules.ipi_adapter import MotionAdapter

    ipi_adapter = MotionAdapter(
        cross_attention_dim=getattr(unet.config, "cross_attention_dim", 1024),
        clip_dim=1024,                 
        num_tokens=8                   
    )
    zero_init_count = 0
    for name, param in ipi_adapter.named_parameters():
        # 这里假设你的最终输出层叫 clip_proj.proj，如果是别的请替换
        # 你可以通过 print(name) 看看倒数第一层的名字到底是什么
        if "clip_proj.proj_out.weight" in name or "clip_proj.proj_out.bias" in name: 
            torch.nn.init.zeros_(param)
            print(f"🛡️ [黑科技激活] 成功零初始化输出层: {name}")
            zero_init_count += 1
            
    if zero_init_count == 0:
        print("⚠️ 警告：没有找到对应的 proj 层，零初始化可能失败，请检查 layer 命名！")
    unet.ipi_adapter = ipi_adapter
    ipi_adapter.to(unet.device, dtype=unet.dtype)
    # -------------------------------------------------------
    pose_net = PoseNet(noise_latent_channels=unet.config.block_out_channels[0])
    face_encoder = FusionFaceId(
        cross_attention_dim=1024,
        id_embeddings_dim=512,
        clip_embeddings_dim=1024,
        num_tokens=4,)
    face_model = FaceModel()

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
                attn_procs[name] = AnimationAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank)
            else:
                layer_name = name.split(".processor")[0]
                weights = {
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
            if cross_attention_dim is None:
                attn_procs[name] = unet.attn_processors[name]
            else:
                attn_procs[name] = unet.attn_processors[name]
    unet.set_attn_processor(attn_procs)

    if args.finetune_mode is True and args.posenet_model_finetune_path is not None and args.face_encoder_finetune_path is not None and args.unet_model_finetune_path is not None:
        print("Loading existing posenet weights, face_encoder weights and unet weights.")
        if args.posenet_model_finetune_path.endswith(".pth"):
            pose_net_state_dict = torch.load(args.posenet_model_finetune_path, map_location="cpu")
            pose_net.load_state_dict(pose_net_state_dict, strict=True)
        else:
            print("posenet weights loading fail")
            print(1/0)
        if args.face_encoder_finetune_path.endswith(".pth"):
            face_encoder_state_dict = torch.load(args.face_encoder_finetune_path, map_location="cpu")
            face_encoder.load_state_dict(face_encoder_state_dict, strict=True)
        else:
            print("face_encoder weights loading fail")
            print(1/0)
        
        # ========================================================
        # 🚀 [核心手术：权重缝合] 只加载时间灵魂，保留 SVD 空间皮囊
        # ========================================================
        if args.unet_model_finetune_path.endswith(".pth"):
            # 1. 加载 StableAnimator (或者你之前练废了的那个) 的全量字典
            animator_state_dict = torch.load(args.unet_model_finetune_path, map_location="cpu")
            
            # 2. 关键！只筛选出带有 "temporal" (时间层) 或 "ipi_adapter" 的参数
            # 过滤掉所有空间层参数，保护 SVD 原生的雕像画功！
            temporal_state_dict = {
                k: v for k, v in animator_state_dict.items() 
                if "temporal" in k or "ipi_adapter" in k
            }
            
            # 3. 将这些“运动基因”注入到当前纯净的 SVD UNet 中
            missing, unexpected = unet.load_state_dict(temporal_state_dict, strict=False)
            print(f"✅ 缝合大法成功！仅移植了 {len(temporal_state_dict)} 个运动层参数，保留了原始 SVD 空间画功。")
        else:
            print("unet weights loading fail")
            print(1/0)
            
        # 🚀 独立加载 IPI Adapter 权重
        if hasattr(args, 'ipi_model_finetune_path') and args.ipi_model_finetune_path is not None:
            if args.ipi_model_finetune_path.endswith(".pth"):
                ipi_state_dict = torch.load(args.ipi_model_finetune_path, map_location="cpu")
                # 严格对齐加载进 unet.ipi_adapter
                unet.ipi_adapter.load_state_dict(ipi_state_dict, strict=True)
                print("🎯 独立版 IPI Adapter 权重加载成功！无缝合体！")
            else:
                print("⚠️ IPI Adapter 权重路径错误！")
        else:
            print("⚠️ 未提供单独的 IPI Adapter 权重！将使用零初始化从头学习。")
        # ========================================================


    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    pose_net.requires_grad_(False)
    face_encoder.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    image_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    if args.use_ema:
        ema_unet = EMAModel(unet.parameters(
        ), model_cls=UNetSpatioTemporalConditionModel, model_config=unet.config)

    if args.enable_xformers_memory_efficient_attention: 
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly")


    if args.gradient_checkpointing: 
        unet.enable_gradient_checkpointing()
        

    if args.allow_tf32: 
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr: 
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps *
            args.per_gpu_batch_size * accelerator.num_processes
        )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    # ===[优化版：混合训练锁定逻辑与双轨制学习率] ===
    unet.requires_grad_(False)
    pose_net.requires_grad_(False)
    face_encoder.requires_grad_(False)
    
    # 1. 明确定义两套学习率
    base_lr = args.learning_rate               # 启动脚本传进来的 5e-6
    adapter_lr = args.learning_rate * 20.0      # IPI 降温：从 10倍降到 2倍 (1e-5)
    
    # 2. 准备参数列表，采用 train.py 的逐参数 append 方式
    parameters_list = []
    
    # PoseNet 全量训练
    for name, para in pose_net.named_parameters():
        para.requires_grad = False
        parameters_list.append({"params": para, "lr": base_lr})
        
    # Face Encoder 全量训练
    for name, para in face_encoder.named_parameters():
        para.requires_grad = False
        parameters_list.append({"params": para, "lr": base_lr})

    # 3. 遍历 UNet 拆分参数，严格对齐 train.py 的 "attentions" 判断
    unet_attn_cnt = 0
    ipi_cnt = 0
    for name, para in unet.named_parameters():
        is_ipi = "ipi_adapter" in name
        is_attn = "attentions" in name  # 完全对齐 train.py，不扩大到 attn2 和 cross_attention

        if is_ipi:
            para.requires_grad = True
            parameters_list.append({"params": para, "lr": adapter_lr})
            ipi_cnt += 1
        elif is_attn:
            para.requires_grad = True
            parameters_list.append({"params": para, "lr": base_lr})
            unet_attn_cnt += 1
        else:
            para.requires_grad = False

    # 🎯[神级确认：让代码打印出来给你看！]
    if accelerator.is_main_process:
        print("\n" + "="*60)
        print("🎯 训练参数分配确认 (对齐 train.py 策略)")
        
        pose_total = sum(p.numel() for p in pose_net.parameters() if p.requires_grad) / 1e6
        face_total = sum(p.numel() for p in face_encoder.parameters() if p.requires_grad) / 1e6
        unet_attn_total = sum(p.numel() for n, p in unet.named_parameters() if p.requires_grad and "ipi_adapter" not in n) / 1e6
        ipi_total = sum(p.numel() for n, p in unet.named_parameters() if "ipi_adapter" in n) / 1e6
        
        print(f"-> [1. PoseNet]      训练参数: {pose_total:.2f} M | 学习率: {base_lr}")
        print(f"-> [2. FaceEncoder]  训练参数: {face_total:.2f} M | 学习率: {base_lr}")
        print(f"-> [3. UNet attentions] 训练参数: {unet_attn_total:.2f} M | 学习率: {base_lr} ({unet_attn_cnt} tensors)")
        print(f"-> [4. IPI Adapter]  训练参数: {ipi_total:.2f} M | 学习率: {adapter_lr} ({ipi_cnt} tensors)")
        print("="*60 + "\n")
        
    # 打印最终统计
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in unet.parameters() if not p.requires_grad)
    
    logger.info("="*50)
    logger.info(f"🚀 [时空解耦模式] 已激活！")
    logger.info(f"✅ UNet 训练参数 (attentions/IPI): {trainable_params / 1e6:.2f} M")
    logger.info(f"🔒 UNet 锁定参数 (其它层): {frozen_params / 1e6:.2f} M")
    logger.info("="*50)
    # =========================================================

    optimizer = optimizer_cls(
        parameters_list,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    if accelerator.is_main_process and args.log_trainable_parameters:   
        rec_txt1 = open('rec_para.txt', 'w')  
        rec_txt2 = open('rec_para_train.txt', 'w')  
    
        for name, para in unet.named_parameters():
            if para.requires_grad is False:
                rec_txt1.write(f'{name}\n')
            else:
                rec_txt2.write(f'{name}\n')
        
        rec_txt1.close()
        rec_txt2.close()

    args.global_batch_size = args.per_gpu_batch_size * accelerator.num_processes

    root_path = args.data_root_path
    txt_path_1 = args.rec_data_path
    txt_path_2 = args.vec_data_path
    txt_path_3 = args.clay_data_path
    train_dataset_1 = LargeScaleAnimationVideos(
        root_path=root_path,
        txt_path=txt_path_1,
        width=512,
        height=512,
        n_sample_frames=args.sample_n_frames,
        sample_frame_rate=4,
        app=face_model.app,
        handler_ante=face_model.handler_ante,
        face_helper=face_model.face_helper
    )
    train_dataloader_1 = torch.utils.data.DataLoader(
        train_dataset_1,
        batch_size=args.per_gpu_batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
    )
    train_dataset_2 = LargeScaleAnimationVideos(
        root_path=root_path,
        txt_path=txt_path_2,
        width=512,
        height=512,
        n_sample_frames=args.sample_n_frames,
        sample_frame_rate=4,
        app=face_model.app,
        handler_ante=face_model.handler_ante,
        face_helper=face_model.face_helper
    )
    train_dataloader_2 = torch.utils.data.DataLoader(
        train_dataset_2,
        batch_size=args.per_gpu_batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
    )
    train_dataset_3 = LargeScaleAnimationVideos(
        root_path=root_path,
        txt_path=txt_path_3,
        width=512,      # 泥塑用 512x512
        height=512,
        n_sample_frames=args.sample_n_frames,
        sample_frame_rate=4,
        app=face_model.app,
        handler_ante=face_model.handler_ante,
        face_helper=face_model.face_helper,
        
    )
    statue_repeat_factor = 5
    train_dataset_3 = ConcatDataset([train_dataset_3] * statue_repeat_factor)
    train_dataloader_3 = torch.utils.data.DataLoader(
        train_dataset_3,
        batch_size=args.per_gpu_batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
    )
    overrode_max_train_steps = False
    # 注意：下面计算长度时不需要再乘以 statue_repeat_factor 了，因为 dataloader 已经变长了
    total_dataloader_len = len(train_dataloader_1) + len(train_dataloader_2) + len(train_dataloader_3)
    num_update_steps_per_epoch = math.ceil(total_dataloader_len / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    unet, pose_net, face_encoder, optimizer, lr_scheduler, train_dataloader_1, train_dataloader_2 , train_dataloader_3 = accelerator.prepare(
        unet, pose_net, face_encoder, optimizer, lr_scheduler, train_dataloader_1, train_dataloader_2 , train_dataloader_3
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # 🚀 修复：计算总步数时，必须把雕像数据的放大倍数(3)算进去，否则学习率调度器会错乱！

    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("StableAnimator", config=vars(args))

    # Train!
    total_batch_size = args.per_gpu_batch_size * \
        accelerator.num_processes * args.gradient_accumulation_steps

    len_zeros = len(train_dataloader_1)
    len_ones = len(train_dataloader_2)
    len_twos = len(train_dataloader_3)  # 新增

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset_1)+len(train_dataset_2)+ len(train_dataset_3)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_gpu_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    # === [IPI 诊断] 初始化分数据集 Loss 记录 ===
    statue_loss_accum = 0.0
    statue_loss_count = 0
    dance_loss_accum = 0.0
    dance_loss_count = 0
    def encode_image(pixel_values):  
        pixel_values = _resize_with_antialiasing(pixel_values, (224, 224))  
        pixel_values = (pixel_values + 1.0) / 2.0   

        pixel_values = pixel_values.to(torch.float32) 
        pixel_values = feature_extractor(
            images=pixel_values,
            do_normalize=True,
            do_center_crop=False,  
            do_resize=False,  
            do_rescale=False,  
            return_tensors="pt", 
        ).pixel_values

        pixel_values = pixel_values.to(  
            device=accelerator.device, dtype=image_encoder.dtype)
        image_embeddings = image_encoder(pixel_values).image_embeds  
        image_embeddings= image_embeddings.unsqueeze(1)  
        return image_embeddings

    def _get_add_time_ids(
        fps, 
        motion_bucket_id, 
        noise_aug_strength, 
        dtype,
        batch_size,
        unet=None, 
        device=None
    ):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength] 
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype, device=device) 
        add_time_ids = add_time_ids.repeat(batch_size, 1) 
        return add_time_ids

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")] 
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1])) 
            path = dirs[-1] if len(dirs) > 0 else None 

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None 
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            
            # ====================================================================
            # 🚀 强行夺回学习率控制权：无视旧 checkpoint 记忆，强制使用新 LR
            # ====================================================================
            if accelerator.is_main_process:
                print(f"⚠️ [强制覆盖] 正在粉碎旧状态，强行将 LR 设为: {args.learning_rate}")
            
            # 1. 强行修改 Optimizer（优化器）里记录的学习率
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.learning_rate
                if 'initial_lr' in param_group:
                    param_group['initial_lr'] = args.learning_rate

            # 2. 强行修改 Scheduler（调度器）的基准学习率，防止它下一步又变回去
            if hasattr(lr_scheduler, 'base_lrs'):
                lr_scheduler.base_lrs = [args.learning_rate for _ in lr_scheduler.base_lrs]
            # ====================================================================

            global_step = int(path.split("-")[1]) 
            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps)

    progress_bar = tqdm(range(global_step, args.max_train_steps),
                        disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        pose_net.train() 
        face_encoder.eval()
        unet.train()
        train_loss = 0.0
        iter1 = iter(train_dataloader_1)
        iter2 = iter(train_dataloader_2)
        iter3 = iter(train_dataloader_3)  
        # 🚀 强行提高雕像采样权重 (5倍)，防止被真人数据淹没
        statue_repeat = 5
        list_0_1_2 = [0] * len_zeros + [1] * len_ones + [2] * len_twos
        # 必须使用固定种子的 Generator 洗牌，保证 3 张显卡拿到的顺序一模一样，杜绝死锁！
        g = torch.Generator()
        g.manual_seed(args.seed + epoch if args.seed else 42 + epoch)
        indices = torch.randperm(len(list_0_1_2), generator=g).tolist()
        list_0_1_2 = [list_0_1_2[i] for i in indices]
        
        for step in range(0, len(list_0_1_2)):
            current_idx = list_0_1_2[step]
            if current_idx == 0:
                try:
                    batch = next(iter1)
                except StopIteration:
                    iter1 = iter(train_dataloader_1)
                    batch = next(iter1)
            elif current_idx == 1:
                try:
                    batch = next(iter2)
                except StopIteration:
                    iter2 = iter(train_dataloader_2)
                    batch = next(iter2)
            elif current_idx == 2:
                try:
                    batch = next(iter3)
                except StopIteration:
                    iter3 = iter(train_dataloader_3)
                    batch = next(iter3)

            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(pose_net, face_encoder, unet):
                with accelerator.autocast():
                    pixel_values = batch["pixel_values"].to(weight_dtype).to(
                        accelerator.device, non_blocking=True
                    )
                    conditional_pixel_values = batch["reference_image"].to(weight_dtype).to(
                        accelerator.device, non_blocking=True
                    )
                    latents = tensor_to_vae_latent(pixel_values, vae).to(dtype=weight_dtype)

                    encoder_hidden_states = encode_image(conditional_pixel_values).to(dtype=weight_dtype)
                    image_embed = encoder_hidden_states.clone()
                    
                    # 1. 提取 DWPose 原始数据
                    dwpose_raw = batch["dwpose_raw"].to(dtype=weight_dtype, device=accelerator.device)
                    
                    # 🚀 【Bug 已修复】移除了导致坐标从 [-1, 1] 被再次映射到 [-3, 1] 的双重归一化错误
                    # 🚨 暴力排查修改：强制打印真实送入网络的坐标极值！
                    if accelerator.is_main_process and (global_step % 50 == 0):
                        test_valid = dwpose_raw[dwpose_raw > -0.5]
                        if len(test_valid) > 0:
                            logger.info(f"🚨 [监控] PoseNet 接收坐标范围: Min={test_valid.min().item():.2f}, Max={test_valid.max().item():.2f}")
                    # 2. 诊断：计算这一组数据的原始运动强度
                    with torch.no_grad():
                        pose_diff_raw = (dwpose_raw[:, 1:] - dwpose_raw[:, :-1]).abs().mean().item()

                    # 3. 【核心补丁】低动力补偿（只放大相对运动，不放大绝对坐标）

                    
                    # 4. 提取 CLIP Image Embedding (驱动序列特征)
                    driving_clip_pixels = batch["driving_clip_pixel_values"].to(
                        dtype=weight_dtype, device=accelerator.device, non_blocking=True
                    )
                    b_size, n_frames, c_c, h_c, w_c = driving_clip_pixels.shape
                    driving_clip_pixels = driving_clip_pixels.reshape(b_size * n_frames, c_c, h_c, w_c)
                    
                    # 🚀 【核心抢救 1：物理致盲 IPI，彻底切断身份特征泄漏！】
                    # 只有致盲，IPI 才只剩下“动作影子”，不再往背景里塞原来的长相

                    if current_idx == 2: # 只对雕像数据进行致盲
                        # 强转灰度 + 51像素极度模糊
                        gray_transform = T.Grayscale(num_output_channels=3)
                        driving_clip_pixels = gray_transform(driving_clip_pixels)
                        blur_transform = T.GaussianBlur(kernel_size=(3, 3), sigma=(1.0, 1.0))
                        driving_clip_pixels = blur_transform(gray_transform(driving_clip_pixels))

                    with torch.no_grad():
                        clip_image_embeds = image_encoder(driving_clip_pixels).image_embeds
                        clip_image_embeds = clip_image_embeds.reshape(b_size, n_frames, -1)
                        
                    # 删掉所有关于 motion_score_normalized 的 if/else 区分，统一按真实运动强度分配 Bucket！
                    if accelerator.is_main_process and (global_step % 50 == 0):
                        print(f"\n" + "!"*40)
                        print(f"🔥 Step: {global_step} | 数据源索引: {current_idx}")
                        print(f"📊 原始运动强度: {pose_diff_raw:.8f}")

                        print("!"*40 + "\n")

                    # 🚀 【修复2：打破SVD的惰性】条件图像噪声增强提高到0.04
                    train_noise_aug = 0.04 
                    conditional_pixel_values = conditional_pixel_values + train_noise_aug * \
                        randn_tensor(conditional_pixel_values.shape, generator=generator, device=conditional_pixel_values.device, dtype=conditional_pixel_values.dtype)
                    conditional_latents = tensor_to_vae_latent(conditional_pixel_values, vae, scale=False)

                    noise = torch.randn_like(latents) 
                    bsz = latents.shape[0] 
                    sigmas = rand_cosine_interpolated(
                        shape=[bsz,], 
                        image_d=image_d, 
                        noise_d_low=noise_d_low,
                        noise_d_high=noise_d_high, 
                        sigma_data=sigma_data, 
                        min_value=min_value, 
                        max_value=max_value
                    ).to(latents.device, dtype=weight_dtype)

                    sigmas_reshaped = sigmas.clone() 
                    while len(sigmas_reshaped.shape) < len(latents.shape): 
                        sigmas_reshaped = sigmas_reshaped.unsqueeze(-1)

                    noisy_latents  = latents + noise * sigmas_reshaped   
                    timesteps = torch.Tensor([0.25 * sigma.log() for sigma in sigmas]).to(latents.device, dtype=weight_dtype)

                    inp_noisy_latents = noisy_latents  / ((sigmas_reshaped**2 + 1) ** 0.5)
                    
                    low_bound = 0.005
                    high_bound = 0.050
                    
                    # 🚀 魏总专属修复 2：统一运动评价体系！无论真人还是雕像，都按真实幅度计算！
                    motion_score = max(low_bound, min(high_bound, pose_diff_raw))
                    motion_score_normalized = (motion_score - low_bound) / (high_bound - low_bound + 1e-6)
                    dynamic_motion_bucket = 80.0 + motion_score_normalized * (160.0 - 80.0)

                    if accelerator.is_main_process and (global_step % 50 == 0):
                        logger.info(f"🌊 真实运动强度({pose_diff_raw:.5f}) -> 分配 Motion Bucket: {dynamic_motion_bucket:.1f}")

                    added_time_ids = _get_add_time_ids(
                        fps=14,
                        #motion_bucket_id=dynamic_motion_bucket, 
                        motion_bucket_id=127.0,
                        noise_aug_strength=train_noise_aug, 
                        dtype=encoder_hidden_states.dtype,
                        batch_size=bsz,
                        unet=unet,
                        device=latents.device
                    )
                    added_time_ids = added_time_ids.to(latents.device)

                    if args.conditioning_dropout_prob is not None:
                        random_p = torch.rand(
                            bsz, device=latents.device, generator=generator)
                        prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                        prompt_mask = prompt_mask.reshape(bsz, 1, 1)
                        null_conditioning = torch.zeros_like(encoder_hidden_states)
                        encoder_hidden_states = torch.where(
                            prompt_mask, null_conditioning, encoder_hidden_states)

                        image_mask_dtype = conditional_latents.dtype
                        image_mask = 1 - (
                            (random_p >= args.conditioning_dropout_prob).to(
                                image_mask_dtype)
                            * (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype)
                        )
                        image_mask = image_mask.reshape(bsz, 1, 1, 1)
                        conditional_latents = image_mask * conditional_latents

                    conditional_latents = conditional_latents.unsqueeze(
                        1).repeat(1, noisy_latents.shape[1], 1, 1, 1)
                    pose_pixels = batch["pose_pixels"].to(
                        dtype=weight_dtype, device=accelerator.device, non_blocking=True
                    )
                    faceid_embeds = batch["faceid_embeds"].to(
                        dtype=weight_dtype, device=accelerator.device, non_blocking=True
                    )
                    
                    pose_latents = pose_net(pose_pixels)

                    # 🚀 [终极补丁 1：彻底切断人脸通道 + 打破特征捷径] 
                    is_dropped_human = False
                    if current_idx == 2:
                        # 雕像：100% 强制全 0
                        faceid_latents = torch.zeros(
                            image_embed.shape[0], 4, image_embed.shape[-1],
                            device=image_embed.device, dtype=image_embed.dtype
                        )
                    else:
                        # 💃 真人：正常提取人脸特征，但有 20% 的概率被强制掐断为全 0！
                        # 目的：逼迫模型在收到全 0 信号时，依然必须乖乖听从 PoseNet 的跳舞动作！
                        pseudo_rand = (global_step * 73 + args.seed) % 100

                        if pseudo_rand < 15:
                            faceid_latents = torch.zeros(
                                image_embed.shape[0], 4, image_embed.shape[-1],
                                device=image_embed.device, dtype=image_embed.dtype
                            )
                            is_dropped_human = True
                            if accelerator.is_main_process and (global_step % 50 == 0):
                                logger.info(f"🎲 真人数据触发随机 Drop：强制使用全 0 FaceID，逼迫模型听从骨架！")
                        else:
                            faceid_latents = face_encoder(faceid_embeds, image_embed)

                    inp_noisy_latents = torch.cat(
                        [inp_noisy_latents, conditional_latents], dim=2)
                    target = latents 

                    # 🚀 雕像虽然是全零特征，但也必须拼进去，防止 UNet 靠 Token 数量作弊！
                    encoder_hidden_states = torch.cat([encoder_hidden_states, faceid_latents], dim=1)
                    
                    encoder_hidden_states = encoder_hidden_states.to(latents.dtype) 
                    inp_noisy_latents = inp_noisy_latents.to(latents.dtype)
                    pose_latents = pose_latents.to(latents.dtype)


                    # 🚀 魏总专属修复 3：[终极补丁] 运动特征的 Dropout，让模型学会 CFG！
                    # 🚀 魏总专属修复 2：拯救 CFG 的灵魂 Dropout！
                    if args.conditioning_dropout_prob is not None:
                        # 🚀 极度关键修复：禁止使用 torch.rand，防止 validation 导致主卡种子错位，引发分布式死锁！
                        drop_mask_1d = torch.zeros(bsz, dtype=torch.bool, device=latents.device)
                        for b_idx in range(bsz):
                            pseudo_rand_cfg = ((global_step * bsz + b_idx) * 37 + args.seed) % 100
                            if pseudo_rand_cfg < int(args.conditioning_dropout_prob * 100):
                                drop_mask_1d[b_idx] = True
                        
                        # 掐断 pose_latents
                        drop_mask_pose = drop_mask_1d.view(bsz, *[1]*(pose_latents.dim()-1))
                        pose_latents = torch.where(drop_mask_pose, torch.zeros_like(pose_latents), pose_latents)
                        
                        # 🚀 [魏总专属修复] 规范化 Mask 广播：确保 dwpose_raw 和 clip_image_embeds 安全掉落
                        drop_mask_seq = drop_mask_1d.view(bsz, *[1]*(dwpose_raw.dim()-1))
                        dwpose_raw = torch.where(drop_mask_seq, torch.zeros_like(dwpose_raw), dwpose_raw)
                        
                        drop_mask_clip = drop_mask_1d.view(bsz, *[1]*(clip_image_embeds.dim()-1))
                        clip_image_embeds = torch.where(drop_mask_clip, torch.zeros_like(clip_image_embeds), clip_image_embeds)

                    # ----- 下面是您原有的代码 -----
                    model_pred = unet(
                        inp_noisy_latents, timesteps, encoder_hidden_states,
                        added_time_ids=added_time_ids,
                        pose_latents=pose_latents,
                        dwpose_raw=dwpose_raw,           
                        clip_image_embeds=clip_image_embeds,  
                        ).sample
                    
                    
                    sigmas = sigmas_reshaped
                    c_out = -sigmas / ((sigmas**2 + 1)**0.5)
                    c_skip = 1 / (sigmas**2 + 1)
                    denoised_latents = model_pred * c_out + c_skip * noisy_latents
                    weighing = (1 + sigmas ** 2) * (sigmas**-2.0)
                    
                    tgt_face_masks = batch["tgt_face_masks"].to(
                        dtype=weight_dtype, device=accelerator.device, non_blocking=True
                    )
                    tgt_face_masks = rearrange(tgt_face_masks, "b f c h w -> (b f) c h w")
                    tgt_face_masks = F.interpolate(tgt_face_masks, size=(target.size()[-2], target.size()[-1]), mode='nearest')
                    tgt_face_masks = rearrange(tgt_face_masks, "(b f) c h w -> b f c h w", f=args.sample_n_frames)

                    # 🚀 [修复后的骨架前景 Mask 提取逻辑]
                    if current_idx == 2 or is_dropped_human:
                        face_weight = 1.0+ tgt_face_masks * 2.0 # 雕像取消面部强约束
                        
                        # 修复：不再使用 mean，而是使用 max，只要有任何一个通道(RGB)有颜色，就是前景
                        # 背景 (-1, -1, -1) 的 max 是 -1.0；红色 (1, -1, -1) 的 max 是 1.0
                        pose_max, _ = pose_pixels.max(dim=2, keepdim=True)  # 取通道最大值
                        
                        # 只要最大值大于 -0.9，说明不是纯黑背景
                        fg_mask = (pose_max > -0.9).float() 
                        fg_mask = rearrange(fg_mask, "b f c h w -> (b f) c h w")
                        fg_mask = F.interpolate(fg_mask, size=(target.size()[-2], target.size()[-1]), mode='nearest')
                        fg_mask = F.avg_pool2d(fg_mask, kernel_size=7, stride=1, padding=3)
                        fg_mask = rearrange(fg_mask, "(b f) c h w -> b f c h w", f=args.sample_n_frames)
                        fg_mask = (fg_mask > 0.05).float()
                        
                        # 🚀 [魏总专属修复] 平衡前景背景，允许雕像产生形变/运动
                        region_weight = fg_mask * 1.5 + (1 - fg_mask) * 0.5
                    else:
                        face_weight = 1 + tgt_face_masks
                        region_weight = 1.0

                    # 基础 Diffusion MSE Loss
                    loss = torch.mean(
                        (weighing.float() * (denoised_latents.float() - target.float()) ** 2 * face_weight * region_weight
                        ).reshape(target.shape[0], -1),
                        dim=1,
                    )
                    loss = loss.mean()


                    # ====================================================================
                    # 🚀 最小侵入版 Pose Loss (特征级对齐，无需引入外部大模型)
                    # ====================================================================
                    # 假设:
                    # model_pred 是你的预测噪声 (B, C, F, H, W)
                    # target 是真实噪声 (B, C, F, H, W)
                    # ipi_features 是你的 IPI Adapter 输出的姿态特征 (通常是 B*F, Tokens, D_model)



                    
                    # 🚀 材质一致性约束：必须对齐没有加过噪的干净 latents
                    if current_idx == 2:
                        ref_latent = latents[:, 0:1].detach() # 绝对干净的真实第一帧 latent
                        pred_first = denoised_latents[:, 0:1]
                        statue_consistency_loss = F.l1_loss(pred_first, ref_latent)
                        loss = loss + 0 * statue_consistency_loss
                    
                    try:
                        if current_idx == 2:
                            statue_loss_accum += loss.item()
                            statue_loss_count += 1
                        else:
                            dance_loss_accum += loss.item()
                            dance_loss_count += 1
                    except Exception:
                        pass
                    
                    avg_loss = accelerator.gather(
                        loss.repeat(args.per_gpu_batch_size)).mean()
                    train_loss += avg_loss.item() / args.gradient_accumulation_steps

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                    optimizer.step()  
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    # with torch.cuda.device(latents.device):
                    #     torch.cuda.empty_cache()

            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
                
                if accelerator.is_main_process and (global_step % 100 == 0):
                    s_avg = statue_loss_accum / max(1, statue_loss_count)
                    d_avg = dance_loss_accum / max(1, dance_loss_count)
                    print(f"\n[STATS] Step {global_step} | 雕像 Loss: {s_avg:.6f} | 跳舞 Loss: {d_avg:.6f}")
                    statue_loss_accum, statue_loss_count = 0.0, 0
                    dance_loss_accum, dance_loss_count = 0.0, 0
                
                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    
                    if args.checkpoints_total_limit is not None and accelerator.is_main_process:
                        checkpoints = os.listdir(args.output_dir)  
                        checkpoints = [    
                            d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(   
                            checkpoints, key=lambda x: int(x.split("-")[1]))

                        if len(checkpoints) >= args.checkpoints_total_limit:   
                            num_to_remove = len(
                                checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(
                                f"removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(
                                    args.output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(
                        args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    
                    unwrap_unet = accelerator.unwrap_model(unet)
                    unwrap_pose_net = accelerator.unwrap_model(pose_net)
                    unwrap_face_encoder = accelerator.unwrap_model(face_encoder)
                    
                    # 🚀 修复：中间存盘的权重必须带上 global_step，不能叫 last
                    unwrap_unet = accelerator.unwrap_model(unet)
                    
                    # 🚀 [IPI 抽离魔法 1]：单独提取 IPI Adapter 的权重并保存
                    ipi_state_dict = unwrap_unet.ipi_adapter.state_dict()
                    torch.save(ipi_state_dict, os.path.join(save_path, f"ipi_adapter-{global_step}.pth"))
                    
                    # 🚀[IPI 抽离魔法 2]：从 UNet 字典中剔除 ipi_adapter 的参数，保存纯净版 UNet
                    full_unet_dict = unwrap_unet.state_dict()
                    clean_unet_dict = {k: v for k, v in full_unet_dict.items() if not k.startswith("ipi_adapter.")}
                    torch.save(clean_unet_dict, os.path.join(save_path, f"unet-{global_step}.pth"))
                    
                    unwrap_pose_net_state_dict = unwrap_pose_net.state_dict()
                    torch.save(unwrap_pose_net_state_dict, os.path.join(save_path, f"pose_net-{global_step}.pth"))
                    
                    unwrap_face_encoder_state_dict = unwrap_face_encoder.state_dict()
                    torch.save(unwrap_face_encoder_state_dict, os.path.join(save_path, f"face_encoder-{global_step}.pth"))
                    torch.save(unwrap_face_encoder_state_dict, os.path.join(save_path, "face_encoder-last.pth"))
                    logger.info(f"Saved state to {save_path}")   
            
                if accelerator.is_main_process:
                    if global_step % args.validation_steps == 0:
                        logger.info(
                            f"Running validation... \n Generating {args.num_validation_images} videos."
                        )
                        if args.use_ema:
                            ema_unet.store(unet.parameters())
                            ema_unet.copy_to(unet.parameters())

                        log_validation(
                            vae=vae,
                            image_encoder=image_encoder,
                            unet=unet,
                            pose_net=pose_net,
                            face_encoder=face_encoder,
                            app=face_model.app,
                            face_helper=face_model.face_helper,
                            handler_ante=face_model.handler_ante,
                            scheduler=noise_scheduler,
                            accelerator=accelerator,
                            feature_extractor=feature_extractor,
                            width=512,
                            height=512,
                            torch_dtype=weight_dtype,
                            validation_image_folder=args.validation_image_folder,
                            validation_image=args.validation_image,
                            validation_control_folder=args.validation_control_folder,
                            output_dir=args.output_dir,
                            generator=generator,
                            global_step=global_step,
                            num_validation_cases=1,
                            num_frames=args.num_frames,
                            pose_data_path=args.pose_data_path,
                            is_statue=args.is_statue,
                        )

                        if args.use_ema:
                            ema_unet.restore(unet.parameters())

                        # with torch.cuda.device(latents.device):
                        #     torch.cuda.empty_cache()

            # 🚀 [终极修复 3：全员集结]
            # 必须写在 if accelerator.is_main_process 外面！跟 if 对齐。
            # 让 Rank 1 和 Rank 2 停在这里，等 Rank 0 验证完了再一起往下走！
            accelerator.wait_for_everyone()

            current_lrs = lr_scheduler.get_last_lr()
            logs = {
                "loss": f"{loss.detach().item():.4f}", 
                "lr_base": f"{current_lrs[0]:.2e}",  # Group 0 (UNet)
                "lr_ipi": f"{current_lrs[1]:.2e}"    # Group 1 (IPI)
            }
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break
            
    if accelerator.is_main_process:
        save_path = os.path.join(
            args.output_dir, f"checkpoint-last")
        os.makedirs(save_path, exist_ok=True)
        accelerator.save_state(save_path)
        unwrap_unet = accelerator.unwrap_model(unet)
        unwrap_pose_net = accelerator.unwrap_model(pose_net)
        unwrap_face_encoder = accelerator.unwrap_model(face_encoder)
        # 🚀 修复：最终训练结束时，必须统一写入 checkpoint-last 文件夹中！
        unwrap_unet = accelerator.unwrap_model(unet)
        
        # 🚀[IPI 抽离魔法 1]：单独提取 IPI Adapter 的权重并保存
        ipi_state_dict = unwrap_unet.ipi_adapter.state_dict()
        torch.save(ipi_state_dict, os.path.join(save_path, "ipi_adapter-last.pth"))
        
        # 🚀[IPI 抽离魔法 2]：从 UNet 字典中剔除 ipi_adapter 的参数，保存纯净版 UNet
        full_unet_dict = unwrap_unet.state_dict()
        clean_unet_dict = {k: v for k, v in full_unet_dict.items() if not k.startswith("ipi_adapter.")}
        torch.save(clean_unet_dict, os.path.join(save_path, "unet-last.pth"))
        
        unwrap_pose_net_state_dict = unwrap_pose_net.state_dict()
        torch.save(unwrap_pose_net_state_dict, os.path.join(save_path, "pose_net-last.pth"))
        
        unwrap_face_encoder_state_dict = unwrap_face_encoder.state_dict()
        torch.save(unwrap_face_encoder_state_dict, os.path.join(save_path, "face_encoder-last.pth"))
        logger.info(f"Saved state to {save_path}")   



def normalize_pose(pose_data, width, height):
    """
    智能归一化：自动识别绝对像素坐标和 [0,1] 坐标，统一映射到 [-1, 1]
    与 inference logic 保持 100% 一致，且能正确过滤无效点 (-1)。
    """
    pose_data = pose_data.clone() if isinstance(pose_data, torch.Tensor) else pose_data.copy()
    
    # 找到有效点 (非-1)
    valid_mask = (pose_data[..., 0] != -1) & (pose_data[..., 1] != -1)
    
    valid_poses = pose_data[valid_mask]
    if len(valid_poses) > 0:
        if valid_poses.max() > 1.5:
            pose_data[valid_mask, 0] = (pose_data[valid_mask, 0] / width) * 2.0 - 1.0
            pose_data[valid_mask, 1] = (pose_data[valid_mask, 1] / height) * 2.0 - 1.0
        else:
            pose_data[valid_mask, 0] = pose_data[valid_mask, 0] * 2.0 - 1.0
            pose_data[valid_mask, 1] = pose_data[valid_mask, 1] * 2.0 - 1.0
            
    return pose_data

def log_validation(
        vae,
        image_encoder,
        unet,
        pose_net,
        face_encoder,
        app,
        face_helper,
        handler_ante,
        scheduler,
        accelerator,
        feature_extractor,
        width,
        height,
        torch_dtype,
        validation_image_folder,
        validation_image,
        validation_control_folder,
        output_dir,
        generator,
        global_step,
        num_frames,
        pose_data_path=None,  # 🚀 接收骨架文件
        num_validation_cases=1,
        is_statue=False,      # 🚀 新增雕像标记控制
):
    logger.info("Running validation... ")
    validation_unet = accelerator.unwrap_model(unet)
    validation_image_encoder = accelerator.unwrap_model(image_encoder)
    validation_vae = accelerator.unwrap_model(vae)
    validation_pose_net = accelerator.unwrap_model(pose_net)
    validation_face_encoder = accelerator.unwrap_model(face_encoder)
    
    # 🚀 核心修改 1：换用完全体 Inference Pipeline
    pipeline = InferenceAnimationPipeline(
        vae=validation_vae,
        image_encoder=validation_image_encoder,
        unet=validation_unet,
        scheduler=scheduler,
        feature_extractor=feature_extractor,
        pose_net=validation_pose_net,
        face_encoder=validation_face_encoder,
    )
    pipeline = pipeline.to(accelerator.device)  
    
    if validation_image_folder is not None and os.path.exists(validation_image_folder):
        validation_images = load_images_from_folder(validation_image_folder)
    else:
        validation_images = []

    validation_image_path = validation_image
    
    # 强制缩放图片
    if validation_image is None:
        validation_image = validation_images[0].resize((width, height))
    else:
        validation_image = Image.open(validation_image).convert('RGB').resize((width, height))
        
    # 🚀 核心对齐：明确数据来源
    # 1. 从 validation_image_folder 读取【驱动真人视频原图】(用于 CLIP 提取动作意图)
    driving_rgb_frames = load_images_from_folder(validation_image_folder)
    driving_rgb_frames = [img.resize((width, height)) for img in driving_rgb_frames]
        
    # 2. 从 validation_control_folder 读取【标准骨架图序列】(用于给模型打骨架)
    validation_control_images = load_images_from_folder(validation_control_folder)
    validation_control_images = [img.resize((width, height)) for img in validation_control_images]

    # -------------------------------------------------------
    # 🚀 核心修改 2：读取 npy 骨架数据并准备 CLIP 特征
    # -------------------------------------------------------
    
    num_frames_val = len(validation_control_images)
    # 1. 准备 dwpose raw 数据
    # === 替换 train_ipi_diaosunoshijian.py 约 731 行的代码 ===
    if pose_data_path and os.path.exists(pose_data_path):
        full_pose_data = np.load(pose_data_path)
        dwpose_raw_np = full_pose_data[:num_frames_val]
        
        # 换成带免疫无效点的智能映射
        valid_mask = dwpose_raw_np >= 0.0
        if valid_mask.any() and dwpose_raw_np[valid_mask].max() <= 1.05:
            dwpose_raw_np[valid_mask] = dwpose_raw_np[valid_mask] * 2.0 - 1.0
            
    else:
        logger.info("⚠️ 验证缺省 npy 数据，使用全 0 骨架！")
        dwpose_raw_np = np.zeros((num_frames_val, 18, 2), dtype=np.float32)

    dwpose_raw_tensor = torch.from_numpy(dwpose_raw_np).float().view(1, num_frames_val, -1).to(device=accelerator.device, dtype=torch_dtype)

    # 2. 🚀 修复：提取验证驱动视频的动态 CLIP 特征
    clip_inputs = feature_extractor(images=driving_rgb_frames, return_tensors="pt")
    val_clip_pixels = clip_inputs.pixel_values.to(device=accelerator.device, dtype=torch_dtype)
    
    # 🚀 如果是雕像模式，才执行致盲操作（灰度+模糊）
    if is_statue:
        import torchvision.transforms as T
        gray_transform = T.Grayscale(num_output_channels=3)
        blur_transform = T.GaussianBlur(kernel_size=(15, 15), sigma=(5.0, 5.0))
        val_clip_pixels = blur_transform(gray_transform(val_clip_pixels))
    
    with torch.no_grad():
        # 得到 [num_frames, 1024] -> 增加 batch 维度 [1, num_frames, 1024]
        clip_image_embeds = validation_image_encoder(val_clip_pixels).image_embeds
        clip_image_embeds = clip_image_embeds.unsqueeze(0)

    val_save_dir = os.path.join(output_dir, "validation_images")
    if not os.path.exists(val_save_dir):
        os.makedirs(val_save_dir)

    with accelerator.autocast():
        for val_img_idx in range(num_validation_cases):
            face_helper.clean_all() 
            validation_face = cv2.imread(validation_image_path)
            
            # 人脸检测，做保底
            validation_image_face_info = app.get(validation_face)   
            if len(validation_image_face_info) > 0:
                validation_image_face_info = sorted(validation_image_face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[-1]
                validation_image_id_ante_embedding = validation_image_face_info['embedding']
            else:
                validation_image_id_ante_embedding = None

            if validation_image_id_ante_embedding is None:
                try:
                    face_helper.read_image(validation_face)
                    face_helper.get_face_landmarks_5(only_center_face=True)
                    face_helper.align_warp_face()
                    if len(face_helper.cropped_faces) == 0:
                        validation_image_id_ante_embedding = np.zeros((512,))
                    else:
                        validation_image_align_face = face_helper.cropped_faces[0]
                        validation_image_id_ante_embedding = handler_ante.get_feat(validation_image_align_face)
                except Exception as e:
                    validation_image_id_ante_embedding = np.zeros((512,))

            # 🚀 核心修改 3：如果是雕像，才物理切断 FaceEncoder
            # 🧨 极度危险：必须保存原始 forward，用完立刻还原！否则 Rank 0 永远丧失人脸提取能力，导致 101 步 DDP 死锁！
            original_forward = validation_face_encoder.forward
            if is_statue:
                logger.info("🗿 验证流：检测到雕像模式，正在物理切断 FaceEncoder...")
                def statue_forward(self, faceid_embeds, image_embed, *args, **kwargs):
                    return torch.zeros(image_embed.shape[0], 4, image_embed.shape[-1], 
                                       device=image_embed.device, dtype=image_embed.dtype)
                import types
                validation_face_encoder.forward = types.MethodType(statue_forward, validation_face_encoder)
            # 👆 修改结束 👆

            video_frames = pipeline(
                image=validation_image,
                image_pose=validation_control_images,
                height=height,
                width=width,
                num_frames=num_frames_val,
                tile_size=num_frames_val,
                tile_overlap=4,
                decode_chunk_size=4,
                motion_bucket_id=200.0,
                fps=7,
                min_guidance_scale=3,
                max_guidance_scale=3,
                noise_aug_strength=0.01,
                num_inference_steps=30,
                generator=generator,
                output_type="pil",
                validation_image_id_ante_embedding=validation_image_id_ante_embedding,
                # 👇 下面是关键的动态参数：
                is_statue=is_statue,  
                dwpose_raw=dwpose_raw_tensor,
                clip_image_embeds=clip_image_embeds,
            ).frames[0]
            
            # 🧨 极度危险：验证结束，立刻还原 FaceEncoder 的 forward 方法，救活 Rank 0！
            validation_face_encoder.forward = original_forward
            out_file = os.path.join(
                val_save_dir,
                f"step_{global_step}_val_img_{val_img_idx}.mp4",
            )
            for i in range(num_frames_val):
                img = video_frames[i]
                video_frames[i] = np.array(img)
            export_to_gif(video_frames, out_file, 8)

    del pipeline
    # torch.cuda.empty_cache()



if __name__ == "__main__":
    main()