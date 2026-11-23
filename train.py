import argparse
import random
import logging
import math
import os

import cv2
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

from animation.dataset.animation_dataset import LargeScaleAnimationVideos  #数据集
from animation.modules.attention_processor import AnimationAttnProcessor   #注意力处理器
from animation.modules.attention_processor_normalized import AnimationIDAttnNormalizedProcessor  #归一化注意力处理器
from animation.modules.face_model import FaceModel  #面部模型
from animation.modules.id_encoder import FusionFaceId  #ID 编码器
from animation.modules.pose_net import PoseNet  #姿态网络
from animation.modules.unet import UNetSpatioTemporalConditionModel  #时空条件 UNet

from animation.pipelines.validation_pipeline_animation import ValidationAnimationPipeline  #验证流水线
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



# def load_images_from_folder(folder):
#     images = []
#     valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}  # Add or remove extensions as needed
#
#     # Function to extract frame number from the filename
#     def frame_number(filename):
#         # First, try the pattern 'frame_x_7fps'
#         new_pattern_match = re.search(r'frame_(\d+)_7fps', filename)
#         if new_pattern_match:
#             return int(new_pattern_match.group(1))
#         # If the new pattern is not found, use the original digit extraction method
#         matches = re.findall(r'\d+', filename)
#         if matches:
#             if matches[-1] == '0000' and len(matches) > 1:
#                 return int(matches[-2])  # Return the second-to-last sequence if the last is '0000'
#             return int(matches[-1])  # Otherwise, return the last sequence
#         return float('inf')  # Return 'inf'
#
#     # Sorting files based on frame number
#     sorted_files = sorted(os.listdir(folder), key=frame_number)
#
#     # Load images in sorted order
#     for filename in sorted_files:
#         ext = os.path.splitext(filename)[1].lower()
#         if ext in valid_extensions:
#             img = Image.open(os.path.join(folder, filename)).convert('RGB')
#             images.append(img)
#
#     return images

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
#用于从插值余弦时间步分布中生成样本，常用于扩散模型（Diffusion Models）中的噪声调度。其核心是通过余弦调度生成随时间变化的噪声水平，并在两个噪声维度（noise_d_low 和 noise_d_high）之间进行插值，以增强采样的灵活性。用于从插值余弦时间步分布中生成样本，常用于扩散模型（Diffusion Models）中的噪声调度。
#其核心是通过余弦调度生成随时间变化的噪声水平，并在两个噪声维度（noise_d_low 和 noise_d_high）之间进行插值，以增强采样的灵活性。
    """ 核心数学原理
    扩散模型中，噪声水平通常用 sigma（标准差）表示，而 sigma 与 logsnr（对数信噪比）存在转换关系：logsnr = -2 * log(sigma / sigma_data)sigma = sigma_data * exp(-logsnr / 2)
    该函数通过以下步骤生成 sigma：
    计算 logsnr 的最小 / 最大值（基于 min_value 和 max_value）。
    生成均匀分布的随机变量 u（用于控制插值比例）。
    基于余弦调度计算 logsnr，并在 noise_d_low 和 noise_d_high 之间插值。
    将 logsnr 转换为最终的 sigma。 """

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
    # 1. 构造均值为loc、标准差为scale的正态分布
    # 2. 计算该正态分布的ICDF在u处的值（得到正态分布样本）
    # 3. 取指数得到对数正态分布样本
    return torch.distributions.Normal(loc, scale).icdf(u).exp()

min_value = 0.002
max_value = 700
image_d = 64
noise_d_low = 32
noise_d_high = 64
sigma_data = 0.5

#用于对输入图像（张量）进行带抗锯齿（antialiasing）的缩放。
#抗锯齿是图像处理中重要的步骤，可避免图像在缩小或放大时产生的边缘锯齿、摩尔纹等失真现象。其核心思路是在缩放前先对图像进行高斯模糊，以平滑高频细节，再执行插值缩放。
def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    #根据输入图像的原始尺寸 (h, w) 和目标尺寸 size，计算高度和宽度方向的缩放因子
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])
    #factors>1缩放  <1放大
    # First, we have to determine sigma抗锯齿的关键是通过高斯模糊消除缩放后可能产生的高频噪声。标准差 sigma 由缩放因子决定（参考 skimage 的实现）
    # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )
    #当图像缩小（factor > 1）时，sigma 为正值，需要模糊；
    #当图像放大（factor <= 1）时，sigma 被限制为 0.001（近似无模糊，避免过度平滑）

    # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
    # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206 高斯核的大小需足够覆盖模糊范围，通常与 sigma 成正比。这里参考 Pillow 的实现，使用 2 * 2 * sigma 计算，并确保核大小为奇数（保证对称性）：
    # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
    # 计算初始核大小（基于 2*2*sigma，最小为 3）
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    # Make sure it is odd # 确保核大小为奇数（若为偶数则 +1）
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]

    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)#使用计算好的核大小 ks 和标准差 sigmas 对输入图像进行高斯模糊，调用内部函数 _gaussian_blur2d 实现
    #对模糊后的图像执行插值缩放，得到目标尺寸的输出：
    output = torch.nn.functional.interpolate(
        input, size=size, mode=interpolation, align_corners=align_corners)
    return output

#该函数用于计算图像处理中卷积或滤波操作所需的填充（padding）参数，确保卷积 / 滤波后输出尺寸与输入尺寸一致（即 “same padding”）。
#填充的核心作用是在输入图像边缘添加像素，避免因核（kernel）滑动导致的输出尺寸缩小，同时处理奇数 / 偶数核大小的不对称性
def _compute_padding(kernel_size):
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:# kernel_size 的长度是否≥2（需包含高度和宽度），否则抛出异常
        raise AssertionError(kernel_size)
    #每个维度（如高度、宽度），基础填充量为 kernel_size - 1。这是因为卷积操作中，若要保持输出尺寸与输入一致，总填充量需等于核大小减 1
    computed = [k - 1 for k in kernel_size] # 例如：kernel_size=(3,3) → computed=[2,2]

    # for even kernels we need to do asymmetric padding :(填充参数需按 (左, 右, 上, 下) 的顺序定义（对应 PyTorch 的 pad 函数要求），初始化一个全零列表
    out_padding = 2 * len(kernel_size) * [0] # 长度为 2×维度数，例如2D核→长度4

    #遍历每个维度（从最后一个维度开始，通常是宽度→高度），将基础填充量 computed_tmp 分为前后两部分
    #若 computed_tmp 为偶数（如 2）：前后填充量相等（如 1 和 1），即对称填充。
    #若 computed_tmp 为奇数（如 3）：前部分比后部分少 1（如 1 和 2），即不对称填充（偶数核的常见情况
    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)] # 从最后一个维度开始（如宽度）

        pad_front = computed_tmp // 2  # 前半部分填充
        pad_rear = computed_tmp - pad_front   # 后半部分填充
        # 按 (左,右,上,下) 顺序赋值（2D为例：i=0→宽度，i=1→高度）
        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding

#这是一个非常经典的多通道、多批次2D滤波函数的实现。它使用conv2d操作来高效地实现自定义滤波
#对4D输入张量 [batch, channels, height, width] 应用自定义的2D卷积核。
def _filter2d(input, kernel):
    # prepare kernel
    b, c, h, w = input.shape# [批次, 通道数, 高, 宽]
    tmp_kernel = kernel[:, None, ...].to(#在索引1的位置插入新维度  如果 kernel 形状是 [out_channels, kh, kw]  变换后：[out_channels, 1, kh, kw] 这样每个输出通道的核可以独立处理所有输入通道
        device=input.device, dtype=input.dtype)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)  #expand 进行广播，不分配新内存  形状从 [out_channels, 1, kh, kw] → [out_channels, c, kh, kw]  现在每个输出通道都有 c 个相同的卷积核，对应输入的所有通道

    height, width = tmp_kernel.shape[-2:] # 内核的高和宽

    padding_shape: list[int] = _compute_padding([height, width])     #填充
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

    # kernel and input tensor reshape to align element-wise or batch-wise params 重塑操作
    # 将 kernel 重塑为 (K*c, 1, kh, kw)，每个子 kernel 对应单个输入通道
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    # 将输入重塑为 (batch_size*c, K, padded_h, padded_w)，便于分组卷积
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = torch.nn.functional.conv2d(
        input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    out = output.view(b, c, h, w)
    return out

#函数用于生成高斯核（Gaussian kernel），是高斯模糊（_gaussian_blur2d）的核心组件。高斯核通过模拟高斯分布的权重，实现对图像的平滑处理，常用于抗锯齿、降噪等场景。
def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])# 转换为张量，形状 (1, 1)
    #从 sigma 的形状中获取批量大小（支持同时生成多个不同参数的高斯核）
    batch_size = sigma.shape[0]# sigma 形状通常为 (batch_size, 1)
    # 生成 [0, 1, ..., window_size-1]，减去核中心偏移量（window_size//2）
    x = (torch.arange(window_size, device=sigma.device,
         dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)# 扩展为 (batch_size, window_size)，每个样本共享相同的坐标序列
    #   若核尺寸为偶数，坐标序列整体加 0.5，确保核中心对齐到像素之间的间隙（避免偏移）
    if window_size % 2 == 0:
        x = x + 0.5  ## 偶数核调整，例如 window_size=4 → 坐标从 [-1.5, -0.5, 0.5, 1.5]

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)

#对输入张良的二维高斯模糊  核心思路是利用高斯核的可分离性，将二维模糊分解为两次一维模糊（先水平方向，再垂直方向），以提高计算效率。
def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):#sigma 是元组（如 (2.0, 1.5)），将其转换为张量，形状为 (1, 2)（批量大小为 1，包含垂直和水平方向的标准差）。
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1]) #从 kernel_size 中提取垂直方向核尺寸 ky 和水平方向核尺寸 kx，并转换为整数
    bs = sigma.shape[0] # 获取批量大小
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1)) # 水平方向高斯核
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))  # 垂直方向高斯核
    out_x = _filter2d(input, kernel_x[..., None, :])  # 水平方向模糊
    out = _filter2d(out_x, kernel_y[..., None])       # 垂直方向模糊


    return out

#将一系列视频帧（图像序列）合成为一个视频文件并保存到指定路径
def export_to_video(video_frames, output_video_path, fps):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")#指定视频的编码格式
    h, w, _ = video_frames[0].shape #第一帧图像中获取视频的高度 h 和宽度 w
    video_writer = cv2.VideoWriter( #创建视频写入器
        output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for i in range(len(video_frames)):#写入视频帧
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)


def export_to_gif(frames, output_gif_path, fps):
    """
    Export a list of frames to a GIF.

    Args:
    - frames (list): List of frames (as numpy arrays or PIL Image objects).
    - output_gif_path (str): Path to save the output GIF.
    - duration_ms (int): Duration of each frame in milliseconds.

    """
    # Convert numpy arrays to PIL Images if needed
    #遍历输入的 frames 列表，将 numpy 数组格式的帧转换为 PIL Image 对象（保持原本就是 PIL Image 的帧不变），确保后续处理格式统一
    pil_frames = [Image.fromarray(frame) if isinstance(
        frame, np.ndarray) else frame for frame in frames]

    pil_frames[0].save(output_gif_path.replace('.mp4', '.gif'), # 处理输出路径，确保后缀为 .gif
                       format='GIF',                              ## 指定格式为 GIF
                       append_images=pil_frames[1:],                # 追加后续所有帧
                       save_all=True,                    # 保存所有帧（而非仅第一帧）
                       duration=125,                  # 每帧持续时间（毫秒）
                       loop=0)                   # 0 表示无限循环播放

#将输入的图像或视频张量（通常是像素空间的数据）编码为变分自编码器（VAE）的潜在空间表示（latents），这在基于扩散模型的视频 / 图像生成任务中是常见的预处理步骤
def tensor_to_vae_latent(t, vae, scale=True):
    t = t.to(vae.dtype)  #t：输入张量
    if len(t.shape) == 5: #5维张量
        video_length = t.shape[1]  # 获取视频的帧数（f）
        # 重塑张量：将 (b, f, c, h, w) 合并为 (b*f, c, h, w)
        t = rearrange(t, "b f c h w -> (b f) c h w")
        # 用VAE编码：将批量帧转换为latents（VAE通常接收4维图像输入）
        latents = vae.encode(t).latent_dist.sample()
        # 重塑回视频格式：将 (b*f, c, h, w) 拆分为 (b, f, c, h, w)
        latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)
    elif len(t.shape) == 4:
        latents = vae.encode(t).latent_dist.sample()
    if scale:#VAE 的 latents 通常需要乘以模型配置中的 scaling_factor（例如 Stable Diffusion 中常用 0.18215），将 latents 的数值范围调整到扩散模型训练时使用的分布，避免因数值范围不匹配影响生成效果。
        latents = latents * vae.config.scaling_factor#缩放因子
    return latents


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train Stable Diffusion XL for InstructPix2Pix."
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
        default=20,
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
            "Run fine-tuning validation every X epochs. The validation process consists of running the text/image prompt"
            " multiple times: `args.num_validation_images`."
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
        default="constant",
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
        default=0.15,
        help="Conditioning dropout probability. Drops out the conditionings (image and edit prompt) used in training InstructPix2Pix. See section 3.2.1 in the paper: https://arxiv.org/abs/2211.09800.",
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
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
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
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
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
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
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
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
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
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image_folder",
        type=str,
        default=None,
        help=(
            "A set of paths to the controlnext conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        help=(
            "A set of paths to the controlnext conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--validation_control_folder",
        type=str,
        default=None,
        help=(
            "the validation control image"
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

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
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


#  This is for training using deepspeed.
#  Since now the DeepSpeed only supports trainging with only one model
#  So we create a virtual wrapper to contail all the models

class DeepSpeedWrapperModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        for name, value in kwargs.items():
            assert isinstance(value, nn.Module)
            self.register_module(name, value)


def main():

    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    torch.multiprocessing.set_start_method('spawn')  #PyTorch多进程启动方式为'spawn'，这是分布式训练的标准做法

    args = parse_args() #解析命令行参数，所有训练配置都在这里定义

    if args.non_ema_revision is not None:  #检查EMA（指数移动平均）模型版本，如果使用旧参数就发出警告    EMA使得数据平滑的过去
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
    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(  #初始化Accelerate库，这是HuggingFace的分布式训练工具
        gradient_accumulation_steps=args.gradient_accumulation_steps,#梯度累积步数，模拟大batch训练
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )

    generator = torch.Generator(
        device=accelerator.device).manual_seed(23123134)

    if args.report_to == "wandb":#配置训练过程中的日志记录工具 Weights & Biases
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Make one log on every process with the configuration for debugging.
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

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load scheduler, tokenizer and models.
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
        low_cpu_mem_usage=True,
        variant="fp16"
    )
    pose_net = PoseNet(noise_latent_channels=unet.config.block_out_channels[0])
    face_encoder = FusionFaceId(
        cross_attention_dim=1024,
        id_embeddings_dim=512,
        clip_embeddings_dim=1024,
        num_tokens=4,)
    face_model = FaceModel()

    # init adapter modules
    lora_rank = 128
    attn_procs = {}
    unet_svd = unet.state_dict()

    #最终目的是在冻结 UNet 大部分参数的同时，通过微调少量适配器参数，使模型适应特定任务
    for name in unet.attn_processors.keys():#为 UNet 模型的注意力层配置自定义注意力处理器
        if "transformer_blocks" in name and "temporal_transformer_blocks" not in name: #筛选出空间注意力层（仅处理单帧图像的空间维度注意力，不含时序维度）
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim  #确定交叉注意力维度  attn1自注意力层（处理同一特征图内部的注意力，无外部条件输入）
            if name.startswith("mid_block"): 
                hidden_size = unet.config.block_out_channels[-1] # 中间块的输出通道数
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])  # 提取上采样块的ID
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id] # 上采样块的通道数（倒序取
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])# 提取下采样块的ID
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:   #作用：通过低秩矩阵微调自注意力层，减少训练参数的同时保留空间特征建模能力。
                # print(f"This is AnimationAttnProcessor: {name}")
                attn_procs[name] = AnimationAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank)
            else:
                # print(f"This is AnimationIDAttnNormalizedProcessor: {name}")
                layer_name = name.split(".processor")[0]# 提取处理器对应的层名称（如"down_blocks.0.attentions.1"）
                weights = {
                    "id_to_k.weight": unet_svd[layer_name + ".to_k.weight"], # 从原UNet中加载key投影权重
                    "id_to_v.weight": unet_svd[layer_name + ".to_v.weight"], # 从原UNet中加载value投影权重
                }
                attn_procs[name] = AnimationIDAttnNormalizedProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank)
                attn_procs[name].load_state_dict(weights, strict=False)
        elif "temporal_transformer_blocks" in name:   #筛选出时序注意力层（处理视频帧之间的时序关系）。
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim# （隐藏层维度计算逻辑与空间注意力层相同，略）
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = XFormersAttnProcessor()
            else:
                attn_procs[name] = XFormersAttnProcessor()
    unet.set_attn_processor(attn_procs)

    # triggering the finetune mode微调模式（finetune mode）的权重加载逻辑，用于在启用微调时加载预训练好的模型权重
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
        if args.unet_model_finetune_path.endswith(".pth"):
            unet_state_dict = torch.load(args.unet_model_finetune_path, map_location="cpu")
            unet.load_state_dict(unet_state_dict, strict=True)
        else:
            print("unet weights loading fail")
            print(1/0)

        #计算 VAE 的缩放因子并初始化图像处理器，用于在训练过程中正确处理输入图像与 VAE latent 空间的尺寸转换。
        """vae.config.block_out_channels 是 VAE 配置中记录的每个下采样块的输出通道数列表，其长度等于下采样块的数量（例如，若长度为 4，则表示有 4 个下采样块）。
            len(vae.config.block_out_channels) - 1 得到下采样的总次数（通常 VAE 的第一个块不做下采样，从第二个块开始，因此减 1）。
            2** 下采样次数 即为缩放因子，例如：若下采样 3 次，缩放因子为 2^3=8，表示 latent 特征尺寸是原始图像的 1/8。
            作用：后续处理图像时，需要根据此缩放因子将原始图像尺寸缩放到与 VAE latent 特征匹配的尺寸（例如，原始图像 512x512，缩放因子 8，则 latent 尺寸为 64x64）。"""


    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

    # Freeze vae and image_encoder
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

    if args.enable_xformers_memory_efficient_attention: #xFormers：优化注意力计算，减少显存占用并加速训练（依赖硬件和库支持）
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


    if args.gradient_checkpointing: #梯度检查点
        unet.enable_gradient_checkpointing()
        

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32: #TF32 计算模式启用
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr: #学习率自动缩放
        args.learning_rate = (#调整后学习率 = 初始学习率 × 梯度累积步数 × 单卡批次大小 × 进程数（总 GPU 数）
            args.learning_rate * args.gradient_accumulation_steps *
            args.per_gpu_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
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

    # if accelerator.distributed_type == DistributedType.DEEPSPEED:   #适配 DeepSpeed 分布式训练框架
    #     ds_wrapper = DeepSpeedWrapperModel(
    #         unet=unet,
    #         controlnext=controlnext
    #     )
    #     unet = ds_wrapper.unet
    #     controlnext = ds_wrapper.controlnext

    #指定训练过程中需要更新的模型参数
    pose_net.requires_grad_(True)
    face_encoder.requires_grad_(True)

    parameters_list = []
    # 收集 pose_net 的参数
    for name, para in pose_net.named_parameters():
        para.requires_grad = True # 强制开启单个参数的梯度计算
        parameters_list.append({"params": para, "lr": args.learning_rate } )
    # 收集 face_encoder 的参数
    for name, para in face_encoder.named_parameters():
        para.requires_grad = True  # 强制开启单个参数的梯度计算
        parameters_list.append({"params": para, "lr": args.learning_rate } )

    
    """
    For more details, please refer to: https://github.com/dvlab-research/ControlNeXt/issues/14#issuecomment-2290450333
    This is the selective parameters part.
    As presented in our paper, we only select a small subset of parameters, which is fully adapted to the SD1.5 and SDXL backbones. By training fewer than 100 million parameters, we still achieve excellent performance. But this is is not suitable for the SD3 and SVD training. This is because, after SDXL, Stability faced significant legal risks due to the generation of highly realistic human images. After that, they stopped refining their models on human-related data, such as SVD and SD3, to avoid potential risks.
    To achieve optimal performance, it's necessary to first continue training SVD and SD3 on human-related data to develop a robust backbone before fine-tuning. Of course, you can also combine the continual pretraining and finetuning. So you can find that we direct provide the full SVD parameters.
    We have experimented with two approaches: 1.Directly training the model from scratch on human dancing data. 2. Continual training using a pre-trained human generation backbone, followed by fine-tuning a selective small subset of parameters. Interestingly, we observed no significant difference in performance between these two methods.
    """

    for name, para in unet.named_parameters():
        if "attentions" in name:
            para.requires_grad = True
            parameters_list.append({"params": para})
        else:
            para.requires_grad = False

    optimizer = optimizer_cls(
        parameters_list,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # 检查参数（记录可训练/不可训练参数）
    if accelerator.is_main_process and args.log_trainable_parameters:
        # 打开两个文件用于    
        rec_txt1 = open('rec_para.txt', 'w')  # 记录不可训练的参数
        rec_txt2 = open('rec_para_train.txt', 'w')  # 可训练的参数
    
        # 遍历 unet 模型的所有参数
        for name, para in unet.named_parameters():
            if para.requires_grad is False:
                # 如果参数不可训练，写入 rec_para.txt
                rec_txt1.write(f'{name}\n')
            else:
                # 如果参数可训练，写入 rec_para_train.txt
                rec_txt2.write(f'{name}\n')
        
        # 关闭文件
        rec_txt1.close()
        rec_txt2.close()
    # DataLoaders creation:
    args.global_batch_size = args.per_gpu_batch_size * accelerator.num_processes

    root_path = args.data_root_path
    txt_path_1 = args.rec_data_path
    txt_path_2 = args.vec_data_path
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
    )
    train_dataset_2 = LargeScaleAnimationVideos(
        root_path=root_path,
        txt_path=txt_path_2,
        width=576,
        height=1024,
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
        shuffle=True
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil((len(train_dataloader_1) + len(train_dataloader_2)) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    unet, pose_net, face_encoder, optimizer, lr_scheduler, train_dataloader_1, train_dataloader_2 = accelerator.prepare(
        unet, pose_net, face_encoder, optimizer, lr_scheduler, train_dataloader_1, train_dataloader_2
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil((len(train_dataloader_1) + len(train_dataloader_2)) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    # 仅在主进程中初始化跟踪器（避免多进程重复初始化）
    if accelerator.is_main_process:
        accelerator.init_trackers("StableAnimator", config=vars(args))

    # Train!
    total_batch_size = args.per_gpu_batch_size * \
        accelerator.num_processes * args.gradient_accumulation_steps

    len_zeros = len(train_dataloader_1)
    len_ones = len(train_dataloader_2)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset_1)+len(train_dataset_2)}")
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
    
    def encode_image(pixel_values):  #将输入的图像像素值编码为特征嵌入
        pixel_values = _resize_with_antialiasing(pixel_values, (224, 224))  #图像缩放.到224*224
        pixel_values = (pixel_values + 1.0) / 2.0   #像素值范围调整  将-1.1 调整到【0.1】

        pixel_values = pixel_values.to(torch.float32) #像素值转换成float32未
        # Normalize the image with for CLIP input CLIP特征处理器预处理
        pixel_values = feature_extractor(
            images=pixel_values,
            do_normalize=True,# 启用归一化（使用 CLIP 预训练时的均值和标准差）
            do_center_crop=False,  # 不进行中心裁剪（已提前缩放到固定尺寸）
            do_resize=False,  # 不重复缩放（已通过抗锯齿缩放完成）
            do_rescale=False,  # 不重新缩放（像素值已调整到 [0,1]）
            return_tensors="pt", # 返回 PyTorch 张量
        ).pixel_values

        pixel_values = pixel_values.to(  #设备和数据类型对齐
            device=accelerator.device, dtype=image_encoder.dtype)
        image_embeddings = image_encoder(pixel_values).image_embeds  #图像编码生成嵌入
        image_embeddings= image_embeddings.unsqueeze(1)  #扩展维度
        return image_embeddings

    #生成和处理与时间相关的附加信息张量，在视频生成和时序相关的扩散模型中使用
    def _get_add_time_ids(
        fps, #视频帧率
        motion_bucket_id, #运动桶 ID，用于量化视频运动强度的参数
        noise_aug_strength, #噪声增强强度，控制生成过程中的噪声添加程度
        dtype,#生成张量的数据类型（如 torch.float32）
        batch_size,
        unet=None, #可选参数，扩散模型的 UNet 结构（此处未实际使用）
        device=None
    ):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength] #创建基本的参数列表

        
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype, device=device) #转化为张量并扩展维度
        add_time_ids = add_time_ids.repeat(batch_size, 1) #按批次大小复制
        return add_time_ids

    # Potentially load in the weights and states from a previous save 检查之前的检查点，加载模型权重和训练状态，实现训练过程的断点续训功能
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint 获取最新的检查点
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")] #检查检查点文件夹
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1])) #按数据排序
            path = dirs[-1] if len(dirs) > 0 else None #取最新的检查点

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None #取消续训，从头开始
        else:#加载检查点    
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))#加载权重和状态
            global_step = int(path.split("-")[1]) # 从检查点名称提取全局步数（如 checkpoint-5000 → 5000）
            #计算续训的全局步数和epoch
            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine. 显示进度条
    progress_bar = tqdm(range(global_step, args.max_train_steps),
                        disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        pose_net.train() #训练模式
        face_encoder.train()
        unet.train()
        train_loss = 0.0
        #创建两个数据加载器的迭代器
        iter1 = iter(train_dataloader_1)
        iter2 = iter(train_dataloader_2)
        list_0_and_1 = [0] * len_zeros + [1] * len_ones  # 生成由 0 和 1 组成的随机列表，用于决定每个 step 使用哪个数据加载器
        random.shuffle(list_0_and_1)
        for step in range(0, len(list_0_and_1)):
            current_idx = list_0_and_1[step] # 获取当前step对应的数据源索引（0或1）
            if current_idx == 0:
                try:# 从第一个数据加载器获取数据# 从第一个数据加载器获取数据
                    batch = next(iter1)
                except StopIteration:# 若迭代器耗尽，重新创建迭代器并获取数据
                    iter1 = iter(train_dataloader_1)
                    batch = next(iter1)
            elif current_idx == 1:
                try:# 从第二个数据加载器获取数据
                    batch = next(iter2)
                except StopIteration:
                    iter2 = iter(train_dataloader_2)
                    batch = next(iter2)

            # Skip steps until we reach the resumed step从 checkpoint 恢复训练时，跳过已经训练过的步骤，直接从断点处继续训练
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(pose_net, face_encoder, unet):#前向计算与梯度累积
                with accelerator.autocast():
                    #数据加载与预处理
                    pixel_values = batch["pixel_values"].to(weight_dtype).to(
                        accelerator.device, non_blocking=True
                    )
                    conditional_pixel_values = batch["reference_image"].to(weight_dtype).to(
                        accelerator.device, non_blocking=True
                    )
                    #VAE 编码：像素空间 → Latent 空间
                    latents = tensor_to_vae_latent(pixel_values, vae).to(dtype=weight_dtype)

                    # Get the text embedding for conditioning.
                    encoder_hidden_states = encode_image(conditional_pixel_values).to(dtype=weight_dtype)
                    image_embed = encoder_hidden_states.clone()
                    #条件图像噪声增强
                    train_noise_aug = 0.02
                    conditional_pixel_values = conditional_pixel_values + train_noise_aug * \
                        randn_tensor(conditional_pixel_values.shape, generator=generator, device=conditional_pixel_values.device, dtype=conditional_pixel_values.dtype)
                    conditional_latents = tensor_to_vae_latent(conditional_pixel_values, vae, scale=False)#条件图像的 Latent 空间转换

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents) #生成与latent匹配的噪音
                    bsz = latents.shape[0] #批次大小
                    # Sample a random timestep for each image
                    sigmas = rand_cosine_interpolated(#生成余弦插值的随机噪声强度
                        shape=[bsz,], 
                        image_d=image_d, 
                        noise_d_low=noise_d_low,
                        noise_d_high=noise_d_high, 
                        sigma_data=sigma_data, 
                        min_value=min_value, 
                        max_value=max_value
                    ).to(latents.device, dtype=weight_dtype)

                    # sigmas = rand_log_normal(shape=[bsz,], loc=0.7, scale=1.6).to(latents)
                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    sigmas_reshaped = sigmas.clone() #复制噪声强度数组 sigmas（避免后续操作修改原始数据）
                    while len(sigmas_reshaped.shape) < len(latents.shape): #调整维度
                        sigmas_reshaped = sigmas_reshaped.unsqueeze(-1)


                    noisy_latents  = latents + noise * sigmas_reshaped   #生成带噪潜在变量 noisy_latents    前向加噪公式
                    #将 sigmas 映射为时间步 timesteps
                    timesteps = torch.Tensor([0.25 * sigma.log() for sigma in sigmas]).to(latents.device, dtype=weight_dtype)

                    #归一化带噪潜在变量 inp_noisy_latents
                    inp_noisy_latents = noisy_latents  / ((sigmas_reshaped**2 + 1) ** 0.5)
                    
                    added_time_ids = _get_add_time_ids(#构建附加时间条件信息的工具函数
                        fps=14,
                        motion_bucket_id=127.0,
                        noise_aug_strength=train_noise_aug, # noise_aug_strength == 0.0 # 噪声增强强度：此处为0.0（关闭）
                        dtype=encoder_hidden_states.dtype,
                        batch_size=bsz,# 批次大小：与当前训练批次匹配
                        unet=unet,
                        device=latents.device
                    )
                    #生成的added_time_ids是一个条件张量，形状通常为 [batch_size, N]（N 为条件维度，如 3 或更多），包含了fps、motion_bucket_id、noise_aug_strength等信息的编码值。
                    added_time_ids = added_time_ids.to(latents.device)

                    # Conditioning dropout to support classifier-free guidance during inference. For more details
                    # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.条件丢弃机制，用于在训练阶段支持扩散模型的无分类器引导（Classifier-Free Guidance, CFG） 推理
                    if args.conditioning_dropout_prob is not None:#args.conditioning_dropout_prob：条件丢弃的基础概率（例如 0.1 表示 10% 的基础丢弃概率）。
                        random_p = torch.rand(#random_p：生成形状为[batch_size,]的随机张量，每个元素是[0,1)区间的均匀分布随机数，用于决定是否丢弃每个样本的条件。
                            bsz, device=latents.device, generator=generator)
                        # Sample masks for the edit prompts.# 生成文本条件的丢弃掩码
                        prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                        prompt_mask = prompt_mask.reshape(bsz, 1, 1)
                        # Final text conditioning.# 用零向量替换被丢弃的文本条件
                        null_conditioning = torch.zeros_like(encoder_hidden_states)
                        encoder_hidden_states = torch.where(
                            prompt_mask, null_conditioning, encoder_hidden_states)

                        # Sample masks for the original images.图像丢弃
                        image_mask_dtype = conditional_latents.dtype# 生成图像条件的丢弃掩码
                        image_mask = 1 - (
                            (random_p >= args.conditioning_dropout_prob).to(
                                image_mask_dtype)
                            * (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype)
                        )
                        image_mask = image_mask.reshape(bsz, 1, 1, 1)
                        # Final image conditioning.# 用零向量替换被丢弃的图像条件
                        conditional_latents = image_mask * conditional_latents

                    # Concatenate the `conditional_latents` with the `noisy_latents`.条件潜变量与噪声潜变量的维度对齐
                    #将静态的conditional_latents（通常是参考图像的潜变量）扩展为与视频序列noisy_latents匹配的时序维度，确保两者能在时间维度上对齐并拼接。
                    conditional_latents = conditional_latents.unsqueeze(
                        1).repeat(1, noisy_latents.shape[1], 1, 1, 1)
                    # 姿态信息处理（pose_pixels → pose_latents）
                    pose_pixels = batch["pose_pixels"].to(
                        dtype=weight_dtype, device=accelerator.device, non_blocking=True
                    )
                    #faceid_embeds：人脸身份特征嵌入，通常由专门的人脸编码器（如 ArcFace）提取，用于唯一标识某个人脸的身份特征，形状可能为 [batch_size, 512]
                    #确保生成的人脸在多帧中保持一致的身份特征
                    faceid_embeds = batch["faceid_embeds"].to(
                        dtype=weight_dtype, device=accelerator.device, non_blocking=True
                    )
                    pose_latents = pose_net(pose_pixels)

                    # print("This is faceid_latents calculation")
                    # print(faceid_embeds.size())  # [1, 512]
                    # print(image_embed.size())  # [1, 1, 1024]
                    #人脸特征与潜变量的融合以及多条件潜变量的拼接
                    #这是一个特征融合模块（可能是自定义的神经网络），用于将人脸身份特征（faceid_embeds）与图像语义特征（image_embed）进行融合，输出融合后的人脸潜变量 faceid_latents。
                    faceid_latents = face_encoder(faceid_embeds, image_embed)

                    #拼接：若 inp_noisy_latents 形状为 [1, 14, 4, 64, 64]（1 个样本、14 帧、4 通道、64x64 尺寸），conditional_latents 形状为 [1, 14, 4, 64, 64]，则拼接后为 [1, 14, 8, 64, 64]
                    inp_noisy_latents = torch.cat(
                        [inp_noisy_latents, conditional_latents], dim=2)
                    target = latents  #目标潜变量target   latents 通常是干净的目标潜变量（即不含噪声的真实潜变量，如原始视频帧经 VAE 编码后的结果

                    # print(f"the size of encoder_hidden_states: {encoder_hidden_states.size()}") # [1, 1, 1024]
                    # print(f"the size of face latents: {faceid_latents.size()}") # [1, 4, 1024]
                    encoder_hidden_states = torch.cat([encoder_hidden_states, faceid_latents], dim=1)
                    #假设 encoder_hidden_states 为 [1, 1, 1024]，faceid_latents 为 [1, 4, 1024]，拼接后 encoder_hidden_states 形状变为 [1, 1+4, 1024] = [1, 5, 1024]
                    encoder_hidden_states = encoder_hidden_states.to(latents.dtype) #数据类型对齐
                    #inp_noisy_latents 是带噪声的输入潜变量（扩散过程中的中间状态），同样需要与 latents（干净的目标潜变量）保持数据类型一致，确保后续损失计算（如 MSE 损失）时的数值兼容性。
                    inp_noisy_latents = inp_noisy_latents.to(latents.dtype)
                    #pose_latents 是姿态信息编码后的潜变量（指导动作 / 姿态生成），转换为 latents 相同的数据类型后，才能与其他潜变量或特征在模型中进行融合计算（如作为 U-Net 的条件输入）。
                    pose_latents = pose_latents.to(latents.dtype)

                    # Predict the noise residual 前向传播
                    model_pred = unet(
                        inp_noisy_latents, timesteps, encoder_hidden_states,
                        added_time_ids=added_time_ids,
                        pose_latents=pose_latents,
                        ).sample
                        #U-Net 的输出通常是一个分布（如高斯分布的均值和方差），.sample 表示从该分布中采样出最终的预测结果 model_pred。
                        #在训练中，这个预测结果会与真实目标（如干净的潜变量 latents 或真实噪声）计算损失，用于更新模型参数。
                    

                    sigmas = sigmas_reshaped
                    # Denoise the latents 扩散模型训练中的去噪计算和人脸掩码的预处理
                    # 计算去噪系数# 计算去噪系数
                    c_out = -sigmas / ((sigmas**2 + 1)**0.5)#c_skip 可理解为 “保留原始噪声潜变量的比例”，噪声越大（sigmas 越大），c_skip 越小，模型预测（model_pred）的权重（c_out）越大，符合 “噪声多的时候更依赖模型去噪” 的逻辑。
                    c_skip = 1 / (sigmas**2 + 1)#c_out 是模型预测的系数，负号表示模型预测的是 “需要减去的噪声”（与噪声方向相反）。
                    # 得到去噪后的潜变量  扩散过程中 “去噪一步” 的结果
                    denoised_latents = model_pred * c_out + c_skip * noisy_latents
                    # 计算损失权重（用于后续损失函数）  用于后续损失函数的加权系数，目的是让模型在不同噪声尺度下的学习更均衡
                    weighing = (1 + sigmas ** 2) * (sigmas**-2.0)
                    # 将人脸掩码转换为正确的数据类型和设备
                    tgt_face_masks = batch["tgt_face_masks"].to(
                        dtype=weight_dtype, device=accelerator.device, non_blocking=True
                    )
                    # 调整维度：将 [batch, frames, channel, height, width] 转换为 [(batch*frames), channel, height, width]
                    tgt_face_masks = rearrange(tgt_face_masks, "b f c h w -> (b f) c h w")
                    # 调整空间尺寸：插值到与目标潜变量（target）相同的高宽
                    tgt_face_masks = F.interpolate(tgt_face_masks, size=(target.size()[-2], target.size()[-1]), mode='nearest')
                    # 恢复时序维度：将 [(batch*frames), channel, height, width] 转换回 [batch, frames, channel, height, width]
                    tgt_face_masks = rearrange(tgt_face_masks, "(b f) c h w -> b f c h w", f=args.sample_n_frames)

                    # MSE loss loss = 均值( 权重因子 × (去噪潜变量 - 目标潜变量)² × (1 + 人脸掩码) )
                    loss = torch.mean(
                        (weighing.float() * (denoised_latents.float() -
                        target.float()) ** 2 * (1 + tgt_face_masks)).reshape(target.shape[0], -1),
                        dim=1,
                    )
                    loss = loss.mean()

                    # Gather the losses across all processes for logging (if we use distributed training).分布式训练场景下收集所有进程的损失值并计算平均值
                    avg_loss = accelerator.gather(
                        loss.repeat(args.per_gpu_batch_size)).mean()
                    train_loss += avg_loss.item() / args.gradient_accumulation_steps

                    # Backpropagate反向传播梯度
                    accelerator.backward(loss)
                    #梯度裁剪的作用：防止训练中出现 “梯度爆炸”（梯度值过大导致参数更新不稳定）。通过限制参数梯度的 L2 范数不超过 args.max_grad_norm（如 1.0），保证训练稳定性。
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                    optimizer.step()  #更新参数
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    with torch.cuda.device(latents.device):#清理当前 CUDA 设备上的未被使用的显存缓存
                        torch.cuda.empty_cache()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                # save checkpoints!
                # if global_step % args.checkpointing_steps == 0 and (accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED):
                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`训练过程中检查点（checkpoint）数量管理的逻辑，用于在保存新检查点前确保总数量不超过用户设定的上限（checkpoints_total_limit），避免磁盘空间被过度占用。
                    if args.checkpoints_total_limit is not None and accelerator.is_main_process:
                        checkpoints = os.listdir(args.output_dir)  #列出输出目录（args.output_dir）中所有文件 / 文件夹。
                        checkpoints = [    #筛选出以checkpoint开头的文件夹  如checkpoint-1000
                            d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(   #按检查点的步数排序（通过分割文件名提取数字部分，如checkpoint-1000提取1000），确保最旧的检查点排在前面。
                            checkpoints, key=lambda x: int(x.split("-")[1]))

                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        #当现有检查点数量 ≥ 最大限制时，计算需要删除的数量：num_to_remove = 现有数量 - 最大限制 + 1（+1是因为即将保存 1 个新检查点，需预留位置）if len(checkpoints) >= args.checkpoints_total_limit:   
                        if len(checkpoints) >= args.checkpoints_total_limit:   
                            num_to_remove = len(
                                checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(
                                f"removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:# 删除旧检查点
                                removing_checkpoint = os.path.join(
                                    args.output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(
                        args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    # 解包模型（去除分布式训练包装，如DDP）  unwrap_model 会剥离这些包装，获取原始模型对象，确保保存的是纯净的模型参数
                    unwrap_unet = accelerator.unwrap_model(unet)
                    unwrap_pose_net = accelerator.unwrap_model(pose_net)
                    unwrap_face_encoder = accelerator.unwrap_model(face_encoder)
                    # 提取模型状态字典（参数） PyTorch 中，state_dict 是一个字典，键为层名称，值为对应层的权重参数（如权重矩阵、偏置项），是保存 / 加载模型的标准方式。
                    unwrap_unet_state_dict = unwrap_unet.state_dict()
                    # 保存状态字典到指定路径
                    torch.save(unwrap_unet_state_dict, os.path.join(args.output_dir, f"checkpoint-{global_step}", f"unet-{global_step}.pth"))
                    unwrap_pose_net_state_dict = unwrap_pose_net.state_dict()
                    torch.save(unwrap_pose_net_state_dict, os.path.join(args.output_dir, f"checkpoint-{global_step}", f"pose_net-{global_step}.pth"))
                    unwrap_face_encoder_state_dict = unwrap_face_encoder.state_dict()
                    torch.save(unwrap_face_encoder_state_dict, os.path.join(args.output_dir, f"checkpoint-{global_step}", f"face_encoder-{global_step}.pth"))
                    logger.info(f"Saved state to {save_path}")   #记录检查点保存路径到日志，便于追踪训练过程中的检查点生成情况。
            
                if accelerator.is_main_process:
                    # sample images!验证流程的核心是调用 log_validation 函数生成样本视频，并在使用 EMA 模型时进行参数切换，最后清理 GPU 缓存。
                    if global_step % args.validation_steps == 0:
                        logger.info(
                            f"Running validation... \n Generating {args.num_validation_images} videos."
                        )
                        # create pipeline  
                        if args.use_ema:
                            # Store the UNet parameters temporarily and load the EMA parameters to perform inference.# 临时保存原始UNet参数，加载EMA参数用于推理
                            #EMA（Exponential Moving Average，指数移动平均）是一种模型参数平滑技术，通过维护参数的滑动平均值提升模型泛化能力。
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
                            height=768,
                            torch_dtype=weight_dtype,
                            validation_image_folder=args.validation_image_folder,
                            validation_image=args.validation_image,
                            validation_control_folder=args.validation_control_folder,
                            output_dir=args.output_dir,
                            generator=generator,
                            global_step=global_step,
                            num_validation_cases=1,
                            num_frames=args.num_frames,
                        )

                        if args.use_ema:
                            # Switch back to the original UNet parameters.# 切换回原始UNet参数，继续训练
                            ema_unet.restore(unet.parameters())

                        with torch.cuda.device(latents.device):#清理 GPU 缓存
                            torch.cuda.empty_cache()

            logs = {"step_loss": loss.detach().item(
            ), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break
            
            
    # save checkpoints!
    # if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
    if accelerator.is_main_process:
        save_path = os.path.join(
            args.output_dir, f"checkpoint-last")
        accelerator.save_state(save_path)
        # 解包模型（去除分布式训练包装，如DDP）  unwrap_model 会剥离这些包装，获取原始模型对象，确保保存的是纯净的模型参数
        unwrap_unet = accelerator.unwrap_model(unet)
        unwrap_pose_net = accelerator.unwrap_model(pose_net)
        unwrap_face_encoder = accelerator.unwrap_model(face_encoder)
        # 提取模型状态字典（参数） PyTorch 中，state_dict 是一个字典，键为层名称，值为对应层的权重参数（如权重矩阵、偏置项），是保存 / 加载模型的标准方式。
        unwrap_unet_state_dict = unwrap_unet.state_dict()
        # 保存状态字典到指定路径
        torch.save(unwrap_unet_state_dict, os.path.join(args.output_dir, f"checkpoint-{global_step}", f"unet-{global_step}.pth"))
        unwrap_pose_net_state_dict = unwrap_pose_net.state_dict()
        torch.save(unwrap_pose_net_state_dict, os.path.join(args.output_dir, f"checkpoint-{global_step}", f"pose_net-{global_step}.pth"))
        unwrap_face_encoder_state_dict = unwrap_face_encoder.state_dict()
        torch.save(unwrap_face_encoder_state_dict, os.path.join(args.output_dir, f"checkpoint-{global_step}", f"face_encoder-{global_step}.pth"))
        logger.info(f"Saved state to {save_path}")   #记录检查点保存路径到日志，便于追踪训练过程中的检查点生成情况。



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
        num_validation_cases=1,
):
    logger.info("Running validation... ")
    # 解包模型（剥离分布式训练包装，如DDP）
    validation_unet = accelerator.unwrap_model(unet)
    validation_image_encoder = accelerator.unwrap_model(image_encoder)
    validation_vae = accelerator.unwrap_model(vae)
    validation_pose_net = accelerator.unwrap_model(pose_net)
    validation_face_encoder = accelerator.unwrap_model(face_encoder)
    # 构建验证专用pipeline
    pipeline = ValidationAnimationPipeline(
        vae=validation_vae,
        image_encoder=validation_image_encoder,
        unet=validation_unet,
        scheduler=scheduler,
        feature_extractor=feature_extractor,
        pose_net=validation_pose_net,
        face_encoder=validation_face_encoder,
    )
    pipeline = pipeline.to(accelerator.device)  
    validation_images = load_images_from_folder(validation_image_folder)

    validation_image_path = validation_image
    # 🚀 强制缩放参考图
    if validation_image is None:
        validation_image = validation_images[0].resize((width, height))
    else:
        validation_image = Image.open(validation_image).convert('RGB').resize((width, height))
        
    validation_control_images = load_images_from_folder(validation_control_folder)
    # 🚀 强制缩放所有骨架图，绝对对齐尺寸！
    validation_control_images = [img.resize((width, height)) for img in validation_control_images]

    val_save_dir = os.path.join(output_dir, "validation_images")
    if not os.path.exists(val_save_dir):
        os.makedirs(val_save_dir)

    with accelerator.autocast():
        for val_img_idx in range(num_validation_cases):
            num_frames = num_frames
            # num_frames = len(validation_control_images)
            face_helper.clean_all() # 清空人脸处理工具的缓存，避免上一轮结果干扰
            # 读取参考图像（BGR格式，cv2默认读取方式）
            validation_face = cv2.imread(validation_image_path)
            # validation_image_bgr = cv2.cvtColor(validation_face, cv2.COLOR_RGB2BGR)
            validation_image_face_info = app.get(validation_face)   # 使用insightface模型检测人脸信息（包含边界框、特征向量等）
            if len(validation_image_face_info) > 0:
                 # 按人脸面积（边界框宽×高）排序，取面积最大的人脸（通常是最清晰、居中的人脸）
                validation_image_face_info = sorted(validation_image_face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[-1]
                # 提取最大人脸的特征向量
                validation_image_id_ante_embedding = validation_image_face_info['embedding']
            else:# 未检测到人脸，特征向量设为None
                validation_image_id_ante_embedding = None

            if validation_image_id_ante_embedding is None:
                # 手动读取图像到人脸处理工具
                face_helper.read_image(validation_face)
                # 提取5点人脸关键点（用于对齐），只保留中心人脸
                face_helper.get_face_landmarks_5(only_center_face=True)
                # 根据关键点对齐并裁剪人脸（标准化人脸姿态和尺寸）
                face_helper.align_warp_face()
                # 检查对齐后是否有有效人脸
                if len(face_helper.cropped_faces) == 0:
                    # 完全无法检测时，用全零向量替代（避免程序崩溃，效果会受影响）
                    validation_image_id_ante_embedding = np.zeros((512,))
                else:
                    # 从对齐后的人脸中提取特征向量
                    validation_image_align_face = face_helper.cropped_faces[0]
                    print('fail to detect face using insightface, extract embedding on align face')
                    validation_image_id_ante_embedding = handler_ante.get_feat(validation_image_align_face)

            video_frames = pipeline(#动画生成验证管道
                image=validation_image,
                image_pose=validation_control_images,
                height=height,
                width=width,
                num_frames=num_frames,
                tile_size=num_frames,
                tile_overlap=4,
                decode_chunk_size=4,
                motion_bucket_id=127.,
                fps=7,
                min_guidance_scale=3,
                max_guidance_scale=3,
                noise_aug_strength=0.02,
                num_inference_steps=25,
                generator=generator,
                output_type="pil",
                validation_image_id_ante_embedding=validation_image_id_ante_embedding,
            ).frames[0]
            # save_combined_frames(video_frames, validation_images, validation_control_images, val_save_dir)

            out_file = os.path.join(
                val_save_dir,
                f"step_{global_step}_val_img_{val_img_idx}.mp4",
            )
            # print(video_frames.size()) # [16, 3, 512, 512]
            for i in range(num_frames):
                img = video_frames[i]
                video_frames[i] = np.array(img)
            export_to_gif(video_frames, out_file, 8)

    del pipeline
    torch.cuda.empty_cache()



if __name__ == "__main__":
    main()
