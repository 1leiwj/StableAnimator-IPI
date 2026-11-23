import argparse
import json
import math
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from diffusers.models.attention_processor import AttnProcessor2_0, XFormersAttnProcessor
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import torch
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler

from animation.modules.attention_processor import AnimationAttnProcessor
from animation.modules.attention_processor_normalized import AnimationIDAttnNormalizedProcessor
from animation.modules.face_model import FaceModel
from animation.modules.id_encoder import FusionFaceId
from animation.modules.pose_net import PoseNet
from animation.modules.unet_ipi import UNetSpatioTemporalConditionModel
from animation.modules.ipi_adapter import MotionAdapter
from animation.pipelines.inference_pipeline_animation_ipi import InferenceAnimationPipeline
import random

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
def ordered_frame_files(folder):
    directory = Path(folder)
    if not directory.is_dir():
        raise ValueError(f"Frame folder does not exist: {directory}")
    numbered = []
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() != ".png":
            continue
        number = path.stem.rsplit("_", 1)[-1]
        if not number.isascii() or not number.isdigit():
            raise ValueError(f"PNG filename has no numeric frame suffix: {path}")
        numbered.append((int(number), path))
    numbered.sort()
    if not numbered:
        raise ValueError(f"No PNG frames found in {directory}.")
    indices, paths = zip(*numbered)
    if any(b != a + 1 for a, b in zip(indices, indices[1:])):
        raise ValueError(f"Frame numbers must be consecutive and unique in {directory}.")
    return indices, paths


def read_frames(paths, width, height):
    frames = []
    for path in paths:
        with Image.open(path) as image:
            frames.append(image.convert("RGB").resize((width, height)))
    return frames


def read_pose_array(path, coordinate_space):
    pose = np.load(path, allow_pickle=False)
    if pose.ndim != 3 or pose.shape[1:] != (133, 2) or not len(pose) or pose.dtype.kind not in "fiu":
        raise ValueError(f"Expected a nonempty numeric [frames, 133, 2] pose array, got {pose.shape} ({pose.dtype}): {path}")
    pose = pose.astype(np.float32)
    missing = (pose == -1).all(axis=-1) if coordinate_space == "model" else (pose == -1).any(axis=-1)
    visible = np.isfinite(pose).all(axis=-1) & ~missing
    values = pose[visible]
    low, high = (-0.5, 1.5) if coordinate_space == "normalized" else (-1.001, 1.001)
    if not len(values) or values.min() < low or values.max() > high:
        raise ValueError(f"Invalid or unexpected {coordinate_space} DWPose coordinate range: {path}. "
                         "Supply normalized x/W,y/H coordinates or explicitly select model coordinates.")
    pose[~visible] = -1
    return pose

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
    - fps (int): Frames per second.

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
                       duration=round(1000 / fps),
                       loop=0)  # 0表示无限循环

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run pose-guided StableAnimator inference."
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
        help="存储验证用控制图像（如姿态序列帧）的文件夹路径",
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

    # 引导尺度（控制提示对生成的影响强度）
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=2.5,
        required=False
    )
    parser.add_argument(
        "--min_guidance_scale",
        type=float,
        default=None,
        required=False,
        help="首帧起始引导尺度 (默认等于 guidance_scale)"
    )
    parser.add_argument(
        "--max_guidance_scale",
        type=float,
        default=None,
        required=False,
        help="末帧最大引导尺度 (默认等于 guidance_scale)"
    )

    # 推理步数（扩散过程迭代次数）
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=25,
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
        default=16,
        required=False
    )

    # 分块重叠区域大小
    parser.add_argument(
        "--overlap",
        type=int,
        default=None,
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
        default=None,
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

    parser.add_argument("--ipi_model_name_or_path", type=str, default=None)
    parser.add_argument("--driving_video_folder", type=str, default=None)
    parser.add_argument("--pose_data_path", type=str, default=None)
    parser.add_argument("--motion_bucket_id", type=int, default=127)
    parser.add_argument("--fps_time_id", type=int, default=6, help="U-Net FPS conditioning ID, independent of playback rate.")
    parser.add_argument("--seed", type=int, default=23123134)
    parser.add_argument("--pose_scale", type=float, default=3.0, help="PoseNet 输出倍率")
    parser.add_argument("--reference_latent_scale", type=float, default=1.0)
    parser.add_argument("--reference_clip_scale", type=float, default=1.0)
    parser.add_argument("--ipi_scale", type=float, help="IPI 时序条件倍率（旧版默认 6，v2 默认 1）")
    parser.add_argument("--ipi_temporal_v2", action="store_true", help="使用逐帧运动残差 IPI")
    parser.add_argument("--ipi_joint_mode", choices=("auto", "on", "off"),
                        default=os.environ.get("IPI_JOINT_MODE", "auto"),
                        help="Use the joint residual when present, require it, or ablate it.")
    parser.add_argument("--retarget_pose", action="store_true", help="按参考图重定位驱动骨架")
    parser.add_argument("--match_ankle_height", action="store_true",
                        help="Match the driver's vertical lower-ankle displacement after retargeting.")
    parser.add_argument("--pose_root_motion_scale", type=float, default=1.0,
                        help="Scale whole-body root displacement after retargeting. 0 anchors the root.")
    parser.add_argument("--no_ipi", action="store_true", default=False)
    parser.add_argument("--is_statue", action="store_true", default=False)
    parser.add_argument("--pose_coordinate_space", choices=("normalized", "model"), default="normalized")
    parser.add_argument("--frame_start", type=int, default=0, help="First frame offset after whole-clip retargeting, inclusive.")
    parser.add_argument("--frame_end", type=int, help="Last frame offset after whole-clip retargeting, exclusive.")
    parser.add_argument("--independent_noise", action="store_true", help="Sample noise per frame rather than repeating the first tile.")
    parser.add_argument("--source_timeline", action="store_true",
                        help="Select the source clip's initial noise and CFG timeline for a sliced run.")
    parser.add_argument("--clip_batch_size", type=int, default=32)
    parser.add_argument("--skip_gif", action="store_true")

    args = parser.parse_args()
    if args.match_ankle_height and not args.retarget_pose:
        parser.error("--match_ankle_height requires --retarget_pose.")
    if args.frames_overlap is None:
        args.frames_overlap = 4 if args.overlap is None else args.overlap
    elif args.overlap is not None and args.overlap != args.frames_overlap:
        print(f"Using --frames_overlap={args.frames_overlap}; legacy --overlap={args.overlap} is overridden.")
    if not 0 <= args.seed < 2**63:
        parser.error("--seed must be in [0, 2**63).")
    if not math.isfinite(args.pose_root_motion_scale) or args.pose_root_motion_scale < 0:
        parser.error("--pose_root_motion_scale must be finite and nonnegative.")
    if args.pose_root_motion_scale != 1 and not args.retarget_pose:
        parser.error("--pose_root_motion_scale requires --retarget_pose.")
    if args.min_guidance_scale is None:
        args.min_guidance_scale = args.guidance_scale
    if args.max_guidance_scale is None:
        args.max_guidance_scale = args.guidance_scale
    if args.ipi_scale is None:
        args.ipi_scale = 1.0 if args.ipi_temporal_v2 else 6.0
    if not args.validation_image or (not args.retarget_pose and not args.validation_control_folder):
        parser.error("A reference image and either --retarget_pose or --validation_control_folder are required.")
    if args.retarget_pose and (not args.pose_data_path or args.pose_coordinate_space != "normalized"):
        parser.error("--retarget_pose requires normalized --pose_data_path coordinates.")
    if min(args.width, args.height) <= 0 or args.width % 8 or args.height % 8:
        parser.error("--width and --height must be positive multiples of 8.")
    if args.tile_size <= 0 or args.frames_overlap < 0 or args.frames_overlap >= args.tile_size:
        parser.error("Require 0 <= --frames_overlap < --tile_size.")
    if args.num_inference_steps <= 0 or args.clip_batch_size <= 0 or (args.decode_chunk_size is not None and args.decode_chunk_size <= 0):
        parser.error("Inference steps and batch sizes must be positive.")
    if args.frame_start < 0 or (args.frame_end is not None and args.frame_end <= args.frame_start):
        parser.error("Require 0 <= --frame_start < --frame_end when an end is supplied.")
    if args.fps_time_id < 0 or args.motion_bucket_id < 0:
        parser.error("FPS conditioning ID and motion bucket ID must be nonnegative.")
    if not all(math.isfinite(s) and s >= 0 for s in (args.pose_scale, args.ipi_scale, args.noise_aug_strength,
                                                      args.reference_latent_scale, args.reference_clip_scale)):
        parser.error("Pose/IPI/reference scales and noise augmentation must be finite and nonnegative.")
    if not all(s is None or math.isfinite(s) and s >= 1 for s in (
        args.guidance_scale, args.min_guidance_scale, args.max_guidance_scale
    )):
        parser.error("Guidance scales must be finite and at least 1. Use 1 for conditional-only denoising.")
    base_weights = (args.unet_model_name_or_path, args.posenet_model_name_or_path, args.face_encoder_model_name_or_path)
    if not all(base_weights):
        parser.error("U-Net, PoseNet and FaceEncoder checkpoint paths must be supplied together.")
    if args.ipi_temporal_v2:
        if not args.no_ipi and not args.ipi_model_name_or_path:
            parser.error("--ipi_temporal_v2 requires an IPI Adapter checkpoint unless --no_ipi is set.")
    return args


def run_inference(args, pipeline_probe=None):
    joint_mode = getattr(args, "ipi_joint_mode", "auto")  # Earlier replay manifests omit this field.
    if joint_mode not in ("auto", "on", "off"):
        raise ValueError("IPI_JOINT_MODE must be auto, on or off.")
    if joint_mode == "on" and (args.no_ipi or not args.ipi_temporal_v2):
        raise ValueError("IPI_JOINT_MODE=on requires IPI v2 with NO_IPI=0.")
    has_ipi_inputs = any((args.driving_video_folder, args.ipi_model_name_or_path, args.pose_data_path and not args.retarget_pose))
    use_ipi = not args.no_ipi and (has_ipi_inputs or args.retarget_pose or args.ipi_temporal_v2)
    if use_ipi and not all((args.driving_video_folder, args.ipi_model_name_or_path, args.pose_data_path)):
        raise ValueError("IPI requires a driving RGB folder, a pose array and Adapter weights. Use --no_ipi for PoseNet only.")
    ipi_mode = ("v2" if args.ipi_temporal_v2 else "legacy") if use_ipi else "off"
    print(f"IPI mode: {ipi_mode}, PoseNet scale: {args.pose_scale}, IPI scale: {args.ipi_scale}, steps: {args.num_inference_steps}", flush=True)

    for path in (args.unet_model_name_or_path, args.posenet_model_name_or_path, args.face_encoder_model_name_or_path,
                 args.ipi_model_name_or_path if use_ipi else None):
        if path and not Path(path).is_file():
            raise FileNotFoundError(f"Missing checkpoint file: {path}")
    output_dir = Path(args.output_dir)
    png_out_file = output_dir / "animated_images"
    if png_out_file.exists():
        raise FileExistsError(f"Output has already been reserved: {png_out_file}. Choose a new --output_dir.")

    validation_image_path = args.validation_image
    with Image.open(validation_image_path) as image:
        validation_image = image.convert("RGB")
    retargeted_pose = None
    source_pose = read_pose_array(args.pose_data_path, args.pose_coordinate_space) if use_ipi or args.retarget_pose else None
    control_indices = None
    if args.retarget_pose:
        full_frame_count = len(source_pose)
    else:
        control_indices, control_paths = ordered_frame_files(args.validation_control_folder)
        full_frame_count = len(control_paths)
    if source_pose is not None and len(source_pose) != full_frame_count:
        raise ValueError(f"Pose array has {len(source_pose)} frames, but the control sequence has {full_frame_count}.")
    if use_ipi:
        driving_indices, driving_paths = ordered_frame_files(args.driving_video_folder)
        if len(driving_paths) != full_frame_count or (control_indices is not None and driving_indices != control_indices):
            raise ValueError("Driving PNGs and pose inputs must have the same frame count and ordering.")
    frame_end = full_frame_count if args.frame_end is None else args.frame_end
    if not args.frame_start < frame_end <= full_frame_count:
        raise ValueError(f"Frame range [{args.frame_start}, {frame_end}) is outside the {full_frame_count}-frame input.")
    selected = slice(args.frame_start, frame_end)
    png_out_file.mkdir(parents=True, exist_ok=False)
    if args.retarget_pose:
        from DWPose.retarget import detect_reference_pose, render_body_poses, retarget_pose

        reference_pose = detect_reference_pose(validation_image)
        retargeted_pose = retarget_pose(source_pose, reference_pose, args.width, args.height,
                                        args.pose_root_motion_scale, ankle_height=args.match_ankle_height)[selected]
        validation_control_images = render_body_poses(retargeted_pose, args.width, args.height)
        np.save(output_dir / "retargeted_poses_dense133.npy", retargeted_pose)
        print(f"Retargeted {full_frame_count} poses; using frames [{args.frame_start}, {frame_end}).", flush=True)
    else:
        validation_control_images = read_frames(control_paths[selected], args.width, args.height)

    num_frames = len(validation_control_images)
    if source_pose is not None:
        pose_input = retargeted_pose if retargeted_pose is not None else source_pose[selected]
        for label, points in (("source", source_pose[selected]), ("PoseNet input" if args.retarget_pose else "IPI input", pose_input)):
            if label != "source" and not args.retarget_pose:
                continue
            missing = (points[:, [10, 13]] == -1).any(axis=-1).sum(axis=0)
            print(f"{label} missing ankle frames (right/left): {missing.tolist()} out of {num_frames}", flush=True)

    seed = args.seed
    seed_everything(seed)
    generator = torch.Generator(device="cuda").manual_seed(seed)

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
        ignore_mismatched_sizes=True,
    )
    spatial_multiple = 2 ** (len(vae.config.block_out_channels) - 1 + unet.num_upsamplers)
    if args.width % spatial_multiple or args.height % spatial_multiple:
        raise ValueError(f"This VAE and U-Net require width and height to be multiples of {spatial_multiple}.")
    ipi_adapter = MotionAdapter(
        cross_attention_dim=getattr(unet.config, "cross_attention_dim", 1024),
        clip_dim=1024,
        num_tokens=8,
    )
    unet.ipi_adapter = ipi_adapter

    pose_net = PoseNet(noise_latent_channels=unet.config.block_out_channels[0])
    face_encoder = FusionFaceId(
        cross_attention_dim=1024,
        id_embeddings_dim=512,
        # clip_embeddings_dim=image_encoder.config.hidden_size,
        clip_embeddings_dim=1024,
        num_tokens=4, )
    face_model = None if args.is_statue else FaceModel()

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
            attn_procs[name] = XFormersAttnProcessor() if is_xformers_available() else AttnProcessor2_0()
    #将构建好的注意力处理器字典（attn_procs）应用到 UNet 模型中，替换原有的注意力处理器，完成模型注意力机制的定制化配置。
    unet.set_attn_processor(attn_procs)

    # resume the previous checkpoint 从之前保存的 checkpoint（检查点）中恢复模型权重，确保推理时使用训练好的参数
    pose_net.load_state_dict(torch.load(args.posenet_model_name_or_path, map_location="cpu"), strict=True)
    face_encoder.load_state_dict(torch.load(args.face_encoder_model_name_or_path, map_location="cpu"), strict=True)
    unet_weights = torch.load(args.unet_model_name_or_path, map_location="cpu")
    # The temporal loader backfills old checkpoints with zero gates. That is valid
    # for legacy/off, but must never masquerade as a trained v2 branch.
    if ipi_mode == "v2":
        missing_motion = [k for k, _ in unet.named_parameters() if "ipi_temporal_" in k and k not in unet_weights]
        if missing_motion:
            raise ValueError(f"Missing trained v2 parameters in U-Net: {missing_motion[:8]}")
    missing, unexpected = unet.load_state_dict(unet_weights, strict=False)
    del unet_weights
    required_missing = [key for key in missing if not key.startswith("ipi_adapter.")
                        and (ipi_mode == "v2" or "ipi_temporal_" not in key)]
    if required_missing or unexpected:
        raise ValueError(f"Incomplete or incompatible U-Net: missing={required_missing[:8]}, unexpected={unexpected[:8]}")
    if torch.count_nonzero(pose_net.final_proj.weight).item() == 0 or pose_net.scale.item() == 0:
        print("WARNING: loaded PoseNet has no spatially varying output; check its checkpoint.", flush=True)

    joint_readout_rms = {}
    if use_ipi:
        state_dict = torch.load(args.ipi_model_name_or_path, map_location="cpu")
        has_joints = any(k.startswith("joint_motion.") for k in state_dict)
        if joint_mode == "on" and not has_joints:
            raise ValueError("The selected adapter checkpoint has no trained joint-motion branch.")
        if has_joints:
            if not args.ipi_temporal_v2:
                raise ValueError("A joint-motion checkpoint requires IPI_TEMPORAL_V2=1.")
            unet.ipi_adapter.enable_joint_motion(unet.config.block_out_channels)
        unet.ipi_adapter.load_state_dict(state_dict, strict=True)
        if has_joints:
            joint_readout_rms = {c: head.weight.detach().float().square().mean().sqrt().item()
                                 for c, head in unet.ipi_adapter.joint_motion.readouts.items()}
            if not all(math.isfinite(v) for v in joint_readout_rms.values()):
                raise ValueError("Non-finite weights in the joint motion readouts.")
            if joint_mode == "on" and not any(joint_readout_rms.values()):
                raise ValueError("All joint motion readouts are still zero. Check training before comparing videos.")
        if joint_mode == "off":
            unet.ipi_adapter.joint_motion = None
        print(f"Localized IPI joint residual: {unet.ipi_adapter.joint_motion is not None}", flush=True)
        unet.ipi_adapter.to(device="cuda", dtype=torch.float16)
        print("Loaded IPI Adapter weights.")

    torch.cuda.empty_cache()
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    unet.eval()
    pose_net.requires_grad_(False)
    face_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    weight_dtype = torch.float16
    # weight_dtype = torch.float32
    # weight_dtype = torch.bfloat16

    pipeline = InferenceAnimationPipeline(
        vae=vae,
        image_encoder=image_encoder,
        unet=unet,
        scheduler=noise_scheduler,
        feature_extractor=feature_extractor,
        pose_net=pose_net,
        face_encoder=face_encoder,
    ).to(device='cuda', dtype=torch.float16)

    validation_image_id_ante_embedding = np.zeros(512, dtype=np.float32)
    face_source = "disabled"
    if face_model is not None:
        face_model.face_helper.clean_all() # # 清空人脸处理工具的缓存，避免之前的残留数据干扰
        validation_image_bgr = cv2.cvtColor(np.asarray(validation_image), cv2.COLOR_RGB2BGR)
        validation_image_face_info = face_model.app.get(validation_image_bgr)  # 用人脸检测模型（如 InsightFace）提取参考图像中的人脸信息
        if len(validation_image_face_info) > 0:
            validation_image_face_info = sorted(validation_image_face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[-1]
            face_source = "detected"
            validation_image_id_ante_embedding = validation_image_face_info['embedding']
        else:
            validation_image_id_ante_embedding = None

        if validation_image_id_ante_embedding is None:
            face_model.face_helper.read_image(validation_image_bgr)
            face_model.face_helper.get_face_landmarks_5(only_center_face=True)
            face_model.face_helper.align_warp_face()

            if len(face_model.face_helper.cropped_faces) == 0:
                face_source = "zero_vector_fused_with_reference"
                validation_image_id_ante_embedding = np.zeros(512, dtype=np.float32)
            else:
                face_source = "aligned_face"
                validation_image_align_face = face_model.face_helper.cropped_faces[0]
                print('fail to detect face using insightface, extract embedding on align face')
                validation_image_id_ante_embedding = face_model.handler_ante.get_feat(validation_image_align_face)

        # generator = torch.Generator(device=accelerator.device).manual_seed(23123134)

    dwpose_raw_tensor = None
    clip_image_embeds = None
    if use_ipi:
        model_pose = pose_input.copy()
        if args.pose_coordinate_space == "normalized":
            visible = (model_pose != -1).all(axis=-1)
            model_pose[visible] = model_pose[visible] * 2 - 1
        dwpose_raw_tensor = torch.from_numpy(model_pose).reshape(1, num_frames, 266).to(device="cuda", dtype=torch.float16)
        with torch.no_grad():
            embeddings = []
            driving_paths = driving_paths[selected]
            for index in range(0, num_frames, args.clip_batch_size):
                frames = read_frames(driving_paths[index:index + args.clip_batch_size], args.width, args.height)
                pixels = feature_extractor(images=frames, return_tensors="pt").pixel_values
                pixels = pixels.to(device="cuda", dtype=torch.float16)
                embeddings.append(image_encoder(pixels).image_embeds)
            clip_image_embeds = torch.cat(embeddings, dim=0).unsqueeze(0)

    decode_chunk_size = args.decode_chunk_size # 从命令行参数获取解码分块大小
    pipeline_inputs = dict(
        image=validation_image,                  # 参考图像（生成视频的基础图像）               
        image_pose=validation_control_images,    # 姿态控制图像序列（指导人物动作）
        height=args.height,                      # 生成视频的高度
        width=args.width,                        # 生成视频的宽度
        num_frames=num_frames,                   # 视频帧数（由姿态控制图像数量决定）
        tile_size=args.tile_size,                # 分块生成时的块大小（用于大尺寸视频的内存优化）
        tile_overlap=args.frames_overlap,        # 分块间的重叠帧数（避免块间衔接生硬）
        decode_chunk_size=decode_chunk_size,     # VAE解码时的分块大小（避免解码大视频时OOM）
        motion_bucket_id=float(args.motion_bucket_id),                   # 运动强度参数（值越大，视频运动越剧烈） 
        fps=args.fps_time_id + 1,                # The pipeline subtracts one for SVD conditioning.
        min_guidance_scale=args.min_guidance_scale,
        max_guidance_scale=args.max_guidance_scale,
        noise_aug_strength=args.noise_aug_strength, # 噪声增强强度（增加生成多样性）
        num_inference_steps=args.num_inference_steps, # 扩散模型推理步数（步数越多，生成越精细）
        generator=generator,                        # 随机数生成器（控制生成结果的可复现性）
        output_type="pil",                           # 输出格式（PIL图像列表）
        validation_image_id_ante_embedding=validation_image_id_ante_embedding,  # 参考图像的人脸特征（保持人脸一致性）
        dwpose_raw=dwpose_raw_tensor,
        clip_image_embeds=clip_image_embeds,
        is_statue=args.is_statue,
        pose_scale=args.pose_scale,
        reference_latent_scale=args.reference_latent_scale,
        reference_clip_scale=args.reference_clip_scale,
        ipi_scale=args.ipi_scale,
        ipi_temporal_v2=args.ipi_temporal_v2,
        repeat_initial_noise=not args.independent_noise,
        timeline_start=args.frame_start if args.source_timeline else 0,
        timeline_length=full_frame_count if args.source_timeline else num_frames,
    )
    if pipeline_probe is not None:
        return pipeline_probe(pipeline, pipeline_inputs)
    video_frames = pipeline(**pipeline_inputs).frames[0]

    if len(video_frames) != num_frames:
        raise RuntimeError(f"The pipeline returned {len(video_frames)} frames; expected {num_frames}.")
    save_frames_as_png(video_frames, png_out_file)
    if not args.skip_gif:
        export_to_gif(video_frames, str(output_dir / "animation_video.gif"), fps=8)
    manifest = {
        **vars(args), "seed": seed, "ipi_mode": ipi_mode,
        "ipi_joint_active": use_ipi and unet.ipi_adapter.joint_motion is not None,
        "ipi_joint_readout_rms": joint_readout_rms,
        "frame_end": frame_end, "num_frames": num_frames, "face_source": face_source,
        "cfg_first_last": pipeline.guidance_scale[0, [0, -1]].flatten().float().cpu().tolist(),
        "scheduler_steps": pipeline.num_timesteps,
        "time_ids": [args.fps_time_id, args.motion_bucket_id, args.noise_aug_strength],
        "pose_net_learned_scale": pose_net.scale.item(),
        "ipi_gate_tanh": {name: param.float().tanh().item() for name, param in unet.named_parameters()
                          if ipi_mode == "v2" and name.endswith("ipi_temporal_gate")},
    }
    (output_dir / "inference_settings.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    run_inference(parse_args())
