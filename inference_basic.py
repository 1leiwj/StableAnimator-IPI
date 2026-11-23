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

    # 引导尺度（控制文本提示对生成的影响强度）
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=2.0,
        required=False
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
        default=4,
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
        low_cpu_mem_usage=True,
    )
    pose_net = PoseNet(noise_latent_channels=unet.config.block_out_channels[0])
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

    # resume the previous checkpoint 从之前保存的 checkpoint（检查点）中恢复模型权重，确保推理时使用训练好的参数
    if args.posenet_model_name_or_path is not None and args.face_encoder_model_name_or_path is not None and args.unet_model_name_or_path is not None:
        print("Loading existing posenet weights, face_encoder weights and unet weights.")
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
        if args.unet_model_name_or_path.endswith(".pth"):
            unet_state_dict = torch.load(args.unet_model_name_or_path, map_location="cpu")
            unet.load_state_dict(unet_state_dict, strict=True)
        else:
            print("unet weights loading fail")
            print(1/0)

    torch.cuda.empty_cache()
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)
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
    ).to(device='cuda', dtype=weight_dtype)

    os.makedirs(args.output_dir, exist_ok=True)

    validation_image_path = args.validation_image
    validation_image = Image.open(args.validation_image).convert('RGB')
    validation_control_images = load_images_from_folder(args.validation_control_folder, width=args.width, height=args.height)

    num_frames = len(validation_control_images)  ## 获取姿态图像序列的长度（即生成视频的帧数）
    face_model.face_helper.clean_all() # # 清空人脸处理工具的缓存，避免之前的残留数据干扰
    validation_face = cv2.imread(validation_image_path) # # 用 OpenCV 读取参考图像（默认读取为 BGR 格式）
    validation_image_bgr = cv2.cvtColor(validation_face, cv2.COLOR_RGB2BGR) # 确保图像为 BGR 格式（符合 OpenCV 处理习惯）
    validation_image_face_info = face_model.app.get(validation_image_bgr)  # 用人脸检测模型（如 InsightFace）提取参考图像中的人脸信息
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
        motion_bucket_id=127.,                   # 运动强度参数（值越大，视频运动越剧烈） 
        fps=7,                                   # 视频帧率（每秒7帧）
        min_guidance_scale=args.guidance_scale,  # 引导尺度最小值（控制参考图像的影响程度）
        max_guidance_scale=args.guidance_scale,  # 引导尺度最大值（与最小值相同，固定引导强度）
        noise_aug_strength=args.noise_aug_strength, # 噪声增强强度（增加生成多样性）
        num_inference_steps=args.num_inference_steps, # 扩散模型推理步数（步数越多，生成越精细）
        generator=generator,                        # 随机数生成器（控制生成结果的可复现性）
        output_type="pil",                           # 输出格式（PIL图像列表）
        validation_image_id_ante_embedding=validation_image_id_ante_embedding,  # 参考图像的人脸特征（保持人脸一致性）
    ).frames[0]

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
