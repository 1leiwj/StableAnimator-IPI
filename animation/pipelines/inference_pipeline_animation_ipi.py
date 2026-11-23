import inspect
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import PIL.Image
import einops
import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor, PipelineImageInput
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion \
    import _resize_with_antialiasing, _append_dims
from diffusers.utils import BaseOutput, logging
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor

from animation.modules.attention_processor import AnimationAttnProcessor, AnimationIDAttnProcessor
from einops import rearrange

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


def _frame_windows(num_frames: int, tile_size: int, tile_overlap: int) -> List[List[int]]:
    if num_frames < 1:
        raise ValueError("At least one pose frame is required.")
    if tile_size < 1:
        raise ValueError("tile_size must be positive.")
    if tile_overlap < 0:
        raise ValueError("tile_overlap cannot be negative.")
    if num_frames <= tile_size:
        return [list(range(num_frames))]
    if tile_overlap >= tile_size:
        raise ValueError("tile_overlap must be smaller than tile_size when tiling.")

    last_start = num_frames - tile_size
    starts = list(range(0, last_start + 1, tile_size - tile_overlap))
    if starts[-1] != last_start:
        starts.append(last_start)
    return [list(range(start, start + tile_size)) for start in starts]


def tensor2vid(video: torch.Tensor, processor: "VaeImageProcessor", output_type: str = "np"):
    # 解析视频张量的形状
    batch_size, channels, num_frames, height, width = video.shape
    outputs = []
    # 遍历批量中的每个视频
    for batch_idx in range(batch_size):
        # 提取单个视频，并调整维度顺序：(channels, num_frames, height, width) → (num_frames, channels, height, width)
        # 因为处理器通常期望帧维度在前
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        # 使用处理器将张量转换为指定格式（如 PIL 图像或 NumPy 数组）
        batch_output = processor.postprocess(batch_vid, output_type)
        outputs.append(batch_output)
    # 返回批量视频的列表
    return outputs


@dataclass
class InferenceAnimationPipelineOutput(BaseOutput):
    r"""
    Output class for mimicmotion pipeline.

    Args:
        frames (`[List[List[PIL.Image.Image]]`, `np.ndarray`, `torch.Tensor`]):
            List of denoised PIL images of length `batch_size` or numpy array or torch tensor of shape `(batch_size,
            num_frames, height, width, num_channels)`.
    """

    frames: Union[List[List[PIL.Image.Image]], np.ndarray, torch.Tensor]


class InferenceAnimationPipeline(DiffusionPipeline):
    r"""
    Pipeline to generate video from an input image using Stable Video Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKLTemporalDecoder`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        image_encoder ([`~transformers.CLIPVisionModelWithProjection`]):
            Frozen CLIP image-encoder ([laion/CLIP-ViT-H-14-laion2B-s32B-b79K]
            (https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)).
        unet ([`UNetSpatioTemporalConditionModel`]):
            A `UNetSpatioTemporalConditionModel` to denoise the encoded image latents.
        scheduler ([`EulerDiscreteScheduler`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images.
        pose_net ([`PoseNet`]):
            A `` to inject pose signals into unet.
    """

    model_cpu_offload_seq = "image_encoder->unet->vae"#定义模型组件的 CPU 卸载顺序。当启用 CPU 内存优化时（如显存不足时），会按照 image_encoder → unet → vae 的顺序将暂时不用的组件移至 CPU，释放 GPU 内存，提升大型模型的运行效率。
    _callback_tensor_inputs = ["latents"]#指定在回调函数（如 callback_on_step_end）中需要传入的张量名称

    def __init__(
            self,
            vae,
            image_encoder,
            unet,
            scheduler,
            feature_extractor,
            pose_net,
            face_encoder,
    ):
        super().__init__() # 调用父类 DiffusionPipeline 的初始化方法  继承 DiffusionPipeline 的基础功能

        self.register_modules(#通过 register_modules 方法注册所有组件，使其被管道统一管理   
            vae=vae,
            image_encoder=image_encoder,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
            pose_net=pose_net,
            face_encoder=face_encoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)#VAE 的缩放因子  缩放因子为 2^(下采样次数)（下采样次数等于 VAE 块数量减 1）。
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)#初始化图像处理器 VaeImageProcessor，用于将输入图像预处理为符合 VAE 输入要求的格式

        self.num_tokens = 4   #定义与姿态或人脸特征相关的令牌（token）数量，可能用于后续注意力机制中对特征的编码
        
        # 注释掉的代码：人脸分析工具初始化
        # self.app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        # self.app.prepare(ctx_id=0, det_size=(640, 640))
        # self.lora_rank = 128
        # self.set_ip_adapter()

    #从输入的人脸图像中提取并预处理人脸特征嵌入（face ID embeddings）
    def get_prepare_faceid(self, face_image):
        faceid_image = np.array(face_image)  #将输入的 face_image（通常是 PIL.Image.Image 格式）转换为 NumPy 数组
        faces = self.app.get(faceid_image)      #用self.app对图像进行人脸检测，返回检测到的人脸信息列表。每个元素包含人脸边界框、特征嵌入（embedding）等信息。
        if faces == []:
            faceid_embeds = torch.zeros_like(torch.empty((1, 512)))  #未检测到人脸（faces 为空列表），返回一个形状为 (1, 512) 的零张量作为默认特征
        else:#检测到人脸，取第一个人脸（faces[0]，通常是置信度最高或面积最大的人脸）的标准化特征嵌入（normed_embedding）
            faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
        return faceid_embeds
    # UNet 模型的注意力层配置自定义的注意力处理器
    def set_ip_adapter(self):
        unet = self.unet #获取 unet 模型
        attn_procs = {} #创建注意力处理字典
        for name in unet.attn_processors.keys():  #遍历 unet 模型的注意力层名称
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim  #attn1 是自注意力层 名称以 attn1.processor 结尾  attn2 是交叉注意力层：需使用 UNet 配置中的 cross_attention_dim
            
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
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=self.lora_rank,
                ).to(self.device, dtype=self.torch_dtype)
            else:
                attn_procs[name] = AnimationIDAttnProcessor(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, scale=1.0, rank=self.lora_rank,
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=self.torch_dtype)
        #调用 UNet 的 set_attn_processor 方法，将构建好的 attn_procs 字典应用到模型中，替换默认的注意力处理器。
        unet.set_attn_processor(attn_procs)
    #将输入图像转换为符合模型要求的特征嵌入（embeddings），用于指导视频生成过程。
    def _encode_image(
            self,
            image: PipelineImageInput,
            device: Union[str, torch.device],
            num_videos_per_prompt: int,
            do_classifier_free_guidance: bool):
        dtype = next(self.image_encoder.parameters()).dtype #图像编码器的参数中获取数据类型

        if not isinstance(image, torch.Tensor):#image 不是 PyTorch 张量
            image = self.image_processor.pil_to_numpy(image)#转换为NumPy 数组
            image = self.image_processor.numpy_to_pt(image)#转换为pytorch张量

            # We normalize the image before resizing to match with the original implementation.
            # Then we unnormalize it after resizing.
            image = image * 2.0 - 1.0  #归一化 将图像像素值从 [0, 1] 范围映射到 [-1, 1]
            image = _resize_with_antialiasing(image, (224, 224))  #抗锯齿 resize  使用抗锯齿算法将图像调整为 CLIP 模型要求的输入尺寸 (224, 224)
            image = (image + 1.0) / 2.0 #反归一化：将像素值从 [-1, 1] 映射回 [0, 1]（(image + 1.0) / 2.0），为后续 CLIP 特征提取做准备

            # Normalize the image with for CLIP input。 CLIP 的特征提取器（feature_extractor）对图像进行标准化处理
            image = self.feature_extractor(
                images=image,
                do_normalize=True,  #do_normalize=True：应用 CLIP 预训练时使用的均值和标准差归一化
                do_center_crop=False, #禁用裁剪
                do_resize=False,    #禁用缩放
                do_rescale=False,#禁用重缩放
                return_tensors="pt",#返回 PyTorch 张量
            ).pixel_values  #存储在 pixel_values 中

        image = image.to(device=device, dtype=dtype)
        image_embeddings = self.image_encoder(image).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)#通过 unsqueeze(1) 增加一个序列维度

        # duplicate image embeddings for each generation per prompt, using mps friendly method扩展特征以支持多视频生成
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)#repeat(1, num_videos_per_prompt, 1)：在序列维度上重复 num_videos_per_prompt 次
        image_embeddings = image_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)#view(...)：重塑张量，将 batch 维度扩展为 原始batch_size * num_videos_per_prompt，确保每个视频都有独立的特征嵌入。

        if do_classifier_free_guidance:
            negative_image_embeddings = torch.zeros_like(image_embeddings)
            # 拼接无条件和有条件嵌入，避免两次前向传播
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeddings = torch.cat([negative_image_embeddings, image_embeddings])
        #返回最终的图像特征嵌入，用于后续 UNet 模型的视频生成过程（作为条件输入控制视频内容）
        return image_embeddings
    
    #将输入图像通过 VAE（变分自编码器）编码为潜在空间表示（latents），并根据生成需求调整潜在特征的结构，以便用于后续的视频生成过程
    def _encode_vae_image(
            self,
            image: torch.Tensor,
            device: Union[str, torch.device],
            num_videos_per_prompt: int,
            do_classifier_free_guidance: bool,
    ):
        image = image.to(device=device, dtype=self.vae.dtype)
        image_latents = self.vae.encode(image).latent_dist.mode()
        image_latents = image_latents.repeat_interleave(num_videos_per_prompt, dim=0)
        if do_classifier_free_guidance:
            image_latents = torch.cat([torch.zeros_like(image_latents), image_latents])

        return image_latents
    #生成并处理用于视频生成的时间相关附加信息嵌入（additional time embeddings），这些信息包括帧率、运动强度和噪声增强强度，用于指导模型生成符合预期运动特征的视频
    def _get_add_time_ids(
            self,
            fps: int,
            motion_bucket_id: int,
            noise_aug_strength: float,
            dtype: torch.dtype,
            batch_size: int,
            num_videos_per_prompt: int,
            do_classifier_free_guidance: bool,
    ):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]
        #passed_add_embed_dim：计算当前参数列表对应的嵌入维度。addition_time_embed_dim 是 UNet 配置中每个时间参数的嵌入维度，乘以参数数量（这里是 3）得到总维度。
        passed_add_embed_dim = self.unet.config.addition_time_embed_dim * len(add_time_ids)
        #expected_add_embed_dim：获取 UNet 中处理附加时间嵌入的第一层线性层（add_embedding.linear_1）的输入特征维度，即模型期望的嵌入维度。
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:#若不一致，抛出错误提示
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, " \
                f"but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. " \
                f"Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)#将参数列表转换为 PyTorch 张量，数据类型与模型一致（dtype）
        add_time_ids = add_time_ids.repeat(batch_size * num_videos_per_prompt, 1)#repeat 方法扩展张量：按 batch_size * num_videos_per_prompt 倍数复制，确保每个输入样本（及每个样本生成的多个视频）都有对应的时间参数嵌入。

        if do_classifier_free_guidance:
            add_time_ids = torch.cat([add_time_ids, add_time_ids])

        return add_time_ids
    #将视频的潜在特征（latents）通过 VAE 解码器转换为实际的视频帧
    def decode_latents(
            self,
            latents: torch.Tensor,
            num_frames: int,
            decode_chunk_size: int = 8):
        latents = (1 / self.vae.config.scaling_factor) * latents
        forward_vae = self.vae._orig_mod.forward if is_compiled_module(self.vae) else self.vae.forward
        accepts_num_frames = "num_frames" in inspect.signature(forward_vae).parameters

        videos = []
        for video_latents in latents:
            frames = []
            for start in range(0, num_frames, decode_chunk_size):
                chunk = video_latents[start:start + decode_chunk_size]
                kwargs = {"num_frames": chunk.shape[0]} if accepts_num_frames else {}
                frames.append(self.vae.decode(chunk, **kwargs).sample.cpu())
            videos.append(torch.cat(frames, dim=0))
        return torch.stack(videos, dim=0).permute(0, 2, 1, 3, 4).float()

    #视频生成前的基础校验步骤，通过限制输入图像的类型和尺寸，避免后续模型处理（如特征提取、潜在空间转换）中出现维度不匹配或格式错误，确保管道能正常运行。
    def check_inputs(self, image, height, width):
        if (
                not isinstance(image, torch.Tensor) #PyTorch 张量
                and not isinstance(image, PIL.Image.Image) #单张 PIL 图像
                and not isinstance(image, list)#多张 PIL 图像组成的列表（用于批量生成）
        ):
            raise ValueError(
                "`image` has to be of type `torch.FloatTensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is"
                f" {type(image)}"
            )
            #查输入图像的高度（height）和宽度（width）是否为 8 的倍数
        if height < 1 or width < 1 or height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
    #为视频生成管道准备初始的潜在变量 是扩散模型（Diffusion Model）生成过程的起点。潜在变量是模型在潜在空间中逐步去噪以生成最终视频的初始输入
    def prepare_latents(
            self,
            batch_size: int,
            num_frames: int,
            num_channels_latents: int,
            height: int,
            width: int,
            dtype: torch.dtype,
            device: Union[str, torch.device],
            generator: torch.Generator,
            latents: Optional[torch.Tensor] = None,
    ):
        shape = (#算潜在变量的维度形状，包含 5 个维度
            batch_size,
            num_frames,
            num_channels_latents // 2,   #潜在空间的通道数（通常是 VAE 编码后的通道数的一半，因模型设计而定）。
            height // self.vae_scale_factor, #潜在空间的高度（原始图像高度除以 VAE 的缩放因子，因 VAE 下采样导致）。
            width // self.vae_scale_factor,
        )
        #校验生成器与批次大小的一致性  传入的 generator 是列表  列表长度必须与 batch_size 相等，否则抛出错误。
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            if tuple(latents.shape) != shape:
                raise ValueError(f"Initial latents must have shape {shape}, got {tuple(latents.shape)}.")
            latents = latents.to(device=device, dtype=dtype)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents
    
    #代码定义了与 “引导尺度（guidance scale）” 相关的属性和说明，是扩散模型中分类器自由引导（Classifier-Free Guidance, CFG）机制的核心参数
    @property   #@property 装饰器将 guidance_scale 定义为类的属性（而非方法），允许通过 self.guidance_scale 直接访问，而无需调用函数（如 self.guidance_scale()）。
    def guidance_scale(self):
        return self._guidance_scale#返回实例变量 _guidance_scale 的值，该变量存储了当前设置的引导尺度参数。

    @property
    def do_classifier_free_guidance(self):
        return getattr(self, "_do_classifier_free_guidance", False)

    #num_timesteps 属性，用于获取扩散模型推理过程中的时间步数（去噪步骤数）
    @property
    def num_timesteps(self):
        return self._num_timesteps  #返回实例变量 _num_timesteps 的值，该变量存储了扩散过程中使用的时间步数。

    #用于为扩散模型调度器（scheduler）的去噪步骤准备额外的参数，确保参数与当前使用的调度器兼容
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        #由于不同调度器（如 Euler、DDIM 等）的 step 方法参数签名不同，需要动态准备适配的参数。
        #特别说明 eta 参数仅用于 DDIM 调度器（参考论文 DDIM），其他调度器会忽略该参数，且其取值需在 [0, 1] 之间。
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    #是 InferenceAnimationPipeline 类的核心生成方法 __call__，用于从输入图像和姿态信息生成动画视频
    @torch.no_grad()   #禁用 PyTorch 的自动求导，减少内存占用并加速推理（因为生成过程无需反向传播）
    def __call__(    #定义 __call__ 方法，使类实例可像函数一样被调用（如 pipe(image, ...)），是触发视频生成的入口。
            self,
            image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor],
            image_pose: Union[torch.FloatTensor],
            height: int = 576,
            width: int = 1024,
            num_frames: Optional[int] = None,
            tile_size: Optional[int] = 16,    #分块处理的尺寸（可选，默认 16），用于大尺寸图像的分块生成，平衡内存与效率。
            tile_overlap: Optional[int] = 4,  #分块间的重叠区域大小（可选，默认 4），减少分块拼接的边缘 artifacts
            num_inference_steps: int = 25,    #去噪步数（默认 25），步数越多生成质量越高，但速度越慢
            min_guidance_scale: float = 1.0,   #最小 / 最大引导尺度（默认 1.0/3.0），用于分类器自由引导（CFG）。引导尺度从第一帧的最小值线性变化到最后一帧的最大值，控制生成内容与输入条件的匹配程度（值越大越贴近条件）。
            max_guidance_scale: float = 3.0, 
            fps: int = 7, 
            motion_bucket_id: int = 127,  #运动桶 ID（默认 127），控制视频运动幅度，值越大运动越剧烈
            noise_aug_strength: float = 0.02,  #添加到初始图像的噪声强度（默认 0.02），值越大视频与原图差异越大，运动更丰富。
            image_only_indicator: bool = False,
            decode_chunk_size: Optional[int] = None,   #一次解码的帧数（默认与总帧数相同），平衡时间一致性与内存占用（值越小内存消耗越低，但可能影响帧间连贯性）
            num_videos_per_prompt: Optional[int] = 1,   #每个输入图像生成的视频数量（默认 1 
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,   #PyTorch 随机数生成器（可选），用于控制生成的随机性，使结果可复现
            latents: Optional[torch.FloatTensor] = None,   #预生成的噪声潜变量（可选），若未提供则自动采样，可用于微调生成结果
            validation_image_id_ante_embedding=None,  #验证图像的 ID 嵌入（可选），可能用于人脸等特定对象的一致性校验。
            output_type: Optional[str] = "pil",       #输出格式（默认 "pil"），可选 PIL 图像列表或 NumPy 数组
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,   #每步去噪结束时的回调函数（可选），用于监控或自定义处理中间结果。
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],                #回调函数需要的张量输入列表（默认仅 "latents"）
            # [IPI Mod] 新增参数
            dwpose_raw: Optional[torch.Tensor] = None,
            clip_image_embeds: Optional[torch.Tensor] = None,
            is_statue: bool = False,    # 🚀 新增雕像模式开关（默认 False）
            return_dict: bool = True,   #否返回结构化输出对象（默认 True），否则返回元组。
            pose_scale: float = 3.0,
            ipi_scale: Optional[float] = None,
            ipi_temporal_v2: bool = False,
            repeat_initial_noise: bool = True,
            timeline_start: int = 0,
            timeline_length: Optional[int] = None,
            reference_latent_scale: float = 1.0,
            reference_clip_scale: float = 1.0,
    ):
        r"""
        The call function to the pipeline for generation.
        这是管道（pipeline）用于生成（动画 / 视频）的调用函数。即通过调用该方法，可触发模型生成视频帧的流程。

        Args:
            image (`PIL.Image.Image` or `List[PIL.Image.Image]` or `torch.FloatTensor`):
                Image or images to guide image generation. If you provide a tensor, it needs to be compatible with
                [`CLIPImageProcessor`](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/blob/main/
                feature_extractor/preprocessor_config.json).
            参数 image：用于引导生成的输入图像。
                类型：可以是单张 PIL 图像、PIL 图像列表，或 PyTorch 张量。
                说明：如果输入是张量，需与 CLIPImageProcessor 的格式兼容（遵循链接中的预处理配置）
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_frames (`int`, *optional*):
                The number of video frames to generate. Defaults to 14 for `stable-video-diffusion-img2vid`
                and to 25 for `stable-video-diffusion-img2vid-xt`
            num_inference_steps (`int`, *optional*, defaults to 25):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter is modulated by `strength`.
            min_guidance_scale (`float`, *optional*, defaults to 1.0):
                The minimum guidance scale. Used for the classifier free guidance with first frame.
            max_guidance_scale (`float`, *optional*, defaults to 3.0):
                The maximum guidance scale. Used for the classifier free guidance with last frame.
            fps (`int`, *optional*, defaults to 7):
                Frames per second.The rate at which the generated images shall be exported to a video after generation.
                Note that Stable Diffusion Video's UNet was micro-conditioned on fps-1 during training.
            motion_bucket_id (`int`, *optional*, defaults to 127):
                The motion bucket ID. Used as conditioning for the generation.
                The higher the number the more motion will be in the video.
            noise_aug_strength (`float`, *optional*, defaults to 0.02):
                The amount of noise added to the init image,
                the higher it is the less the video will look like the init image. Increase it for more motion.
            image_only_indicator (`bool`, *optional*, defaults to False):
                Whether to treat the inputs as batch of images instead of videos.
            decode_chunk_size (`int`, *optional*):
                The number of frames to decode at a time.The higher the chunk size, the higher the temporal consistency
                between frames, but also the higher the memory consumption.
                By default, the decoder will decode all frames at once for maximal quality.
                Reduce `decode_chunk_size` to reduce memory usage.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            device:
                On which device the pipeline runs on.

        Returns:
            [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`,
                [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list of list with the generated frames.

            repeat_initial_noise:
                Keep the legacy periodic noise sequence. Set to False to sample full-length noise
                independently of the denoising window size.

        Examples:

        ```py
        from diffusers import StableVideoDiffusionPipeline
        from diffusers.utils import load_image, export_to_video

        pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16")
        pipe.to("cuda")

        image = load_image(
        "https://lh3.googleusercontent.com/y-iFOHfLTwkuQSUegpwDdgKmOjRSTvPxat63dQLB25xkTs4lhIbRUFeNBWZzYf370g=s1200")
        image = image.resize((1024, 576))

        frames = pipe(image, num_frames=25, decode_chunk_size=8).frames[0]
        export_to_video(frames, "generated.mp4", fps=7)
        - 说明：展示了使用该管道生成视频的示例代码。
            1. 导入必要的类和工具函数。
            2. 加载预训练模型（`stable-video-diffusion-img2vid-xt`），指定数据类型为 float16 以节省显存，并将模型移至 GPU。
            3. 加载输入图像并调整尺寸为 (1024, 576)。
            4. 调用管道生成 25 帧，每次解码 8 帧，获取生成的帧序列。
            5. 使用 `export_to_video` 将帧序列合成为 MP4 视频（帧率 7）。


            ### 总结
            该文档字符串详细描述了 `__call__` 方法的功能和参数，帮助用户理解如何配置输入以生成符合预期的视频，
            同时提供简洁的示例便于快速上手。核心参数围绕生成视频的质量（去噪步骤、引导尺度）、运动特性（运动桶 ID、噪声增强）、性能（解码批次大小、设备）等方面展开。       
        ```
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor  #self.unet.config.sample_size（UNet 模型配置中的样本尺寸）乘以 self.vae_scale_factor（VAE 的缩放因子）作为默认高度
        width = width or self.unet.config.sample_size * self.vae_scale_factor  #同上

        num_frames = len(image_pose) if num_frames is None else num_frames
        if num_frames > len(image_pose):
            raise ValueError(f"Requested {num_frames} frames, only {len(image_pose)} pose frames supplied.")
        if (dwpose_raw is None) != (clip_image_embeds is None):
            raise ValueError("IPI requires both pose coordinates and driver image embeddings.")
        tile_size = num_frames if tile_size is None else tile_size
        tile_overlap = 0 if tile_overlap is None else tile_overlap
        indices = _frame_windows(num_frames, tile_size, tile_overlap)
        decode_chunk_size = num_frames if decode_chunk_size is None else decode_chunk_size
        if decode_chunk_size < 1 or num_inference_steps < 1:
            raise ValueError("decode_chunk_size and num_inference_steps must be positive.")
        if num_videos_per_prompt is None or num_videos_per_prompt < 1:
            raise ValueError("num_videos_per_prompt must be positive.")

        # 1. Check inputs. Raise error if not correct  对输入的图像、高度和宽度参数进行合法性校验，确保符合模型运行的基本要求，若不合法则抛出错误。
        #逻辑：调用当前类（InferenceAnimationPipeline）的 check_inputs 方法，传入 image（输入图像）、height（生成视频帧的高度）、width（生成视频帧的宽度）三个参数，由该方法内部实现具体校验逻辑
        self.check_inputs(image, height, width)

        # 2. Define call parameters据输入图像的类型确定批量大小（batch_size），即一次处理的图像数量
        if isinstance(image, PIL.Image.Image):   #若输入 image 是单张 PIL 图像（PIL.Image.Image），则批量大小为 1
            batch_size = 1
        elif isinstance(image, list):#若输入 image 是图像列表（list），则批量大小为列表长度（即列表中图像的数量）
            batch_size = len(image)
        else:#若输入 image 是张量（torch.Tensor），则批量大小为张量的第 0 维尺寸（image.shape[0]，通常表示样本数量）
            batch_size = image.shape[0]
        device = self._execution_device
        # Keep the CFG decision shared by conditioning, denoising and the scheduler.
        effective_batch_size = batch_size * num_videos_per_prompt
        if not all(math.isfinite(scale) and scale >= 1 for scale in (min_guidance_scale, max_guidance_scale)):
            raise ValueError("Guidance scales must be finite and at least 1.")
        do_classifier_free_guidance = max(min_guidance_scale, max_guidance_scale) > 1.0
        self._do_classifier_free_guidance = do_classifier_free_guidance
        uncond_slice = slice(0, effective_batch_size)
        cond_slice = slice(effective_batch_size, None) if do_classifier_free_guidance else slice(None)
        self._guidance_scale = max_guidance_scale

        # 3. Encode input image对输入图像进行编码，生成用于引导视频生成的特征嵌入（embeddings）
        #调用当前类的 _encode_image 方法，传入输入图像 image、执行设备 device、每个提示生成的视频数量 num_videos_per_prompt 以及是否启用无分类器引导的标志 do_classifier_free_guidance，
        # 返回编码后的图像特征 image_embeddings
        image_embeddings = self._encode_image(image, device, num_videos_per_prompt, do_classifier_free_guidance)
        # self.image_encoder.cpu()

        # NOTE: Stable Diffusion Video was conditioned on fps - 1, which
        # is why it is reduced here.调整帧率参数以匹配模型训练时的条件设置
        fps = fps - 1#：将输入的帧率 fps 减 1，因为 Stable Diffusion Video 模型在训练时使用的是 fps - 1 作为条件输入

        # 4. Encode input image using VAE使用 VAE（变分自编码器）对输入图像进行编码，生成潜在空间表示，并融合人脸特征等条件信息，为后续扩散过程做准备。

        # print(image_embeddings.size()) # [2, 1, 1024]  处理验证图像的人脸 ID 嵌入（预先计算的人脸特征），转换为模型可处理的张量格式。
        cond_image_embeddings = image_embeddings[cond_slice]
        if not all(math.isfinite(s) and s >= 0 for s in (reference_latent_scale, reference_clip_scale)):
            raise ValueError("Reference scales must be finite and nonnegative.")
        if is_statue or validation_image_id_ante_embedding is None:
            faceid_latents = cond_image_embeddings.new_zeros(
                (effective_batch_size, self.num_tokens, cond_image_embeddings.shape[-1])
            )
        else:
            faceid = torch.as_tensor(
                validation_image_id_ante_embedding, device=device, dtype=image_embeddings.dtype
            )
            if faceid.ndim == 1:
                faceid = faceid.unsqueeze(0)
            if faceid.ndim != 2 or faceid.shape[0] not in (1, batch_size):
                raise ValueError("Provide one face embedding or one per reference image.")
            faceid = faceid.expand(batch_size, -1).repeat_interleave(num_videos_per_prompt, dim=0)
            faceid_latents = self.face_encoder(faceid, cond_image_embeddings)

        # Scale the global CLIP token after face fusion so identity tokens remain independently controlled.
        if reference_clip_scale != 1:
            cond_image_embeddings = cond_image_embeddings * reference_clip_scale
        cond_image_embeddings = torch.cat([cond_image_embeddings, faceid_latents], dim=1)
        if do_classifier_free_guidance:
            uncond_image_embeddings = torch.cat(
                [image_embeddings[uncond_slice], torch.zeros_like(faceid_latents)], dim=1
            )
            image_embeddings = torch.cat([uncond_image_embeddings, cond_image_embeddings], dim=0)
        else:
            image_embeddings = cond_image_embeddings

        #功能：预处理输入图像并添加噪声，模拟扩散过程的初始状态 作用：为 VAE 编码提供带噪声的输入，符合扩散模型从噪声中生成数据的逻辑
        image = self.image_processor.preprocess(image, height=height, width=width).to(device) #self.image_processor.preprocess 调整图像尺寸至 height 和 width，并转换为张量。

        noise = randn_tensor(image.shape, generator=generator, device=device, dtype=image.dtype)#randn_tensor 生成与图像形状相同的随机噪声（符合高斯分布）
        image = image + noise_aug_strength * noise #按 noise_aug_strength 比例将噪声添加到图像中，增强生成多样性（噪声越强，视频与原图差异越大）
        vae_dtype = self.vae.dtype
        needs_upcasting = vae_dtype in (torch.float16, torch.bfloat16) and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)
        try:
            image_latents = self._encode_vae_image(
                image,
                device=device,
                num_videos_per_prompt=num_videos_per_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
            ).to(image_embeddings.dtype)
        finally:
            if needs_upcasting:
                self.vae.to(dtype=vae_dtype)

        # Repeat the image latents for each frame so we can concatenate them with the noise
        # image_latents [batch, channels, height, width] ->[batch, num_frames, channels, height, width]
        #功能：扩展图像潜在变量的维度，使其适配视频的多帧结构·  作用：使单张图像的潜在变量扩展为与视频帧数匹配的结构，便于后续与扩散过程的噪声帧拼接
        if reference_latent_scale != 1:
            image_latents = image_latents * reference_latent_scale
        image_latents = image_latents.unsqueeze(1).expand(-1, num_frames, -1, -1, -1)

        # 5. Get Added Time IDs生成并处理用于时间嵌入的附加参数（added_time_ids），为视频生成提供时间相关的条件信息
        added_time_ids = self._get_add_time_ids(
            fps,  #帧率（已按模型要求调整为 fps - 1）
            motion_bucket_id, #控制视频运动幅度的参数（值越高，运动越剧烈）
            noise_aug_strength,
            image_embeddings.dtype,
            batch_size,
            num_videos_per_prompt,  #每个提示生成的视频数量
            do_classifier_free_guidance,
        )
        added_time_ids = added_time_ids.to(device)

        # 4. Prepare timesteps 获取扩散过程中使用的时间步（timesteps），并确认实际的推理步数，为后续的图像 / 视频生成扩散过程做准备。
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, None)

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        timeline_length = num_frames if timeline_length is None else timeline_length
        if timeline_start < 0 or timeline_start + num_frames > timeline_length:
            raise ValueError("Requested clip falls outside its noise and CFG timeline.")
        noise_frames = tile_size if repeat_initial_noise else timeline_length
        if repeat_initial_noise and latents is not None:
            if latents.ndim != 5 or latents.shape[1] < 1:
                raise ValueError("Initial latents must be a 5-D tensor with at least one frame.")
            noise_frames = latents.shape[1]
        latents = self.prepare_latents(  # 初始化潜在变量
            batch_size * num_videos_per_prompt,  # 总批量大小（批次×每个提示生成的视频数）
            noise_frames,
            num_channels_latents,  # 潜在变量的通道数（与UNet输入通道一致）
            height,  # 生成视频的高度
            width,  # 生成视频的宽度
            image_embeddings.dtype,  # 数据类型（与图像嵌入保持一致）
            device,  # 计算设备（如GPU）
            generator,  # 随机数生成器（用于潜在变量的随机初始化）
            latents,  # 预生成的潜在变量（可选，若未提供则自动生成）
        )
        # 扩展潜在变量的时间维度以匹配目标帧数，超出部分截断
        if repeat_initial_noise:
            positions = torch.arange(timeline_start, timeline_start + num_frames, device=device) % noise_frames
            latents = latents.index_select(1, positions)
        else:
            latents = latents[:, timeline_start:timeline_start + num_frames]

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline准备扩散过程中调度器（scheduler）步骤所需的额外参数。
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, 0.0)  #prepare_extra_step_kwargs 方法生成额外参数字典 extra_step_kwargs，用于后续的调度器去噪步骤

        # 7. Prepare guidance scale
        guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, timeline_length)
        guidance_scale = guidance_scale[timeline_start:timeline_start + num_frames].unsqueeze(0)
        guidance_scale = guidance_scale.to(device, latents.dtype)  # 转换设备和数据类型 
        guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)  # 按批量复制 则复制后形状从 (1, num_frames) 变为 (2, num_frames)，每个样本都有独立的引导尺度序列
        guidance_scale = _append_dims(guidance_scale, latents.ndim)  # 扩展维度以匹配潜在变量 潜在变量的形状通常为 (batch_size, num_frames, channels, height, width)（5 维），而当前引导尺度为 2 维（(batch_size, num_frames)）。扩展后会在末尾添加空维度（如变为 (batch_size, num_frames, 1, 1, 1)

        self._guidance_scale = guidance_scale  # 存储为实例变量

        # 8. Denoising loop

        self._num_timesteps = len(timesteps)
        
        window_weights = []
        noise_pred_cnt = image_latents.new_zeros((num_frames,))
        for idx in indices:
            chunk_len = len(idx)
            weight = (torch.arange(chunk_len, device=device) + 0.5) * 2.0 / chunk_len
            weight = torch.minimum(weight, 2 - weight).clamp_min(0.25)
            noise_pred_cnt[idx] += weight
            window_weights.append(weight)
        if not (torch.isfinite(noise_pred_cnt) & (noise_pred_cnt > 0)).all():
            raise RuntimeError("Denoising windows must have positive finite coverage for every frame.")

        # 处理姿势图像（pose）数据
        pose_pil_image_list = []
        for pose in image_pose[:num_frames]:
            pose = torch.from_numpy(np.array(pose)).float()  # 转换为Tensor并转为浮点型
            if pose.shape != (height, width, 3):
                raise ValueError(f"Expected HWC RGB pose on {width}x{height} canvas, got {pose.shape}.")
            pose = pose / 127.5 - 1  # 归一化到[-1, 1]范围（与模型输入要求一致）
            pose_pil_image_list.append(pose)
        pose_pil_image_list = torch.stack(pose_pil_image_list, dim=0)  # 堆叠为(batch_size, height, width, channels)
        pose_pil_image_list = rearrange(pose_pil_image_list, "f h w c -> f c h w")  # 调整维度为(frames, channels, height, width)


        # print(indices)  # [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]
        # print(pose_pil_image_list.size())  # [16, 3, 512, 512]

        self.pose_net.to(device)
        self.unet.to(device)

        if torch.device(device).type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.empty_cache()

        # 使用进度条跟踪整个去噪过程的进度，总步数为时间步数量 × 分块数量
        with self.progress_bar(total=len(timesteps) * len(indices)) as progress_bar:

            ipi_features_all = None
            uncond_ipi_features_all = None
            window_ipi = None
            if dwpose_raw is not None and clip_image_embeds is not None:
                conditions = []
                for name, condition in (("dwpose_raw", dwpose_raw), ("clip_image_embeds", clip_image_embeds)):
                    if condition.ndim != 3 or condition.shape[1] < num_frames:
                        raise ValueError(f"{name} must have shape [batch, frames, features] with at least {num_frames} frames.")
                    if condition.shape[0] == batch_size and num_videos_per_prompt > 1:
                        condition = condition.repeat_interleave(num_videos_per_prompt, dim=0)
                    elif condition.shape[0] != effective_batch_size:
                        raise ValueError(f"{name} batch must match the references or the generated videos.")
                    conditions.append(condition[:, :num_frames].to(device=device, dtype=image_embeddings.dtype))
                dwpose_raw, clip_image_embeds = conditions

                if ipi_temporal_v2:
                    window_ipi = []
                    for idx in indices:
                        pose_window = dwpose_raw[:, idx]
                        clip_window = clip_image_embeds[:, idx]
                        previous = None
                        if self.unet.ipi_adapter.joint_motion is not None and idx[0] > 0:
                            before = slice(idx[0] - 1, idx[0])
                            previous = (dwpose_raw[:, before], clip_image_embeds[:, before])
                        conditional = self.unet.ipi_adapter(
                            pose_window, clip_window, ipi_temporal_v2=True, previous=previous
                        )
                        unconditional = None
                        if do_classifier_free_guidance:
                            unconditional = self.unet.ipi_adapter(
                                torch.zeros_like(pose_window), torch.zeros_like(clip_window),
                                ipi_temporal_v2=True, skip_joint_motion=True
                            )
                        window_ipi.append((conditional, unconditional))
                else:
                    ipi_list = []
                    uncond_ipi_list = []
                    for start in range(0, num_frames, 14):
                        pose_window = dwpose_raw[:, start:start + 14]
                        clip_window = clip_image_embeds[:, start:start + 14]
                        conditional = self.unet.ipi_adapter(pose_window, clip_window, frame_offset=0)
                        token_shape = (effective_batch_size, pose_window.shape[1], *conditional.shape[1:])
                        ipi_list.append(conditional.reshape(token_shape))
                        if do_classifier_free_guidance:
                            unconditional = self.unet.ipi_adapter(
                                torch.zeros_like(pose_window), torch.zeros_like(clip_window), frame_offset=0
                            )
                            uncond_ipi_list.append(unconditional.reshape(token_shape))

                    ipi_features_all = torch.cat(ipi_list, dim=1)
                    if do_classifier_free_guidance:
                        uncond_ipi_features_all = torch.cat(uncond_ipi_list, dim=1)
                    print(f"✅ [IPI 预计算完成] shape={ipi_features_all.shape}")

            single_window_pose = None
            for i, t in enumerate(timesteps):
                # 如果启用分类器自由引导（CFG），则将潜在变量复制一份用于无条件分支（无引导）
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                # 按调度器要求缩放潜在变量（适配当前时间步的噪声水平）
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # 将图像潜在变量（参考图像的编码）与当前潜在变量在通道维度拼接，作为UNet的输入
                latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)

                # 初始化噪声预测结果和计数张量（用于分块处理时的加权平均）
                noise_pred = torch.zeros_like(image_latents)
                for window_index, (idx, weight) in enumerate(zip(indices, window_weights)):
                    # ✅ 切片预计算好的 IPI 特征（PE 连续无断裂）
                    if window_ipi is not None:
                        cur_ipi_chunk, zero_ipi_chunk = window_ipi[window_index]
                    elif ipi_features_all is not None:
                        cur_ipi_chunk = ipi_features_all[:, idx].flatten(0, 1)
                        zero_ipi_chunk = None
                        if do_classifier_free_guidance:
                            zero_ipi_chunk = uncond_ipi_features_all[:, idx].flatten(0, 1)
                    else:
                        cur_ipi_chunk = None
                        zero_ipi_chunk = None

                    if single_window_pose is None:
                        pose_latents = self.pose_net(pose_pil_image_list[idx].to(device=device, dtype=latent_model_input.dtype))
                        if effective_batch_size > 1:
                            pose_latents = pose_latents.repeat(effective_batch_size, 1, 1, 1)
                        if len(indices) == 1:
                            single_window_pose = pose_latents
                    else:
                        pose_latents = single_window_pose

                    if do_classifier_free_guidance:
                        uncond_noise = self.unet(
                            latent_model_input[uncond_slice, idx], t,
                            encoder_hidden_states=image_embeddings[uncond_slice],
                            added_time_ids=added_time_ids[uncond_slice],
                            pose_latents=torch.zeros_like(pose_latents),
                            precomputed_ipi_features=zero_ipi_chunk,
                            image_only_indicator=image_only_indicator,
                            global_frame_indices=idx,
                            pose_scale=pose_scale,
                            ipi_scale=ipi_scale,
                            ipi_temporal_v2=ipi_temporal_v2,
                            return_dict=False,
                        )[0]
                        noise_pred[uncond_slice, idx] += uncond_noise * weight[:, None, None, None]

                    cond_noise = self.unet(
                        latent_model_input[cond_slice, idx], t,
                        encoder_hidden_states=image_embeddings[cond_slice],
                        added_time_ids=added_time_ids[cond_slice],
                        pose_latents=pose_latents,
                        precomputed_ipi_features=cur_ipi_chunk,
                        image_only_indicator=image_only_indicator,
                        global_frame_indices=idx,
                        pose_scale=pose_scale,
                        ipi_scale=ipi_scale,
                        ipi_temporal_v2=ipi_temporal_v2,
                        return_dict=False,
                    )[0]
                    noise_pred[cond_slice, idx] += cond_noise * weight[:, None, None, None]

                    progress_bar.update()

                # 对分块重叠区域的噪声预测结果进行归一化（除以总权重）
                noise_pred.div_(noise_pred_cnt[:, None, None, None])

                # 应用分类器自由引导：合并无条件和有条件的噪声预测
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)  # 拆分无条件和有条件分支
                    # 公式：噪声预测 = 无条件预测 + 引导尺度 × (有条件预测 - 无条件预测)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # 使用调度器更新潜在变量（从x_t去噪到x_{t-1}）
                latents = self.scheduler.step(noise_pred, t, latents,** extra_step_kwargs, return_dict=False)[0]

                # 执行步骤结束回调（如记录中间结果、可视化等）
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]  # 收集回调所需的张量（如latents）
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    # 从回调结果中更新潜在变量（若回调修改了latents）
                    latents = callback_outputs.pop("latents", latents)

        # 可选：将部分模型移回CPU释放GPU内存（此处注释掉，可能用于调试或后续优化）
        # self.pose_net.cpu()
        # self.unet.cpu()
        # self.face_encoder.cpu()

        # 判断输出类型是否为"latent"（潜在变量），如果不是则进行解码
        if not output_type == "latent":
            # 将VAE解码器移动到当前计算设备（如GPU）
            self.vae.decoder.to(device)
            # 调用解码方法，将潜在变量转换为视频帧
            # latents: 去噪后的潜在变量，num_frames: 视频帧数，decode_chunk_size: 分块解码大小（避免显存溢出）
            frames = self.decode_latents(latents, num_frames, decode_chunk_size)
            # 打印调试信息（已注释）：查看解码后帧和原始潜在变量的形状
            # print(frames.size()) # [1, 3, 16, 512, 512] 即 [批次, 通道, 帧数, 高度, 宽度]
            # print(latents.size()) # [1, 16, 4, 64, 64] 即 [批次, 帧数, 潜在通道, 潜在高度, 潜在宽度]
            
            # 将解码得到的视频帧张量转换为指定输出类型（如PIL图像、numpy数组或张量）
            # self.image_processor用于后处理（如归一化、格式转换）
            frames = tensor2vid(frames, self.image_processor, output_type=output_type)
            # 打印调试信息（已注释）：查看转换后单批次帧的形状
            # print(frames[0].size()) # [16, 3, 512, 512] 即 [帧数, 通道, 高度, 宽度]
        else:
            # 如果输出类型为"latent"，则直接返回潜在变量，不进行解码
            frames = latents

        # 释放模型钩子资源（如梯度检查点、缓存等），优化内存使用
        self.maybe_free_model_hooks()

        # 根据return_dict参数决定返回格式
        if not return_dict:
            # 返回原始帧数据（列表、数组或张量）
            return frames

        # 返回封装后的管道输出对象，包含生成的帧
        return InferenceAnimationPipelineOutput(frames=frames)
