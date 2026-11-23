
import torch
import torch.nn as nn
# 1. 确保导入了 Resampler
from animation.modules.IPI_module import DwposeEncoder, ImageProjModel, Resampler
from animation.modules.ipi_joint_motion import JointMotionEncoder

class MotionAdapter(nn.Module):
    def __init__(self, 
                 cross_attention_dim=1024, 
                 clip_dim=1024, 
                 num_tokens=8):
        super().__init__()
        
        # 保存 num_tokens 方便 forward 使用
        self.num_tokens = num_tokens
        
        # 1. Pose Encoder 保持不变
        self.pose_encoder = DwposeEncoder(
            d_model=cross_attention_dim,
            input_dim=266,
            num_tokens=num_tokens
        )
        
        # 2. 升级：将 ImageProjModel 替换为 Resampler
        # depth=4 对应你草图中的 x4 结构
        self.clip_proj = Resampler(
            dim=cross_attention_dim,
            depth=4,                  # 4层 Transformer 结构
            num_queries=num_tokens,   # 输出 8 个 token
            embedding_dim=clip_dim,
            output_dim=cross_attention_dim,
        )
        
        # 3. 融合层
        self.fusion_norm = nn.LayerNorm(cross_attention_dim)
        self.motion_scale_raw = nn.Parameter(torch.zeros(1))
        self.joint_motion = None
        self.clip_dim = clip_dim

    def enable_joint_motion(self, widths):
        if self.joint_motion is None:
            self.joint_motion = JointMotionEncoder(self.clip_dim, widths).to(self.motion_scale_raw)

    def train(self, mode=True):
        super().train(mode)
        if self.joint_motion is not None:
            # The parent pilot is frozen during joint-only training, including its dropout.
            self.pose_encoder.eval()
            self.clip_proj.eval()
        return self

    def forward(self, dwpose_raw, clip_embeds, frame_offset=0, ipi_temporal_v2=False,
                previous=None, skip_joint_motion=False):
        # 处理姿态特征
        pose_feat = self.pose_encoder(
            dwpose_raw, frame_offset=frame_offset, ipi_temporal_v2=ipi_temporal_v2
        )
        
        clip_embeds_flat = clip_embeds.reshape(-1, 1, clip_embeds.shape[-1])
        if ipi_temporal_v2:
            # Let the current frame's pose guide extraction from its driving image.
            init_latents = pose_feat
        else:
            init_latents = clip_embeds.new_zeros(pose_feat.shape)
        clip_feat = self.clip_proj(clip_embeds_flat, init_latents)

        # 4. 级联融合 (8+8=16)
        motion_context = torch.cat([pose_feat, clip_feat], dim=1)
        # ✅ [修复] 将 sigmoid 范围从 (0, 2.0) 扩大为 (0, 4.0)，给可学习参数更大动态范围
        # ✅ [修复] 删掉末尾硬编码 * 2.2，避免与外部 ipi_scale 叠加导致信号不可控
        scale = 1.2 * torch.sigmoid(self.motion_scale_raw)
        motion_context = self.fusion_norm(motion_context) * scale
        
        
        if ipi_temporal_v2 and self.joint_motion is not None and not skip_joint_motion:
            return self.joint_motion(dwpose_raw, clip_embeds, motion_context, previous)
        return motion_context
