import math
import torch
import torch.nn as nn
from .transformer import (
    TransformerEncoder,
    TransformerEncoderLayer,
    PositionalEncoding,
    TransformerDecoderLayer,
    TransformerDecoder,
)

class ImageProjModel(nn.Module):
    """Projection Model"""
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()        
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = nn.LayerNorm(cross_attention_dim)
        
    def forward(self, image_embeds):
        #embeds = image_embeds
        embeds = image_embeds.type(list(self.proj.parameters())[0].dtype)
        clip_extra_context_tokens = self.proj(embeds).reshape(-1, self.clip_extra_context_tokens, self.cross_attention_dim)
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


# FFN
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )
    
    
def reshape_tensor(x, heads):
    bs, length, width = x.shape
    #(bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)


    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)
        
        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        
        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1) # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v
        
        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)



class Resampler(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=1024,
        output_dim=1024,
        ff_mult=4,
        our_weight = 1,
        video_length=None, # using frame-wise version or not
    ):
        super().__init__()
        ## queries for a single frame / image
        self.num_queries = num_queries 
        self.video_length = video_length
        self.our_weight = our_weight

        ## <num_queries> queries for each frame
        if video_length is not None: 
            num_queries = num_queries * video_length

        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)


        # print('self.latents.shape:', self.latents.shape)

        self.proj_in = nn.Linear(embedding_dim, dim)
        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x, latents):
        # import pdb; pdb.set_trace()
        latents = self.latents.repeat(x.size(0), 1, 1) * self.our_weight+ latents ## B (T L) C
        x = self.proj_in(x)
        
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
            
        latents = self.proj_out(latents)
        latents = self.norm_out(latents) # B L C or B (T L) C

        return latents

def _reset_parameters(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

class DwposeEncoder(nn.Module):
    def __init__(
        self,
        d_model=1024,
        nhead=8,
        num_encoder_layers=3,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        pos_embed_len=33,
        input_dim=266,
        num_tokens=4,

    ):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        _reset_parameters(self.encoder)

        self.pos_embed = PositionalEncoding(d_model, pos_embed_len)

        increase_embed_dim = [nn.Linear(input_dim, d_model//2), nn.Linear(d_model//2, d_model)]
        self.increase_embed_dim = nn.Sequential(*increase_embed_dim)

        self.pre_image_condition_p = nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.SiLU(),
                    nn.Linear(d_model, d_model*num_tokens))

        nn.init.xavier_uniform_(self.pre_image_condition_p[-1].weight, gain=0.02)
        nn.init.zeros_(self.pre_image_condition_p[-1].bias)

        self.d_model = d_model
    def _extended_position_encoding(self, x, frame_offset):
        """Compute sinusoidal positions without the legacy 33-frame table limit."""
        positions = torch.arange(frame_offset, frame_offset + x.size(0), device=x.device, dtype=torch.float32)
        channels = torch.arange(0, self.d_model, 2, device=x.device, dtype=torch.float32)
        angles = positions[:, None] * 10000.0 ** (-channels / self.d_model)
        pos = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1)[:, :self.d_model]
        return pos.unsqueeze(1).to(dtype=x.dtype)

    def forward(self, x, pad_mask=None, frame_offset=0, ipi_temporal_v2=False):
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # 1. 维度变换
        x = self.increase_embed_dim(x)
        x = x.permute(1, 0, 2) # (L, B, C) -> 变成 (14, 1, 1024)

        if ipi_temporal_v2:
            pos = self._extended_position_encoding(x, frame_offset)
        else:
            # 🚀 2. 终极防爆的位置编码提取 (物理免疫所有维度爆炸)
            # 必须获取足够长（涵盖 frame_offset）的位置编码，而不是只能获取 seq_len 长度！
            max_required_len = frame_offset + seq_len
            # 如果超出预设的最大长度，pos_embed 会自动截断到最大值，后续会有 pad_tensor 补齐
            pos = self.pos_embed(max_required_len) 
            pos = pos.reshape(-1, self.d_model).to(x.device) # 强行展平为 (Max_Len, C)

            # 支持 frame_offset 滑动窗口，截取当前需要的帧位置
            start_idx = frame_offset
            end_idx = min(frame_offset + seq_len, pos.shape[0])
            pos_slice = pos[start_idx : end_idx, :]

            # 如果长度不够，自动用最后一个位置补齐 (防止生成超过长视频且超出预设 pos 长度时报错)
            if pos_slice.shape[0] < seq_len:
                pad_len = seq_len - pos_slice.shape[0]
                pad_tensor = pos[-1:].repeat(pad_len, 1) if pos.shape[0] > 0 else torch.zeros(pad_len, self.d_model, device=x.device)
                pos_slice = torch.cat([pos_slice, pad_tensor], dim=0)

            pos = pos_slice.unsqueeze(1) # 强行锁定为 (L, 1, C) -> 变成 (14, 1, 1024)，完美对齐 x！

        # 3. Transformer 计算
        style = self.encoder(x, pos=pos, src_key_padding_mask=pad_mask) # (L, B, C)
        
        # 4. 先转置并强制内存连续
        style = style.permute(1, 0, 2).contiguous() # (B, L, C)
        
        # 5. 变换为 (Batch*Frames, 1, d_model) -> 绝对是 14*1*1024，再也不会是 473088
        style = style.view(batch_size * seq_len, 1, self.d_model) 
        
        # 6. 经过 Linear 层 (输出变为 14 * 1 * 8192)
        style = self.pre_image_condition_p(style) 
        
        # 7. 动态适配 Token 数量 (变成 14 * 8 * 1024)
        style = style.view(batch_size * seq_len, -1, self.d_model)

        return style