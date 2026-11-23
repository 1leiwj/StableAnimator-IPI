import torch

ckpt = torch.load(
    "/gaojiawei/wjl/StableAnimator/checkpoints/Animation_ipi_2/checkpoint-500/ipi_adapter-500.pth",
    map_location="cpu"
)

# 检查1: motion_scale 是否已经爆炸
for k, v in ckpt.items():
    if 'motion_scale' in k:
        print(f"motion_scale = {v.item():.4f}")   # 预期：如果 > 5 就是漂移了

# 检查2: ipi_adapter 各层是否含 NaN / Inf
nan_layers, inf_layers = [], []
for k, v in ckpt.items():
    if 'ipi_adapter' in k:
        if torch.isnan(v).any():
            nan_layers.append(k)
        if torch.isinf(v).any():
            inf_layers.append(k)

print("NaN layers:", nan_layers[:5])
print("Inf layers:", inf_layers[:5])

# 检查3: 找出各层最大绝对值，看哪层数值最异常
for k, v in ckpt.items():
    if 'ipi_adapter' in k and v.numel() > 0:
        max_abs = v.abs().max().item()
        if max_abs > 100:  # bf16 上限 65504，超过100就可疑
            print(f"⚠️ 大值异常: {k} | max_abs={max_abs:.2f}")
