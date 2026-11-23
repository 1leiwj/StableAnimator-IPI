"""CPU integration check for the IPI adapter and a real UNet temporal block."""
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from animation.modules.ipi_adapter import MotionAdapter
from animation.modules.transformer_temporal_ipi import TransformerSpatioTemporalModel


def main():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    width, frames = 32, 4
    adapter = MotionAdapter(cross_attention_dim=width, clip_dim=16, num_tokens=2).eval()
    adapter.requires_grad_(False)
    block = TransformerSpatioTemporalModel(4, 8, width, cross_attention_dim=width).eval()
    block.requires_grad_(False)
    block.ipi_temporal_gate.fill_(0.25)
    pose = torch.rand(2, frames, 266) * 1.6 - .8
    pose[1].zero_()                              # unconditional member of the batch
    pose[0, 2, 2 * 4:2 * 4 + 2] = -1           # occluded right wrist
    clip = torch.randn(2, frames, 16)
    clip[1].zero_()
    latent = torch.randn(2 * frames, width, 4, 6)
    context = torch.randn(2 * frames, 5, width)
    indicator = torch.zeros(2, frames)

    def denoise(condition):
        return block(latent, encoder_hidden_states=context, ipi_temporal_v2=True,
                     ipi_features=condition, image_only_indicator=indicator, return_dict=False)[0]

    tokens = adapter(pose, clip, ipi_temporal_v2=True)
    base = denoise(tokens)
    assert not torch.equal(base, denoise(None)), "Global IPI control is inactive"
    adapter.enable_joint_motion([width])
    adapter.train()
    assert not adapter.pose_encoder.training
    condition = adapter(pose, clip, ipi_temporal_v2=True)
    assert torch.equal(tokens, condition.tokens)
    assert torch.equal(base, denoise(condition)), "Zero heads changed the parent model"
    weights = condition.spatial_weights(4, 6)
    assert not weights[frames:].any() and not weights[2, :, 1].any()
    target = torch.randn_like(base)
    (denoise(condition) * target).mean().backward()
    head = adapter.joint_motion.readouts[str(width)].weight
    assert head.grad is not None and head.grad.abs().sum() > 0
    assert all(p.grad is None for p in block.parameters())
    assert all(p.grad is None for p in adapter.pose_encoder.parameters())
    with torch.no_grad():
        head.add_(head.grad.sign(), alpha=0.01)
    adapter.zero_grad(set_to_none=True)
    trained = adapter(pose, clip, ipi_temporal_v2=True)
    result = denoise(trained)
    assert not torch.equal(result[:frames], base[:frames])
    assert torch.equal(result[frames:], base[frames:])
    assert torch.equal(denoise(trained.scaled(0)), denoise(torch.zeros_like(tokens)))
    assert torch.equal(denoise(trained.tokens), base), "Disabling only the joints changed global IPI"
    (result * target).mean().backward()
    assert adapter.joint_motion.encode[0].weight.grad.abs().sum() > 0
    assert adapter.joint_motion.rgb_delta.weight.grad.abs().sum() > 0
    local = adapter(pose[:, 2:], clip[:, 2:], ipi_temporal_v2=True, previous=(pose[:, 1:2], clip[:, 1:2]))
    torch.testing.assert_close(local.values[width].unflatten(0, (2, 2)),
                               trained.values[width].unflatten(0, (2, frames))[:, 2:])
    # Both joints on the same frame have independent spatial locations.
    points = pose[0, 0].reshape(133, 2)
    points[4] = torch.tensor([-0.75, 0.75])
    points[7] = torch.tensor([0.75, -0.75])
    single = adapter(pose[:1, :1], clip[:1, :1], ipi_temporal_v2=True)
    maps = single.spatial_weights(4, 4)[0]
    assert maps[:, 1].argmax() == 12 and maps[:, 3].argmax() == 3
    clone = MotionAdapter(cross_attention_dim=width, clip_dim=16, num_tokens=2)
    clone.enable_joint_motion([width])
    clone.load_state_dict(adapter.state_dict(), strict=True)
    clone.eval()
    torch.testing.assert_close(clone(pose[:1, :1], clip[:1, :1], ipi_temporal_v2=True).values[width],
                               single.values[width])
    print("PASS: zero-init, global-only/off, temporal-block gradients, CFG, occlusion, "
          "joint locations, window overlap and checkpoint reload.")


if __name__ == "__main__":
    main()
