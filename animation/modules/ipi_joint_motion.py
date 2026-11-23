"""Localized temporal conditioning for the OpenPose body joints in dense133."""
from typing import NamedTuple

import torch
from torch import nn


class JointMotionCondition(NamedTuple):
    tokens: torch.Tensor
    centers: torch.Tensor                 # [B*F, 8, 2], model coordinates
    visible: torch.Tensor                 # [B*F, 8]
    values: dict                          # channel width -> [B*F, 8, C]

    def scaled(self, scale):
        if scale == 1:
            return self
        return JointMotionCondition(self.tokens * scale, self.centers, self.visible,
                                    {c: v * scale for c, v in self.values.items()})

    def spatial_weights(self, height, width):
        # Evaluate the Gaussian at latent pixel centers. The same radius in x and y
        # (in latent pixels) also works for portrait and landscape canvases.
        centers = self.centers.float()
        x = torch.arange(width, device=centers.device, dtype=torch.float32) + 0.5
        y = torch.arange(height, device=centers.device, dtype=torch.float32) + 0.5
        radius = max(1., 0.05 * min(height, width))
        dx = (x[None, :, None] - (centers[:, None, :, 0] + 1) * (width / 2)) / radius
        dy = (y[None, :, None] - (centers[:, None, :, 1] + 1) * (height / 2)) / radius
        wx = (-0.5 * dx.square()).exp() * self.visible[:, None, :]
        wy = (-0.5 * dy.square()).exp()
        return (wy[:, :, None, :] * wx[:, None, :, :]).flatten(1, 2)

    def inject(self, hidden, weights):
        values = self.values[hidden.shape[-1]]
        residual = torch.bmm(weights.to(hidden.dtype), values.to(hidden.dtype))
        return hidden + residual.to(hidden.dtype)


class JointMotionEncoder(nn.Module):
    """Encode bone orientation and successive observations at each limb joint."""

    def __init__(self, clip_dim, widths):
        super().__init__()
        # Right/left elbows, wrists, knees and ankles, in OpenPose-18 order.
        self.register_buffer("joints", torch.tensor([3, 4, 6, 7, 9, 10, 12, 13]), persistent=False)
        self.register_buffer("parents", torch.tensor([2, 3, 5, 6, 8, 9, 11, 12]), persistent=False)
        self.register_buffer("identity", torch.eye(8), persistent=False)
        self.rgb_delta = nn.Linear(clip_dim, 16, bias=False)
        # center xy, bone xy, center displacement xy, bone change xy, joint id, RGB change
        self.encode = nn.Sequential(nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, 64), nn.LayerNorm(64))
        self.readouts = nn.ModuleDict({str(c): nn.Linear(64, c, bias=False) for c in sorted(set(widths))})
        for head in self.readouts.values():
            nn.init.zeros_(head.weight)

    def forward(self, pose, clip, tokens, previous=None):
        # Include the preceding observation when entering an overlapping window.
        # At the clip boundary, repeating frame zero gives exactly zero velocity.
        first_pose, first_clip = (pose[:, :1], clip[:, :1]) if previous is None else previous
        sequence = torch.cat((first_pose, pose), dim=1).unflatten(-1, (133, 2))
        clip_sequence = torch.cat((first_clip, clip), dim=1)
        centers = sequence.index_select(2, self.joints)
        parents = sequence.index_select(2, self.parents)
        present = torch.isfinite(sequence).all(-1) & (sequence != -1).any(-1)
        # CFG dropout represents an absent *frame* by an all-zero tensor.
        present = present & (sequence != 0).any(-1).any(-1, keepdim=True)
        valid = present.index_select(2, self.joints)
        bone_valid = valid & present.index_select(2, self.parents)
        centers = torch.where(valid[..., None], centers, 0.)
        bones = torch.where(bone_valid[..., None], centers - parents, 0.)
        velocity = torch.where((valid[:, 1:] & valid[:, :-1])[..., None], centers.diff(dim=1), 0.)
        bend = torch.where((bone_valid[:, 1:] & bone_valid[:, :-1])[..., None], bones.diff(dim=1), 0.)
        centers = centers[:, 1:]
        rgb = self.rgb_delta(clip_sequence.diff(dim=1).to(self.rgb_delta.weight.dtype))
        features = torch.cat((centers, bones[:, 1:], velocity, bend,
                              self.identity.expand(*centers.shape[:-1], 8),
                              rgb.unsqueeze(2).expand(-1, -1, 8, -1)), dim=-1)
        encoded = self.encode(features.flatten(0, 1).to(self.encode[0].weight.dtype))
        values = {int(c): head(encoded) for c, head in self.readouts.items()}
        return JointMotionCondition(tokens, centers.flatten(0, 1), valid[:, 1:].flatten(0, 1), values)
