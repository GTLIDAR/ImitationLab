"""Quaternion (w, x, y, z) helpers used by the G1 world-model state."""

from __future__ import annotations

import torch
from torch import Tensor


def quat_mul(a: Tensor, b: Tensor) -> Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        -1,
    )


def quat_conj(q: Tensor) -> Tensor:
    return torch.cat((q[..., :1], -q[..., 1:]), -1)


def quat_exp(rotvec: Tensor) -> Tensor:
    """Rotation vector -> unit quaternion, with a Taylor floor near zero."""
    theta2 = (rotvec * rotvec).sum(-1, keepdim=True)
    theta = theta2.clamp_min(1e-12).sqrt()
    small = theta2 < 1e-8
    half = 0.5 * theta
    w = torch.where(small, 1 - theta2 / 8, torch.cos(half))
    s = torch.where(small, 0.5 - theta2 / 48, torch.sin(half) / theta)
    return torch.cat((w, s * rotvec), -1)


def quat_normalize(q: Tensor) -> Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-9)
