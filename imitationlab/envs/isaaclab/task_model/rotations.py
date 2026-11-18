"""Batched, differentiable SO(3) helpers for the world model (quaternions w, x, y, z).

Quaternion-to-matrix and rot6d live in ``teleop.g1_kinematics``.
"""

from __future__ import annotations

import torch
from torch import Tensor


def mat_to_quat(R: Tensor) -> Tensor:
    """Rotation matrix (..., 3, 3) -> unit quaternion (..., 4), w >= 0."""
    m = R
    tr = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    cand = torch.stack(
        (
            1 + tr,
            1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2],
            1 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2],
            1 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2],
        ),
        -1,
    )
    best = cand.argmax(-1, keepdim=True)
    s = cand.clamp_min(1e-12).sqrt()  # 2 * |component|
    w = torch.stack(
        (
            s[..., 0] / 2,
            (m[..., 2, 1] - m[..., 1, 2]) / (2 * s[..., 1]),
            (m[..., 0, 2] - m[..., 2, 0]) / (2 * s[..., 2]),
            (m[..., 1, 0] - m[..., 0, 1]) / (2 * s[..., 3]),
        ),
        -1,
    )
    x = torch.stack(
        (
            (m[..., 2, 1] - m[..., 1, 2]) / (2 * s[..., 0]),
            s[..., 1] / 2,
            (m[..., 0, 1] + m[..., 1, 0]) / (2 * s[..., 2]),
            (m[..., 0, 2] + m[..., 2, 0]) / (2 * s[..., 3]),
        ),
        -1,
    )
    y = torch.stack(
        (
            (m[..., 0, 2] - m[..., 2, 0]) / (2 * s[..., 0]),
            (m[..., 0, 1] + m[..., 1, 0]) / (2 * s[..., 1]),
            s[..., 2] / 2,
            (m[..., 1, 2] + m[..., 2, 1]) / (2 * s[..., 3]),
        ),
        -1,
    )
    z = torch.stack(
        (
            (m[..., 1, 0] - m[..., 0, 1]) / (2 * s[..., 0]),
            (m[..., 0, 2] + m[..., 2, 0]) / (2 * s[..., 1]),
            (m[..., 1, 2] + m[..., 2, 1]) / (2 * s[..., 2]),
            s[..., 3] / 2,
        ),
        -1,
    )
    q = torch.stack(tuple(c.gather(-1, best).squeeze(-1) for c in (w, x, y, z)), -1)
    q = q * torch.where(q[..., :1] < 0, -1.0, 1.0)
    return q / q.norm(dim=-1, keepdim=True)


def skew(v: Tensor) -> Tensor:
    x, y, z = v.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack(
        (
            torch.stack((o, -z, y), -1),
            torch.stack((z, o, -x), -1),
            torch.stack((-y, x, o), -1),
        ),
        -2,
    )


def so3_exp(w: Tensor) -> Tensor:
    """Rotation vector (..., 3) -> rotation matrix, Rodrigues with a Taylor floor."""
    theta2 = (w * w).sum(-1, keepdim=True).unsqueeze(-1)
    theta = theta2.clamp_min(1e-12).sqrt()
    small = theta2 < 1e-6
    a = torch.where(small, 1 - theta2 / 6, torch.sin(theta) / theta)
    b = torch.where(
        small, 0.5 - theta2 / 24, (1 - torch.cos(theta)) / theta2.clamp_min(1e-12)
    )
    K = skew(w)
    eye = torch.eye(3, dtype=w.dtype, device=w.device).expand(K.shape)
    return eye + a * K + b * (K @ K)


def so3_log(R: Tensor) -> Tensor:
    """Rotation matrix -> rotation vector (..., 3); exact away from theta = pi."""
    vee = torch.stack(
        (
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1],
        ),
        -1,
    )
    cos = ((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1) / 2).clamp(-1.0, 1.0)
    sin = 0.5 * vee.norm(dim=-1).clamp_min(1e-12)
    theta = torch.atan2(sin, cos)
    scale = torch.where(theta < 1e-4, 0.5 + theta * theta / 12, theta / (2 * sin))
    return vee * scale.unsqueeze(-1)


def rot_error_sq(R_a: Tensor, R_b: Tensor) -> Tensor:
    """Smooth squared rotation distance 3 - tr(R_a^T R_b) = 4 sin^2(theta/2).

    Equals theta^2 to second order and, unlike acos/atan2 forms, has a
    well-defined gradient at zero error.
    """
    return 3.0 - (R_a * R_b).sum(dim=(-1, -2))


def yaw_of(R: Tensor) -> Tensor:
    return torch.atan2(R[..., 1, 0], R[..., 0, 0])


def yaw_mat(yaw: Tensor) -> Tensor:
    c, s = torch.cos(yaw), torch.sin(yaw)
    o, i = torch.zeros_like(yaw), torch.ones_like(yaw)
    return torch.stack(
        (
            torch.stack((c, -s, o), -1),
            torch.stack((s, c, o), -1),
            torch.stack((o, o, i), -1),
        ),
        -2,
    )


def twist_mat(R: Tensor) -> Tensor:
    """Rotation about world z by the swing-twist twist angle of R.

    The env's ``heading_quat`` (zero a quaternion's x/y parts, renormalize):
    tan(theta / 2) = q_z / q_w = (R10 - R01) / (1 + tr R). Unlike
    ``heading_mat`` (the ZYX yaw) it stays continuous through pitch = 90 deg.
    """
    a = R[..., 1, 0] - R[..., 0, 1]
    b = 1.0 + R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    n = (a * a + b * b).clamp_min(1e-12)
    c, s = (b * b - a * a) / n, 2.0 * a * b / n
    o, i = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack(
        (
            torch.stack((c, -s, o), -1),
            torch.stack((s, c, o), -1),
            torch.stack((o, o, i), -1),
        ),
        -2,
    )


def heading_mat(R: Tensor) -> Tensor:
    """Yaw-only rotation of R (the heading frame), differentiable via the
    projected x axis instead of atan2."""
    fx = R[..., 0, 0]
    fy = R[..., 1, 0]
    n = (fx * fx + fy * fy).clamp_min(1e-9).sqrt()
    c, s = fx / n, fy / n
    o, i = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack(
        (
            torch.stack((c, -s, o), -1),
            torch.stack((s, c, o), -1),
            torch.stack((o, o, i), -1),
        ),
        -2,
    )
