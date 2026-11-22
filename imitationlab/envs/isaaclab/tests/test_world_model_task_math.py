"""World-model state math and G1 kinematics (CPU, no Isaac).

The env-level contract (analytic reward == the env's reward terms, history
rebuild == Isaac's buffer) is checked against the live env by
``imitationlab.evaluation.world_model.validate_task``.
"""

from __future__ import annotations

import math
import subprocess
import sys

import torch

from imitationlab.envs.isaaclab.task_model.g1_kinematics import (
    G1_TRACKED_BODY_NAMES,
    G1Kinematics,
    quat_wxyz_to_matrix,
)
from imitationlab.envs.isaaclab.task_model.g1_math import (
    increment,
    integrate,
    invariant_features,
    state_error,
    unpack,
)
from imitationlab.envs.isaaclab.task_model.quat import (
    quat_exp,
    quat_mul,
    quat_normalize,
)
from imitationlab.envs.isaaclab.task_model.rotations import (
    heading_mat,
    so3_exp,
    so3_log,
)

F64 = torch.float64


def test_importing_g1_math_does_not_register_simulator_tasks():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from imitationlab.envs.isaaclab.task_model.g1_kinematics import G1_URDF_PATH
from imitationlab.envs.isaaclab.task_model.g1_task import G1TrackingTask

assert G1_URDF_PATH.is_file()
assert "isaaclab" not in sys.modules
assert "imitationlab.envs.isaaclab.tasks" not in sys.modules
""",
        ],
        check=True,
    )


def _random_state(n: int, gen: torch.Generator) -> torch.Tensor:
    p = torch.randn(n, 3, generator=gen, dtype=F64) + torch.tensor(
        [0.0, 0.0, 0.8], dtype=F64
    )
    q = quat_normalize(torch.randn(n, 4, generator=gen, dtype=F64))
    v = torch.randn(n, 3, generator=gen, dtype=F64)
    w = torch.randn(n, 3, generator=gen, dtype=F64) * 2
    qj = torch.randn(n, 29, generator=gen, dtype=F64) * 0.5
    qd = torch.randn(n, 29, generator=gen, dtype=F64) * 2
    return torch.cat((p, q, v, w, qj, qd), -1)


def _yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(yaw)
    return torch.stack((torch.cos(yaw / 2), z, z, torch.sin(yaw / 2)), -1)


def _transform(
    state: torch.Tensor, yaw: torch.Tensor, shift: torch.Tensor
) -> torch.Tensor:
    """Rotate the whole state about world z by ``yaw`` and translate it."""
    p, q, v, w, qj, qd = unpack(state)
    Rz = quat_wxyz_to_matrix(_yaw_quat(yaw))
    rot = lambda x: (Rz @ x.unsqueeze(-1)).squeeze(-1)  # noqa: E731
    return torch.cat(
        (rot(p) + shift, quat_mul(_yaw_quat(yaw), q), rot(v), rot(w), qj, qd), -1
    )


def test_fk_velocity_matches_finite_difference() -> None:
    gen = torch.Generator().manual_seed(0)
    fk = G1Kinematics().double()
    s = _random_state(8, gen)
    p, q, v, w, qj, qd = unpack(s)
    R = quat_wxyz_to_matrix(q)
    pos, rot, lin, ang = fk.forward_with_velocity(
        p, R, qj, v, w, qd, G1_TRACKED_BODY_NAMES
    )
    pos2, rot2 = fk(p, R, qj, G1_TRACKED_BODY_NAMES)
    assert torch.allclose(pos, pos2) and torch.allclose(rot, rot2)
    eps = 1e-6
    R_eps = so3_exp(w * eps) @ R  # world-frame angular velocity
    pos_e, rot_e = fk(p + v * eps, R_eps, qj + qd * eps, G1_TRACKED_BODY_NAMES)
    assert torch.allclose((pos_e - pos) / eps, lin, atol=1e-4)
    ang_fd = so3_log(rot_e @ rot.transpose(-1, -2)) / eps
    assert torch.allclose(ang_fd, ang, atol=1e-4)


def test_increment_integrate_roundtrip() -> None:
    gen = torch.Generator().manual_seed(1)
    s0 = _random_state(64, gen)
    inc = torch.randn(64, 70, generator=gen, dtype=F64) * 0.05
    s1 = integrate(s0, inc)
    assert torch.allclose(increment(s0, s1), inc, atol=1e-9)
    assert torch.allclose(
        state_error(s1, s1), torch.zeros(64, 70, dtype=F64), atol=1e-9
    )
    back = integrate(s0, increment(s0, s1))
    # Quaternions agree up to sign.
    assert torch.allclose(back[:, :3], s1[:, :3], atol=1e-9)
    dot = (back[:, 3:7] * s1[:, 3:7]).sum(-1).abs()
    assert torch.allclose(dot, torch.ones(64, dtype=F64), atol=1e-9)


def test_world_model_frames_are_yaw_and_translation_invariant() -> None:
    gen = torch.Generator().manual_seed(2)
    s0 = _random_state(32, gen)
    s1 = integrate(s0, torch.randn(32, 70, generator=gen, dtype=F64) * 0.05)
    yaw = torch.rand(32, generator=gen, dtype=F64) * 2 * math.pi
    shift = torch.randn(32, 3, generator=gen, dtype=F64) * torch.tensor(
        [5.0, 5.0, 0.0], dtype=F64
    )
    t0, t1 = _transform(s0, yaw, shift), _transform(s1, yaw, shift)
    assert torch.allclose(invariant_features(t0), invariant_features(s0), atol=1e-9)
    assert torch.allclose(increment(t0, t1), increment(s0, s1), atol=1e-9)
    assert torch.allclose(
        state_error(t1, _transform(s0, yaw, shift)), state_error(s1, s0), atol=1e-9
    )


def test_heading_and_quat_exp() -> None:
    gen = torch.Generator().manual_seed(3)
    q = quat_normalize(torch.randn(16, 4, generator=gen, dtype=F64))
    R = quat_wxyz_to_matrix(q)
    H = heading_mat(R)
    tilt = H.transpose(-1, -2) @ R
    # the tilt has no yaw: its x axis has no y component
    assert torch.allclose(tilt[:, 1, 0], torch.zeros(16, dtype=F64), atol=1e-9)
    rv = torch.randn(16, 3, generator=gen, dtype=F64) * 0.7
    assert torch.allclose(quat_wxyz_to_matrix(quat_exp(rv)), so3_exp(rv), atol=1e-9)


def _quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return (quat_wxyz_to_matrix(q).transpose(-1, -2) @ v.unsqueeze(-1)).squeeze(-1)


def _env_heading_quat(q: torch.Tensor) -> torch.Tensor:
    """``_compiled.heading_quat`` in (w, x, y, z): zero x and y, renormalize."""
    out = torch.zeros_like(q)
    out[..., 0], out[..., 3] = q[..., 0], q[..., 3]
    return out / out.norm(dim=-1, keepdim=True)


def test_twist_mat_matches_env_heading_quat() -> None:
    from imitationlab.envs.isaaclab.task_model.rotations import mat_to_quat, twist_mat

    gen = torch.Generator().manual_seed(5)
    q = quat_normalize(torch.randn(256, 4, generator=gen, dtype=F64))
    R = quat_wxyz_to_matrix(q)
    assert torch.allclose(
        twist_mat(R), quat_wxyz_to_matrix(_env_heading_quat(q)), atol=1e-9
    )
    # continuous through pitch = 90 deg, where the ZYX yaw (heading_mat) is not
    pitch = torch.tensor([math.pi / 2 - 1e-3, math.pi / 2 + 1e-3], dtype=F64)
    Ry = so3_exp(pitch.unsqueeze(-1) * torch.tensor([0.0, 1.0, 0.0], dtype=F64))
    T = twist_mat(Ry)
    assert torch.allclose(T[0], T[1], atol=1e-2)
    assert torch.allclose(mat_to_quat(T)[:, 3], torch.zeros(2, dtype=F64), atol=1e-9)


def test_frozen_skill_encoder_loads_binds_and_refuses_other_frames(tmp_path) -> None:
    import pytest
    from imitationlab.encoders.diffsr.pretrain import HighLevelSkillDiffSRConfig
    from imitationlab.encoders.skill_encoder import build_skill_encoder

    from imitationlab.envs.isaaclab.task_model.g1_task import FrozenSkillEncoder

    torch.manual_seed(7)
    cfg = HighLevelSkillDiffSRConfig(
        horizon_steps=4,
        z_dim=8,
        encoder_hidden_dims=(16,),
        encoder_window_mode="intermediate",
        macro_anchor_mode="robot_heading",
        macro_frame_stride=1,
        source_anchor="current",
    )
    enc = build_skill_encoder(
        state_dim=38,
        window_steps=3,
        z_dim=8,
        hidden_dims=(16,),
        spec=cfg.latent_spec(),
        activation=cfg.encoder_activation,
        layer_norm=cfg.encoder_layer_norm,
    )
    path = tmp_path / "enc.pt"
    torch.save(
        {"config": cfg.to_dict(), "skill_encoder_state_dict": enc.state_dict()}, path
    )
    skill = FrozenSkillEncoder(path, torch.device("cpu"))
    assert skill.frames == 4 and skill.command_dim == 10
    frames = torch.randn(2, 3, 4, 38, requires_grad=True)
    cmd = skill(frames)
    assert cmd.shape == (2, 3, 10)
    assert torch.equal(cmd[..., 8:], torch.tensor([0.0, 1.0]).expand(2, 3, 2))
    expected = enc(frames[:, :, 0].reshape(-1, 38), frames[:, :, 1:].reshape(-1, 3, 38))
    assert torch.allclose(cmd[..., :8].reshape(-1, 8), expected, atol=1e-6)
    cmd.sum().backward()
    assert frames.grad is not None and frames.grad.abs().sum() > 0
    assert all(
        p.grad is None and not p.requires_grad for p in skill.encoder.parameters()
    )
    assert (
        FrozenSkillEncoder(path, torch.device("cpu")).weights_sha256
        == skill.weights_sha256
    )

    other = HighLevelSkillDiffSRConfig.from_dict(
        {**cfg.to_dict(), "macro_anchor_mode": "robot"}
    )
    torch.save(
        {"config": other.to_dict(), "skill_encoder_state_dict": enc.state_dict()}, path
    )
    with pytest.raises(ValueError, match="macro_anchor_mode"):
        FrozenSkillEncoder(path, torch.device("cpu"))
