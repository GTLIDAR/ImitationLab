"""Anchor-jitter randomization of the live macro-window anchor."""

from __future__ import annotations

import types

import torch


def test_anchor_jitter_offsets_only_xy_and_resets_the_walk():
    from imitationlab.envs.isaaclab.envs.expert_data_plane import ExpertDataPlane

    plane = ExpertDataPlane.__new__(ExpertDataPlane)
    plane._env = types.SimpleNamespace(
        num_envs=6,
        cfg=types.SimpleNamespace(
            expert_macro_anchor_walk_std=0.01,
            expert_macro_anchor_walk_max=0.02,
            expert_macro_anchor_jitter=0.005,
            expert_macro_anchor_jump_prob=1.0,
            expert_macro_anchor_jump=0.03,
        ),
    )
    env_ids = torch.arange(6)
    base = torch.zeros(6, 3)
    torch.manual_seed(0)
    out = plane._jittered_live_anchor(base, env_ids)
    assert out.shape == (6, 3)
    assert torch.all(out[:, 2] == 0.0)
    assert out[:, :2].abs().max() <= 0.02 + 0.005 + 0.03 + 1e-6
    assert out[:, :2].abs().sum() > 0.0
    for _ in range(50):
        plane._jittered_live_anchor(base, env_ids)
    assert plane._anchor_walk_state.abs().max() <= 0.02 + 1e-6
    plane.reset_anchor_jitter(torch.tensor([0, 1]))
    assert plane._anchor_walk_state[:2].abs().sum() == 0.0
    plane._env.cfg.expert_macro_anchor_walk_std = 0.0
    plane._env.cfg.expert_macro_anchor_jitter = 0.0
    plane._env.cfg.expert_macro_anchor_jump_prob = 0.0
    assert plane._jittered_live_anchor(base, env_ids) is base
