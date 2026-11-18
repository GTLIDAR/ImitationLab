"""Online geometry must not shift the first encoded reward target."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from imitationlab.envs.isaaclab.envs.expert_data_plane import ExpertDataPlane
from imitationlab.envs.isaaclab.tasks.manager_based.imitation.mdp._compiled import (
    body_pose_in_anchor_frame,
    heading_anchor_frame,
    quat_to_rot6d_flat,
)
from imitationlab.encoders.diffsr.pretrain import (
    HighLevelSkillDiffSRConfig,
    _encoder_input_window,
)
from imitationlab.encoders.skill_encoder import DeterministicSkillEncoder


def _plane(stride=1):
    plane = object.__new__(ExpertDataPlane)
    plane._env = SimpleNamespace(
        cfg=SimpleNamespace(
            expert_macro_anchor_mode="robot_heading",
            expert_macro_rollout_anchor_mode="previous_expert_heading",
            expert_macro_frame_stride=stride,
            expert_macro_state_terms=[
                "expert_motion_qpos",
                "expert_anchor_pos_b",
                "expert_anchor_ori_b",
            ],
        ),
        device="cpu",
        num_envs=3,
        _command_ee_body_names=(),
        _command_keypoint_body_names=(),
    )
    length = torch.tensor([64, 64])
    plane.trajectory_manager = SimpleNamespace(
        state_device=torch.device("cpu"),
        length=length,
        _length=length,
        start=torch.tensor([0, 64]),
        env_traj_rank=torch.tensor([0, 1, 1]),
        env_step=torch.tensor([5, 0, 20]),
        num_trajectories=2,
    )
    plane._expert_anchor_body_name = "pelvis"
    plane._get_joint_ids_tensor_fast = lambda ids: ids
    frame = torch.arange(128, dtype=torch.float32)
    pos = torch.stack([0.1 * frame, 0.03 * frame, 0.8 + 0.001 * frame], dim=-1)
    angle = frame * 0.02
    quat = torch.stack(
        [
            torch.zeros_like(frame),
            torch.zeros_like(frame),
            torch.sin(angle / 2),
            torch.cos(angle / 2),
        ],
        dim=-1,
    )
    qpos = frame[:, None].expand(-1, 29).clone()
    cache = {"joint_pos": qpos, "anchor_pos_w": pos, "anchor_quat_w": quat}
    plane._ensure_compact_macro_cache = lambda: cache
    return plane, cache


@pytest.mark.parametrize("stride,history", [(1, 0), (1, 5), (3, 0), (3, 5)])
def test_online_input_starts_at_reward_reference_and_uses_previous_frame(
    stride, history
):
    plane, cache = _plane(stride)
    manager = plane.trajectory_manager
    cursor = manager.env_step.clone()
    index = manager.start[manager.env_traj_rank] + cursor
    batch = plane.current_expert_macro_transition_batch(10, state_history_steps=history)
    state, future = batch["hl", "state"], batch["hl", "future_window"]

    # Joint identity pins the exact reward reference; anchoring cannot shift it.
    torch.testing.assert_close(state[:, :29], cache["joint_pos"][index])
    torch.testing.assert_close(batch["hl", "local_step"], cursor)
    torch.testing.assert_close(manager.env_step, cursor)
    previous_index = manager.start[manager.env_traj_rank] + (cursor - 1).clamp_min(0)
    origin, heading = heading_anchor_frame(
        cache["anchor_pos_w"][previous_index], cache["anchor_quat_w"][previous_index]
    )
    expected_pos, expected_quat = body_pose_in_anchor_frame(
        origin,
        heading,
        cache["anchor_pos_w"][index, None],
        cache["anchor_quat_w"][index, None],
    )
    torch.testing.assert_close(state[:, 29:32], expected_pos[:, 0])
    torch.testing.assert_close(state[:, 32:38], quat_to_rot6d_flat(expected_quat[:, 0]))
    assert state[0, 29:31].norm() > 0
    # A reset at frame zero must not borrow the last frame of another trajectory.
    torch.testing.assert_close(state[1, 29:31], torch.zeros(2), atol=1e-6, rtol=0)

    # Inspect the actual first MLP input: source state is prepended and the
    # intermediate slice still excludes the endpoint. It is exactly 10 x 38.
    config = HighLevelSkillDiffSRConfig(
        horizon_steps=10, encoder_window_mode="intermediate"
    )
    encoder = DeterministicSkillEncoder(
        state_dim=38, window_steps=9, z_dim=4, hidden_dims=(8,)
    )
    captured = []
    hook = encoder.net[0].register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().clone())
    )
    encoder(state, _encoder_input_window(config, future))
    hook.remove()
    actual_input = captured[0].reshape(3, 10, 38)
    expected_indices = index[:, None] + stride * torch.arange(10)[None]
    torch.testing.assert_close(actual_input[:, :, 0], expected_indices.float())
    torch.testing.assert_close(
        batch["hl", "target"][:, 0], (index + stride * 10).float()
    )
    if history:
        torch.testing.assert_close(batch["hl", "state_history"][:, -1], state)


def test_offline_sampler_is_bitwise_unchanged_by_rollout_override():
    plane, _ = _plane()
    kwargs = dict(
        batch_size=8, horizon_steps=10, state_history_steps=5, trajectory_ranks=[0, 1]
    )
    torch.manual_seed(2026)
    changed = plane.sample_expert_macro_transition_batch(**kwargs)
    plane._env.cfg.expert_macro_rollout_anchor_mode = "inherit"
    torch.manual_seed(2026)
    original = plane.sample_expert_macro_transition_batch(**kwargs)
    for key in original["hl"].keys():
        assert torch.equal(changed["hl", key], original["hl", key]), key


def test_previous_anchor_requires_explicit_origin_metadata():
    plane, _ = _plane()
    window = plane._sample_expert_macro_window_for_trajectory_ranks(
        torch.tensor([0]), torch.tensor([5]), past_steps=0, future_steps=10
    )
    with pytest.raises(ValueError, match="context='rollout'"):
        plane._build_expert_macro_window_terms(
            window, torch.tensor([0]), context="rollout", past_steps=0
        )


def test_invalid_rollout_anchor_is_rejected():
    plane, _ = _plane()
    plane._env.cfg.expert_macro_rollout_anchor_mode = "typo"
    with pytest.raises(ValueError, match="expert_macro_rollout_anchor_mode"):
        plane.current_expert_macro_transition_batch(10)


def test_tracker_config_records_override_without_changing_encoder_mode():
    from imitationlab.envs.isaaclab.tasks.manager_based.imitation.config.g1.imitation_g1_env import (
        ImitationG1EnvCfg,
    )

    cfg = ImitationG1EnvCfg()
    original_mode = cfg.expert_macro_anchor_mode
    assert cfg.expert_macro_rollout_anchor_mode == "inherit"
    cfg.expert_macro_rollout_anchor_mode = "previous_expert_heading"
    assert (
        cfg.to_dict()["expert_macro_rollout_anchor_mode"] == "previous_expert_heading"
    )
    assert cfg.expert_macro_anchor_mode == original_mode


def test_body_pose_fallback_matches_compact_cache():
    plane, cache = _plane()
    expected = plane.current_expert_macro_transition_batch(10)
    plane._ensure_compact_macro_cache = lambda: None
    plane._get_reference_body_ids_fast = lambda names: torch.tensor([0])
    manager = plane.trajectory_manager
    manager.storage_device = torch.device("cpu")
    # The fallback sampler swizzles dataset WXYZ into the runtime XYZW order.
    manager.rb = TensorDict(
        {
            "joint_pos": cache["joint_pos"],
            "body_pos_w": cache["anchor_pos_w"][:, None],
            "body_quat_w": cache["anchor_quat_w"][:, None, [3, 0, 1, 2]],
        },
        batch_size=[128],
    )
    manager.attach_reference_fields = lambda window, use_buffers: window
    actual = plane.current_expert_macro_transition_batch(10)
    for key in expected["hl"].keys():
        torch.testing.assert_close(actual["hl", key], expected["hl", key])
