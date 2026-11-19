"""Independent numerical contracts for the replica-derived imitation-g1-v1 task.

Expected values are elementary equations and explicit per-coordinate checks,
not a second call to the implementation or imported replica code.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from imitationlab.envs.isaaclab.task_model.g1_kinematics import G1_ISAACLAB_JOINT_NAMES
from imitationlab.envs.isaaclab.task_model.g1_math import (
    CONTROL_DT,
    OBS_NOISE_WIDTH,
    G1MDP,
    G1ReferenceLibrary,
    joint_limits,
    normalize_context,
    rotation_error_sq,
    rotation_vector,
)
from imitationlab.envs.isaaclab.task_model.g1_task import G1TrackingTask
from imitationlab.envs.isaaclab.task_model.quat import quat_exp
from imitationlab.envs.isaaclab.task_model.rotations import so3_exp


def bank(lengths=(20, 5), *, moving=False) -> G1ReferenceLibrary:
    n = sum(lengths)
    qpos = torch.zeros(n, 36)
    qpos[:, 2], qpos[:, 3] = 0.8, 1
    qpos[:, 7:] = joint_limits("cpu").mean(-1)
    offset = 0
    for clip, length in enumerate(lengths):
        t = torch.arange(length) * CONTROL_DT
        qpos[offset : offset + length, 0] = clip * 100 + (t.square() if moving else 0)
        if moving:
            qpos[offset : offset + length, 3:7] = quat_exp(
                torch.stack((0 * t, 0 * t, 0.5 * t), -1)
            )
            qpos[offset : offset + length, 7] += t.square()
        offset += length
    return G1ReferenceLibrary.from_qpos(qpos, lengths)


def task(lib: G1ReferenceLibrary, *, noise=True) -> G1TrackingTask:
    q0 = joint_limits("cpu").mean(-1)
    robot = SimpleNamespace(
        data=SimpleNamespace(default_joint_pos=q0.unsqueeze(0)),
        find_joints=lambda names, preserve_order: (list(range(29)), names),
    )
    env = SimpleNamespace(
        trajectory_manager=object(),
        reward_manager=object(),
        scene={"robot": robot},
        g1_library=lib,
        step_dt=CONTROL_DT,
    )
    return G1TrackingTask(env, "cpu", sensor_noise=noise)


def test_reference_differentiates_each_clip_without_cross_clip_leakage():
    lib = bank((5, 3), moving=True)
    assert lib.resident_bytes() == 8 * 71 * 4
    # p(t)=t^2: endpoint slope dt, interior 2*t, final 2*t-dt.
    assert torch.allclose(
        lib.states[:5, 7], torch.tensor([0.02, 0.04, 0.08, 0.12, 0.14]), atol=1e-6
    )
    assert torch.allclose(
        lib.states[5:, 7], torch.tensor([0.02, 0.04, 0.06]), atol=4e-4
    )
    assert torch.allclose(lib.states[:5, 42], lib.states[:5, 7], atol=1e-5)
    assert torch.allclose(lib.states[:, 12], torch.full((8,), 0.5), atol=1e-5)
    assert torch.allclose(lib.states[:, 10:12], torch.zeros(8, 2), atol=1e-6, rtol=1e-5)
    # End extension clamps the state but records validity independently.
    ref = lib.gather(torch.tensor([1]), torch.tensor([[2, 3, 10]]))
    assert ref.valid.tolist() == [[True, False, False]]
    assert torch.equal(ref.root_pos[:, :1].expand_as(ref.root_pos), ref.root_pos)


def test_goal_future_order_reference_heading_and_terminal_velocity():
    lib = bank((5,), moving=True)
    rank, frame = torch.tensor([0]), torch.tensor([3])
    goal = lib.goal(rank, frame).reshape(1, 10, 73)
    assert goal.shape == (1, 10, 73)
    # All future samples hold the last pose; only f+1 is recorded here.
    assert torch.equal(goal[:, :1, :9].expand_as(goal[..., :9]), goal[..., :9])
    assert torch.count_nonzero(goal[:, 0, 9:15]) > 0
    assert torch.count_nonzero(goal[:, 0, 44:]) > 0
    assert torch.count_nonzero(goal[:, 1:, 9:15]) == 0
    assert torch.count_nonzero(goal[:, 1:, 44:]) == 0
    # Translation uses current REFERENCE heading, never the imagined robot.
    delta_x = 0.08**2 - 0.06**2
    theta = 0.5 * 0.06
    assert torch.allclose(
        goal[0, 0, :3],
        torch.tensor([math.cos(theta) * delta_x, -math.sin(theta) * delta_x, 0]),
        atol=1e-7,
    )
    assert torch.allclose(
        goal[0, 0, 44], torch.tensor(0.14 * 0.1), atol=1e-6, rtol=1e-5
    )


def test_observation_exact_layout_raw_units_and_relative_orientation():
    lib = bank((5,))
    t = task(lib, noise=False)
    rank, frame = torch.tensor([0]), torch.tensor([2])
    state = lib.ref_state(rank, frame).clone()
    state[:, 3:7] = quat_exp(torch.tensor([[0.0, 0.0, math.pi / 2]]))
    state[:, 10:13] = torch.tensor([[1.0, 2.0, 3.0]])
    state[:, 42:] = 7.0
    context = torch.zeros(1, 33)
    context[:, 1:30] = 0.01
    action = torch.arange(29).float().unsqueeze(0)
    prop = t.proprio_at(state, action, context, rank, frame, False)
    assert prop.shape == (1, 99)
    assert torch.allclose(prop[:, :29], torch.full((1, 29), -0.01), atol=1e-7)
    assert torch.equal(prop[:, 29:58], torch.full((1, 29), 7.0))  # no unused OBS_SCALE
    assert torch.allclose(
        prop[0, 58:61], torch.tensor([2.0, -1.0, 3.0]), atol=1e-6, rtol=1e-5
    )
    assert torch.allclose(
        prop[0, 61:64], torch.tensor([0.0, 0.0, -1.0]), atol=1e-6, rtol=1e-5
    )
    assert torch.equal(prop[:, 64:93], action)
    # R_robot^T R_reference: first two columns, interleaved by row.
    assert torch.allclose(
        prop[0, 93:], torch.tensor([0.0, 1.0, -1.0, 0.0, 0.0, 0.0]), atol=1e-6
    )


def test_reward_terms_match_hand_computed_kernels_and_penalties():
    lib = bank((5,))
    mdp = G1MDP(lib, joint_limits("cpu").mean(-1))
    rank, frame = torch.tensor([0]), torch.tensor([1])
    state = lib.ref_state(rank, frame).clone()
    ref = lib.gather(rank, frame)
    action, previous = torch.zeros(1, 29), torch.zeros(1, 29)
    # Measured-body override tests scalar formulas independently of FK.
    body_state = (
        ref.body_pos + torch.tensor([0.3, 0.0, 0.0]),
        ref.body_rot,
        ref.body_lin_vel + torch.tensor([1.0, 0.0, 0.0]),
        ref.body_ang_vel + torch.tensor([0.0, 0.0, 3.14]),
    )
    action[:, 0] = 0.5
    state[:, 13] = mdp.soft_hi[0] + 0.1
    reward, terms = mdp.reward(
        state, action, previous, rank, frame, body_state=body_state
    )
    for name in ("body_pos", "body_lin_vel", "body_ang_vel"):
        assert torch.allclose(
            terms[name], torch.tensor([math.exp(-1) * 0.02]), atol=1e-6
        )
    assert torch.allclose(terms["root_pos"], torch.tensor([0.01]), atol=1e-7)
    assert torch.allclose(terms["action_rate"], torch.tensor([-0.0005]), atol=1e-7)
    assert torch.allclose(terms["joint_limit"], torch.tensor([-0.004]), atol=1e-7)
    assert torch.equal(reward, torch.stack(list(terms.values()), -1).sum(-1))


def test_root_xy_reanchoring_removes_local_body_translation_error():
    lib = bank((5,))
    t = task(lib)
    rank, frame = torch.tensor([0]), torch.tensor([1])
    state = lib.ref_state(rank, frame).clone()
    state[:, 0] += 0.3
    zero = torch.zeros(1, 29)
    _, terms = t.mdp.reward(state, zero, zero, rank, frame)
    assert torch.allclose(terms["body_pos"], torch.tensor([0.02]), atol=1e-7)
    assert torch.allclose(
        terms["root_pos"], torch.tensor([0.01 * math.exp(-1)]), atol=1e-7
    )


def test_transition_reward_uses_next_reference_frame():
    lib = bank((5,), moving=True)
    t = task(lib)
    rank, frame = torch.tensor([0]), torch.tensor([1])
    next_state = lib.ref_state(rank, frame + 1)
    zero, xi = torch.zeros(1, 29), torch.zeros(1, 33)
    _, terms = t.reward(next_state, zero, zero, xi, rank, frame)
    assert torch.allclose(terms["root_pos"], torch.tensor([0.01]), atol=1e-7)
    _, old = t.mdp.reward(next_state, zero, zero, rank, frame)
    assert old["root_pos"].item() < terms["root_pos"].item()


def test_termination_strict_thresholds_soft_probability_and_clip_end():
    lib = bank((5,))
    t = task(lib)
    rank, frame = torch.tensor([0, 0, 0]), torch.tensor([1, 1, 1])
    state = lib.ref_state(rank, frame).clone()
    # Use exactly representable base height to test strict > 0.25.
    lib.states[:, 2] = 0.75
    state[:, 2] = torch.tensor([1.0, 1.0001, 0.75])
    state[2, 3:7] = quat_exp(torch.tensor([math.acos(0.19), 0.0, 0.0]))
    hard = t.mdp.termination(state, rank, frame, hard=True)
    assert hard.tolist() == [False, True, True]
    probability = t.mdp.termination(state[:1], rank[:1], frame[:1])
    assert torch.allclose(probability, torch.tensor([0.5]), atol=1e-6, rtol=1e-5)
    # Clip end is separate from physical termination.
    end = lib.ref_state(rank[:1], torch.tensor([4]))
    assert not t.mdp.termination(end, rank[:1], torch.tensor([4]), hard=True).item()
    assert t.clip_end(rank[:1], torch.tensor([4])).item()


def test_imagination_start_needs_full_horizon_with_final_frame_included():
    t = task(bank((5, 8)))
    rank = torch.tensor([0, 0, 0, 1, 1])
    frame = torch.tensor([2, 3, 4, 5, 6])
    assert t.valid_imagination_start(rank, frame, 2).tolist() == [
        True,
        False,
        False,
        True,
        False,
    ]
    # The filter also handles batched replay-row scans.
    assert t.valid_imagination_start(rank[None], frame[None], 2).shape == (1, 5)


def test_context_normalization_and_critic_dimensions():
    lib = bank((5,))
    t = task(lib)
    xi = torch.cat(
        (torch.tensor([[1.2]]), torch.full((1, 29), 0.01), torch.full((1, 3), 0.05)), -1
    )
    assert torch.allclose(
        normalize_context(xi), torch.ones(1, 33), atol=1e-6, rtol=1e-5
    )
    rank, frame = torch.tensor([0]), torch.tensor([1])
    state = lib.ref_state(rank, frame)
    value = t.critic_obs(state, xi, rank, frame)
    assert value.shape == (1, 914)
    assert t.critic_obs_dim == 914
    assert torch.equal(value[:, 151:184], normalize_context(xi))
    assert t.wm_input(state, torch.zeros(1, 29), xi).shape == (1, 133)


def test_persistent_noise_repeats_reset_draw_and_survives_shift():
    t = task(bank())
    env, step = torch.tensor([2, 7]), torch.tensor([100, 100])
    noise = t.keyed_proprio_noise(123, env, step, episode_step=torch.tensor([0, 20]))
    assert torch.equal(noise[0], noise[0, -1].expand_as(noise[0]))
    nxt = t.keyed_proprio_noise(123, env, step + 1, episode_step=torch.tensor([1, 21]))
    assert torch.equal(noise[:, 1:], nxt[:, :-1])
    assert torch.all(noise.abs() <= torch.tensor(OBS_NOISE_WIDTH))
    assert torch.count_nonzero(noise[..., 64:93]) == 0


def test_deployed_policy_uses_recorded_noisy_history_without_redrawing():
    lib = bank()
    t = task(lib)
    rank = torch.tensor([0])
    states = lib.ref_state(rank, torch.arange(1, 11).unsqueeze(0))
    prop = torch.randn(1, 990)
    inputs = (
        states[..., :13].flatten(1),
        states[..., 13:].flatten(1),
        torch.zeros(1, 290),
        torch.tensor([[0.0, 10.0, 10.0]]),
        torch.zeros(1, 33),
        prop,
    )
    first = t.policy_obs_from_env(*inputs)
    second = t.policy_obs_from_env(*inputs)
    assert torch.equal(first[:, :990], prop)
    assert torch.equal(first, second)


def test_policy_history_uses_each_reference_frame_and_reset_fill():
    lib = bank((20,), moving=True)
    t = task(lib, noise=False)
    rank, frame, ep_step = torch.tensor([0]), torch.tensor([12]), torch.tensor([2])
    hist_frame = torch.tensor([[10] * 8 + [11, 12]])
    states = lib.ref_state(rank, hist_frame).clone()
    # Hold robot root rotation fixed to expose historical reference rotations.
    states[..., 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    actions, xi = torch.zeros(1, 10, 29), torch.zeros(1, 33)
    obs = t.policy_obs(states, actions, xi, rank, frame, episode_step=ep_step)
    assert obs.shape == (1, 1720)
    prop = obs[:, :990].reshape(1, 10, 99)
    assert torch.equal(prop[:, :8], prop[:, :1].expand(1, 8, 99))
    for slot, f in enumerate(hist_frame[0]):
        angle = 0.5 * float(f) * 0.02
        expected = torch.tensor(
            [math.cos(angle), -math.sin(angle), math.sin(angle), math.cos(angle), 0, 0]
        )
        assert torch.allclose(prop[0, slot, 93:], expected, atol=1e-6, rtol=1e-5)


def test_reward_and_soft_termination_have_finite_action_path_gradients():
    t = task(bank())
    rank, frame = torch.tensor([0]), torch.tensor([1])
    state = t.library.ref_state(rank, frame).clone().requires_grad_(True)
    action = torch.randn(1, 29, requires_grad=True) * 0.01
    reward, death = t.reward_and_termination(
        state, action, torch.zeros_like(action), torch.zeros(1, 33), rank, frame
    )
    gs, ga = torch.autograd.grad((reward * (1 - death)).sum(), (state, action))
    assert torch.isfinite(gs).all() and torch.isfinite(ga).all()
    assert ga.norm() > 0


def test_rotation_log_near_pi_and_reward_rotation_floor():
    vector = torch.tensor([[math.pi - 1e-5, 0.0, 0.0]], dtype=torch.float64)
    assert torch.allclose(rotation_vector(so3_exp(vector)), vector, atol=1e-8)
    identity = torch.eye(3, dtype=torch.float64).unsqueeze(0)
    assert torch.allclose(
        rotation_error_sq(identity, identity),
        torch.tensor([math.acos(1 - 1e-6) ** 2], dtype=torch.float64),
    )


def test_dynamics_coordinates_roundtrip_and_half_turn():
    t = task(bank())
    state = t.library.ref_state(torch.tensor([0]), torch.tensor([1])).double()
    delta = torch.randn(1, 70, dtype=torch.float64) * 0.01
    nxt = t.integrate(state, delta)
    assert torch.allclose(t.increment(state, nxt), delta, atol=1e-8)
    assert torch.allclose(t.state_error(nxt, state), delta, atol=1e-8)
    half_turn = state.clone()
    half_turn[:, 3:7] = quat_exp(torch.tensor([[math.pi, 0, 0]], dtype=torch.float64))
    assert torch.allclose(
        t.state_error(half_turn, state)[:, 3:6],
        torch.tensor([[math.pi, 0, 0]], dtype=torch.float64),
        atol=1e-8,
    )


def test_invalid_reference_bank_and_goal_contract_are_rejected():
    with pytest.raises(ValueError, match="at least two"):
        bank((1,))
    lib = bank()
    t = task(lib)
    with pytest.raises(ValueError, match="explicit 10-frame"):
        G1TrackingTask(t.isaac_env, "cpu", goal_mode="z", skill_encoder="not-used.pt")
    assert tuple(G1_ISAACLAB_JOINT_NAMES) == tuple(t.kin.joint_names)


def test_latent_encodes_next_frame_and_keeps_state_gradients(tmp_path):
    from imitationlab.encoders.diffsr.pretrain import HighLevelSkillDiffSRConfig
    from imitationlab.encoders.skill_encoder import build_skill_encoder
    from imitationlab.envs.isaaclab.task_model.g1_task import G1LatentTrackingTask

    cfg = HighLevelSkillDiffSRConfig(
        horizon_steps=4,
        z_dim=8,
        encoder_hidden_dims=(16,),
        encoder_window_mode="intermediate",
        macro_anchor_mode="robot_heading",
        macro_frame_stride=1,
        source_anchor="current",
    )
    encoder = build_skill_encoder(
        state_dim=38,
        window_steps=3,
        z_dim=8,
        hidden_dims=(16,),
        spec=cfg.latent_spec(),
        activation=cfg.encoder_activation,
        layer_norm=cfg.encoder_layer_norm,
    )
    path = tmp_path / "encoder.pt"
    torch.save(
        {"config": cfg.to_dict(), "skill_encoder_state_dict": encoder.state_dict()},
        path,
    )
    explicit = task(bank((20, 5), moving=True), noise=False)
    latent = G1LatentTrackingTask(explicit.isaac_env, "cpu", skill_encoder=str(path))
    rank, frame = torch.tensor([0, 1]), torch.tensor([1, 3])
    state = latent.library.ref_state(rank, frame).clone().requires_grad_(True)
    frames = latent.skill_frames(state, rank, frame)
    expected = latent.library.gather(rank, frame[:, None] + torch.arange(1, 5))
    assert torch.equal(frames[..., :29], expected.joint_pos)
    # The row-only frames (no forward kinematics) are the generic path's, bit for bit.
    assert latent.skill_anchor == "pelvis"
    assert torch.equal(frames, latent._skill_frames_fk(state, rank, frame + 1))
    assert latent.skill_bf16 is False and latent.describe()["skill_bf16"] is False
    with_bf16 = G1LatentTrackingTask(
        explicit.isaac_env, "cpu", skill_encoder=str(path), skill_bf16=True
    )
    assert with_bf16.describe()["skill_bf16"] is True
    # A training precision, not a binding: the fp32-encoder evaluator loads it.
    assert with_bf16.goal_binding() == latent.goal_binding()
    # Off the GPU the autocast is disabled: the goal is the fp32 one.
    assert torch.equal(with_bf16.goal(state, rank, frame), latent.skill(frames))
    # The short clip must hold its final pose, never leak into another clip.
    assert torch.equal(frames[1, :1, :29].expand(4, 29), frames[1, :, :29])
    goal = latent.goal(state, rank, frame)
    assert goal.shape == (2, 10)
    assert torch.equal(goal, latent.skill(frames))
    history = torch.zeros(2, 10, 99)
    assert torch.equal(
        latent.policy_obs_from_parts(history, state, rank, frame)[:, -10:], goal
    )
    assert torch.equal(
        latent.critic_obs(state, torch.zeros(2, 33), rank, frame)[:, -10:], goal
    )
    assert latent.policy_obs_dim == 1000
    assert latent.critic_obs_dim == 194
    goal.sum().backward()
    assert state.grad is not None and state.grad.abs().sum() > 0
    assert all(
        p.grad is None and not p.requires_grad
        for p in latent.skill.encoder.parameters()
    )
    action = torch.zeros(2, 29)
    assert torch.equal(
        latent.reward(state, action, action, torch.zeros(2, 33), rank, frame)[0],
        explicit.reward(state, action, action, torch.zeros(2, 33), rank, frame)[0],
    )
    assert latent.goal_binding()["reference_start_offset"] == 1


@pytest.mark.parametrize(
    "origin,frame_dim,qvel_scale",
    [("xy", 38, 1.0), ("xy", 67, 0.1), ("xyz", 38, 1.0)],
)
def test_reference_anchored_window_reads_nothing_from_the_robot(
    tmp_path, origin, frame_dim, qvel_scale
):
    """``rollout_anchor=previous_reference``: the window f+1..f+F in the yaw
    frame of reference frame f, at its xy (``xy``) or its full position (``xyz``).
    Expected values are written out per coordinate from the reference rows."""
    from imitationlab.encoders.diffsr.pretrain import HighLevelSkillDiffSRConfig
    from imitationlab.encoders.skill_encoder import build_skill_encoder
    from imitationlab.envs.isaaclab.task_model.g1_task import G1LatentTrackingTask

    cfg = HighLevelSkillDiffSRConfig(
        horizon_steps=4,
        z_dim=8,
        encoder_hidden_dims=(16,),
        encoder_window_mode="intermediate",
        macro_anchor_mode="robot_heading",
        macro_frame_stride=1,
        source_anchor="current",
        macro_origin=origin,
        rollout_anchor="previous_reference",
        # A velocity scale is only valid on the recipe's objective.
        **(
            {}
            if qvel_scale == 1.0
            else {
                "macro_qvel_scale": qvel_scale,
                "transition_objective": "jepa_ntp",
                "source_history_steps": 2,
            }
        ),
    )
    values = cfg.to_dict()
    encoder = build_skill_encoder(
        state_dim=frame_dim,
        window_steps=3,
        z_dim=8,
        hidden_dims=(16,),
        spec=cfg.latent_spec(),
        activation=cfg.encoder_activation,
        layer_norm=cfg.encoder_layer_norm,
    )
    path = tmp_path / "encoder.pt"
    torch.save(
        {"config": values, "skill_encoder_state_dict": encoder.state_dict()}, path
    )
    explicit = task(bank((20, 5), moving=True), noise=False)
    latent = G1LatentTrackingTask(explicit.isaac_env, "cpu", skill_encoder=str(path))
    rank, frame = torch.tensor([0, 1]), torch.tensor([1, 3])
    lib = latent.library
    state = lib.ref_state(rank, frame).clone()
    frames = latent.skill_frames(state, rank, frame)
    assert frames.shape == (2, 4, frame_dim)

    # Moving and turning the robot changes nothing.
    moved = state.clone()
    moved[:, :3] += torch.tensor([0.7, -0.4, 0.2])
    moved[:, 3:7] = torch.nn.functional.normalize(
        moved[:, 3:7] + torch.tensor([0.0, 0.1, -0.2, 0.5]), dim=-1
    )
    assert torch.equal(latent.skill_frames(moved, rank, frame), frames)

    current = lib.ref_state(rank, frame)
    future = lib.ref_state(rank, frame[:, None] + torch.arange(1, 5))
    w, z = current[:, 3], current[:, 6]
    yaw = 2 * torch.atan2(z, w)
    cos, sin = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
    dx = future[..., 0] - current[:, None, 0]
    dy = future[..., 1] - current[:, None, 1]
    height = future[..., 2] - (current[:, None, 2] if origin == "xyz" else 0.0)
    joints = 29 if frame_dim == 38 else 58
    assert torch.equal(frames[..., :29], future[..., 13:42])
    if frame_dim == 67:
        torch.testing.assert_close(frames[..., 29:58], future[..., 42:71] * qvel_scale)
    torch.testing.assert_close(
        frames[..., joints], cos * dx + sin * dy, atol=1e-6, rtol=1e-5
    )
    torch.testing.assert_close(
        frames[..., joints + 1], -sin * dx + cos * dy, atol=1e-6, rtol=1e-5
    )
    torch.testing.assert_close(frames[..., joints + 2], height, atol=1e-6, rtol=1e-5)
    binding = latent.goal_binding()
    assert binding["encoder_rollout_anchor"] == "previous_reference"
    assert binding["encoder_frame_dim"] == frame_dim
    assert ("encoder_origin" in binding) == (origin == "xyz")


def test_first_release_encoders_keep_their_goal_binding(tmp_path):
    from imitationlab.encoders.diffsr.pretrain import HighLevelSkillDiffSRConfig

    values = HighLevelSkillDiffSRConfig(
        horizon_steps=4, z_dim=8, encoder_hidden_dims=(16,)
    ).to_dict()
    # A record written before these fields existed loads with the first
    # release's behaviour.
    values.pop("macro_origin"), values.pop("rollout_anchor")
    restored = HighLevelSkillDiffSRConfig.from_dict(values)
    assert (restored.macro_origin, restored.rollout_anchor) == ("xy", "robot")
