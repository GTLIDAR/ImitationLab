"""imitation-g1-v1 protocol pins and lifecycle proofs without starting the simulator."""

from types import SimpleNamespace
from unittest.mock import MagicMock
import re

import gymnasium as gym
import pytest
import torch

from imitationlab.envs.isaaclab.envs.g1_env import G1RLEnv
from imitationlab.envs.isaaclab.tasks.manager_based.imitation.config.g1.imitation_g1_env import (
    ImitationG1EnvCfg,
)
from imitationlab.envs.isaaclab.tasks.manager_based.imitation.mdp import g1
from imitationlab.envs.isaaclab.task_model.g1_kinematics import G1_ISAACLAB_JOINT_NAMES


def _readout_cfg() -> ImitationG1EnvCfg:
    """The settings ``imitationlab evaluate`` applies: pinned clips, clean physics."""
    return ImitationG1EnvCfg(
        pinned_reset=True,
        reset_noise=False,
        observation_noise=False,
        randomize_context=False,
    )


@pytest.mark.parametrize("encoder", ["", "encoder.pt"])
def test_public_g1_task_binds_wmbptt_to_the_configured_interface(encoder):
    from imitationlab.envs.isaaclab.tasks.manager_based.imitation.command_interface import (
        bind_command_interface,
    )
    from imitationlab.envs.isaaclab.tasks.manager_based.imitation.config.g1.agents.wmbptt_cfg import (
        EXPLICIT_TASK,
        LATENT_TASK,
        G1WMBPTTConfig,
    )

    spec = gym.spec("imitation-g1-v1")
    assert spec.kwargs["env_cfg_entry_point"].endswith(":ImitationG1EnvCfg")
    cfg = ImitationG1EnvCfg(skill_encoder_path=encoder)
    agent = G1WMBPTTConfig()
    bind_command_interface(agent, cfg)
    expected = (LATENT_TASK, "z") if encoder else (EXPLICIT_TASK, "explicit")
    assert (agent.wm.task_model, agent.wm.goal) == expected


@pytest.mark.parametrize("step_mode", ["masked", "graph"])
def test_the_eager_env_refuses_a_step_mode_it_would_ignore(step_mode):
    """``G1RLEnv`` is the eager step; ``G1GraphRLEnv`` runs the masked and graph steps."""
    with pytest.raises(ValueError, match="step_mode"):
        G1RLEnv(SimpleNamespace(step_mode=step_mode))


def test_robot_and_physics_are_the_replica_contract():
    cfg = ImitationG1EnvCfg()
    cfg.resolve()
    assert cfg.sim.dt == 0.02 and cfg.decimation == 1
    assert cfg.sim.physics.num_substeps == 4
    solver = cfg.sim.physics.solver_cfg
    assert (solver.njmax, solver.nconmax, solver.cone, solver.integrator) == (
        500,
        128,
        "pyramidal",
        "implicitfast",
    )
    assert solver.use_mujoco_contacts
    assert not cfg.scene.robot.spawn.articulation_props.enabled_self_collisions
    assert cfg.scene.robot.soft_joint_pos_limit_factor == 0.9
    assert cfg.scene.terrain.physics_material.static_friction == 0.3
    assert cfg.scene.terrain.physics_material.dynamic_friction == 0.3
    assert cfg.episode_length_s == 10 and not cfg.is_finite_horizon
    assert cfg.curriculum is None
    assert cfg.command_interface.reference.anchor_body_name == "pelvis"
    assert cfg.data.resolved_reference_prefetch_mode() == "next"
    assert cfg.terminations.reference_finished.time_out
    assert cfg.terminations.time_out.time_out
    assert not cfg.terminations.tracking_failed.time_out
    # Independently specified motor table from the replica, by anatomical group.
    motors = {
        "hip_pitch": (88, 32, 0.010177520),
        "hip_roll": (139, 20, 0.025101925),
        "hip_yaw": (88, 32, 0.010177520),
        "knee": (139, 20, 0.025101925),
        "ankle_pitch": (50, 37, 0.007219450),
        "ankle_roll": (50, 37, 0.007219450),
        "waist_yaw": (88, 32, 0.010177520),
        "waist_roll": (50, 37, 0.007219450),
        "waist_pitch": (50, 37, 0.007219450),
        "shoulder_pitch": (25, 37, 0.003609725),
        "shoulder_roll": (25, 37, 0.003609725),
        "shoulder_yaw": (25, 37, 0.003609725),
        "elbow": (25, 37, 0.003609725),
        "wrist_roll": (25, 37, 0.003609725),
        "wrist_pitch": (5, 22, 0.00425),
        "wrist_yaw": (5, 22, 0.00425),
    }
    natural_frequency = 10 * 2 * 3.1415926535

    def value_for(value, name):
        if not isinstance(value, dict):
            return value
        return next(v for pat, v in value.items() if re.fullmatch(pat, name))

    for name in G1_ISAACLAB_JOINT_NAMES:
        key = name.replace("left_", "").replace("right_", "").replace("_joint", "")
        effort, velocity, armature = motors[key]
        actuator = next(
            a
            for a in cfg.scene.robot.actuators.values()
            if any(re.fullmatch(pat, name) for pat in a.joint_names_expr)
        )
        assert value_for(actuator.joint_effort_limit, name) == effort
        assert value_for(actuator.joint_velocity_limit, name) == velocity
        assert value_for(actuator.actuator_velocity_limit, name) == velocity
        assert value_for(actuator.armature, name) == pytest.approx(armature)
        assert value_for(actuator.stiffness, name) == pytest.approx(
            armature * natural_frequency**2
        )
        assert value_for(actuator.damping, name) == pytest.approx(
            4 * armature * natural_frequency
        )
        assert value_for(cfg.actions.joint_pos.scale, name) == pytest.approx(
            0.25 * effort / (armature * natural_frequency**2)
        )


def test_the_task_rejects_a_changed_control_rate():
    cfg = ImitationG1EnvCfg()
    cfg.sim.dt = 0.01
    with pytest.raises(ValueError, match="0.02"):
        cfg.resolve()


@pytest.mark.parametrize("mode", ["off", "next", " NEXT "])
def test_prefetch_uses_the_nested_data_setting(mode):
    cfg = ImitationG1EnvCfg()
    cfg.data.reference_prefetch_mode = mode
    cfg.resolve()
    assert cfg.data.resolved_reference_prefetch_mode() == mode.strip().lower()


@pytest.mark.parametrize(
    "mode", ["next_and_reset", " next-and-reset ", "NEXT_AND_RESET"]
)
def test_the_task_rejects_nested_predictive_reset_prefetch(mode):
    cfg = ImitationG1EnvCfg()
    cfg.data.reference_prefetch_mode = mode
    with pytest.raises(ValueError, match="current returns"):
        cfg.resolve()


def test_manager_reward_uses_live_bodies_and_exact_current_target():
    state, body = object(), object()
    ranks, frames = torch.tensor([2]), torch.tensor([19])
    reward = torch.tensor([0.08])
    mdp = SimpleNamespace(reward=MagicMock(return_value=(reward, {"root_pos": reward})))
    env = SimpleNamespace(
        g1_cursor=lambda: (ranks, frames),
        g1_state=lambda: state,
        g1_body_state=lambda: body,
        g1_mdp=mdp,
        step_dt=0.02,
        action_manager=SimpleNamespace(action="action", prev_action="previous"),
    )
    assert torch.equal(g1.replica_reward(env), torch.tensor([4.0]))
    mdp.reward.assert_called_once_with(
        state, "action", "previous", ranks, frames, body_state=body
    )
    assert env._g1_reward_parts == {"root_pos": reward}


@pytest.mark.parametrize("wm_only", [False, True])
@pytest.mark.parametrize("manual_reset", [False, True])
def test_step_installs_f_plus_one_and_captures_terminal_history_before_reset(
    manual_reset, wm_only
):
    """Natural and UI endings retain the same true terminal-state boundary."""
    calls = []
    frame = torch.tensor([3, 8, 5])
    current_state = torch.tensor([[10.0], [20.0], [30.0]])
    env = SimpleNamespace(
        device="cpu",
        cfg=SimpleNamespace(
            decimation=1,
            sim=SimpleNamespace(render_interval=1),
            wm_only_observations=wm_only,
        ),
        _physics_handles_decimation=True,
        _sim_step_counter=0,
        step_dt=0.02,
        episode_length_buf=torch.tensor([2, 7, 4]),
        common_step_counter=0,
        render_enabled=False,
        has_rtx_sensors=False,
        extras={},
        video_recorders=[],
        _g1_reward_parts={},
        g1_library=SimpleNamespace(env_win_length=0),
    )
    env.expert_data_plane = SimpleNamespace(
        begin_next_reference=lambda: calls.append("begin"),
        finish_next_reference=lambda ids: (calls.append("install_f+1"), frame.add_(1)),
    )
    env.action_manager = SimpleNamespace(
        process_action=lambda a: calls.append("action"), apply_action=lambda: None
    )
    env.scene = SimpleNamespace(write_data_to_sim=lambda: None, update=lambda dt: None)
    env.sim = SimpleNamespace(
        is_rendering=False,
        step=lambda **kw: (calls.append("physics"), current_state.add_(1)),
        consume_reset_request=MagicMock(return_value=manual_reset),
    )
    env.recorder_manager = SimpleNamespace(
        record_pre_step=lambda: None,
        record_post_physics_decimation_step=lambda: None,
        active_terms=[],
    )

    def terminate():
        assert frame.tolist() == [4, 9, 6]
        calls.append("termination")
        return torch.tensor([True, True, False])

    env.termination_manager = SimpleNamespace(
        compute=terminate,
        terminated=torch.tensor([True, False, False]),
        time_outs=torch.tensor([False, True, False]),
    )

    def reward(dt):
        assert frame.tolist() == [4, 9, 6]
        calls.append("reward")
        return torch.tensor([0.1, 0.2, 0.3])

    env.reward_manager = SimpleNamespace(compute=reward)
    env.g1_reset_sampler = SimpleNamespace(
        accumulate=lambda r: calls.append("accumulate")
    )
    env._g1_append_history = lambda: calls.append("append_history")
    computed = []

    def wm_group():
        return {"state": current_state.clone(), "frame": frame.clone()}

    def compute(update_history):
        computed.append(("all", update_history))
        return {"wm": wm_group(), "policy": {"features": frame.clone()}}

    def compute_group(group, update_history):
        computed.append((group, update_history))
        return wm_group()

    env.observation_manager = SimpleNamespace(
        compute=compute, compute_group=compute_group
    )
    env._g1_observations = lambda: G1RLEnv._g1_observations(env)

    def publish(ids):
        calls.append("terminal_obs")
        G1RLEnv._publish_terminal_obs(env, ids)

    env._publish_terminal_obs = publish

    def reset(ids):
        calls.append("reset")
        assert ids.tolist() == ([0, 1, 2] if manual_reset else [0, 1])
        assert env.reset_terminated.tolist() == [True, False, False]
        assert env.reset_time_outs.tolist() == [False, True, manual_reset]
        current_state[ids] = -1
        frame[ids] = 0

    env._reset_idx = reset
    env.command_manager = SimpleNamespace(compute=lambda dt: None)
    env._append_causal_planner_history = lambda: None
    obs, _, terminated, truncated, extras = G1RLEnv.step(env, torch.zeros(3, 29))
    assert calls == [
        "begin",
        "action",
        "physics",
        "install_f+1",
        "termination",
        "reward",
        "accumulate",
        "append_history",
        "terminal_obs",
        "reset",
    ]
    final = extras["final_obs"]
    assert final["_env_ids"].tolist() == ([0, 1, 2] if manual_reset else [0, 1])
    assert final["obs"]["wm"]["state"].tolist() == (
        [[11.0], [21.0], [31.0]] if manual_reset else [[11.0], [21.0]]
    )
    assert final["obs"]["wm"]["frame"].tolist() == (
        [4, 9, 6] if manual_reset else [4, 9]
    )
    assert obs["wm"]["state"].tolist() == [
        [-1.0],
        [-1.0],
        [-1.0 if manual_reset else 31.0],
    ]
    assert obs["wm"]["frame"].tolist() == [0, 0, 0 if manual_reset else 6]
    # Both points of the step, and never a second group in this mode.
    assert computed == [("wm" if wm_only else "all", False)] * 2
    assert set(final["obs"]) == ({"wm"} if wm_only else {"wm", "policy"})
    assert set(obs) == ({"wm"} if wm_only else {"wm", "policy"})
    assert terminated.tolist() == [True, False, False]
    assert truncated.tolist() == [False, True, manual_reset]
    env.sim.consume_reset_request.assert_called_once_with()


def _env_with_groups(wm_only: bool) -> G1RLEnv:
    env = G1RLEnv.__new__(G1RLEnv)
    env.cfg = SimpleNamespace(wm_only_observations=wm_only)
    env.scene = SimpleNamespace(num_envs=1)
    return env


def test_wm_only_observations_defaults_off():
    assert ImitationG1EnvCfg().wm_only_observations is False
    assert _readout_cfg().wm_only_observations is False


def test_wm_only_observations_needs_the_wm_group():
    cfg = ImitationG1EnvCfg()
    cfg.wm_only_observations = True
    cfg.resolve()
    cfg.observations.wm = None
    with pytest.raises(ValueError, match="wm_only_observations"):
        cfg.resolve()


def test_qualification_tools_refuse_wm_only_observations():
    from imitationlab.evaluation.world_model.prefetch_validation import (
        capture_trace,
    )
    from imitationlab.evaluation.world_model.g1_env_validation import (
        qualify_g1_environment,
    )

    assert _env_with_groups(False).terminal_observation_is_post_step is True
    env = _env_with_groups(True)
    with pytest.raises(ValueError, match="wm_only_observations"):
        capture_trace(env, steps=1, seed=0, action_std=0.1)
    with pytest.raises(ValueError, match="wm_only_observations"):
        qualify_g1_environment(env)


def test_torchrl_wrapper_refuses_a_step_without_the_policy_groups():
    from imitationlab.envs.isaaclab.envs.torchrl_adapter import IsaacLabWrapper

    wrapper = SimpleNamespace(_base_isaac_env=lambda: _env_with_groups(True))
    with pytest.raises(ValueError, match="WMBPTT-only"):
        IsaacLabWrapper._step(wrapper, None)


def test_reset_fail_share_defaults_off_and_rejects_a_useless_setting():
    assert ImitationG1EnvCfg().reset_fail_share == 0.0
    cfg = ImitationG1EnvCfg()
    cfg.reset_fail_share = 0.3
    cfg.resolve()
    for share in (1.0, -0.1):
        cfg.reset_fail_share = share
        with pytest.raises(ValueError, match="reset_fail_share"):
            cfg.resolve()
    readout = _readout_cfg()
    readout.reset_fail_share = 0.3
    with pytest.raises(ValueError, match="pinned_reset"):
        readout.resolve()


def test_reward_kernel_scale_defaults_to_one_and_must_be_positive():
    assert ImitationG1EnvCfg().reward_kernel_scale == 1.0
    cfg = ImitationG1EnvCfg()
    cfg.reward_kernel_scale = 0.7
    cfg.resolve()
    for scale in (0.0, -0.7, float("nan")):
        cfg.reward_kernel_scale = scale
        with pytest.raises(ValueError, match="reward_kernel_scale"):
            cfg.resolve()


def test_a_pinned_readout_keeps_the_task_kernels():
    """The replica readout is `pinned_reset` on the task, scored with its own kernels."""
    cfg = _readout_cfg()
    cfg.resolve()
    cfg.reward_kernel_scale = 0.7
    with pytest.raises(ValueError, match="own kernels"):
        cfg.resolve()


@pytest.mark.parametrize(
    ("mdp", "scale", "refused"),
    [
        (SimpleNamespace(), 1.0, False),  # an MDP without the field is at scale 1
        (SimpleNamespace(), 0.7, True),  # no silent no-op
        (SimpleNamespace(reward_kernel_scale=0.7), 0.7, False),
        (SimpleNamespace(reward_kernel_scale=1.0), 0.7, True),
    ],
)
def test_env_refuses_a_scale_the_shared_mdp_did_not_take(mdp, scale, refused):
    env = SimpleNamespace(g1_mdp=mdp, cfg=SimpleNamespace(reward_kernel_scale=scale))
    if refused:
        with pytest.raises(ValueError, match="reward_kernel_scale"):
            G1RLEnv._g1_check_reward_kernels(env)
    else:
        G1RLEnv._g1_check_reward_kernels(env)


def test_heading_velocity_reward_is_off_by_default_and_must_reach_the_mdp():
    assert ImitationG1EnvCfg().reward_heading_velocity is False
    cfg = SimpleNamespace(reward_kernel_scale=1.0, reward_heading_velocity=True)
    with pytest.raises(ValueError, match="reward_heading_velocity"):
        G1RLEnv._g1_check_reward_kernels(
            SimpleNamespace(g1_mdp=SimpleNamespace(), cfg=cfg)
        )
    mdp = SimpleNamespace(reward_heading_velocity=True)
    G1RLEnv._g1_check_reward_kernels(SimpleNamespace(g1_mdp=mdp, cfg=cfg))


def test_a_pinned_readout_keeps_world_frame_velocity_targets():
    cfg = _readout_cfg()
    cfg.resolve()
    cfg.reward_heading_velocity = True
    with pytest.raises(ValueError, match="reward_heading_velocity"):
        cfg.resolve()


def test_persist_reset_stats_defaults_off_and_rejects_a_useless_setting():
    assert ImitationG1EnvCfg().persist_reset_stats is False
    readout = _readout_cfg()
    readout.persist_reset_stats = True
    with pytest.raises(ValueError, match="persist_reset_stats"):
        readout.resolve()
    cfg = ImitationG1EnvCfg()
    cfg.persist_reset_stats = True
    cfg.resolve()


def test_reset_noise_widths_are_the_replica_table():
    from imitationlab.envs.isaaclab.envs.g1_env import RESET_NOISE_WIDTHS

    assert RESET_NOISE_WIDTHS == {
        "position": (0.05, 0.05, 0.01),
        "rotation": (0.1, 0.1, 0.2),
        "linear_velocity": (0.5, 0.5, 0.2),
        "angular_velocity": (0.52, 0.52, 0.78),
    }


def test_the_newton_selector_keeps_the_replica_solver_and_rejects_drift():
    cfg = ImitationG1EnvCfg()
    cfg.sim.physics = cfg.sim.physics.newton_mjwarp
    cfg.resolve()
    assert cfg.sim.physics.solver_cfg.njmax == 500
    assert cfg.sim.physics.solver_cfg.nconmax == 128
    assert cfg.sim.physics.solver_cfg.tolerance == 1e-6
    cfg.sim.physics.solver_cfg.nconmax = 200
    with pytest.raises(ValueError, match="nconmax=128"):
        cfg.resolve()


def test_the_agent_uses_the_replica_action_clip_and_persistent_proprio_history():
    from imitationlab.envs.isaaclab.tasks.manager_based.imitation.config.g1.agents.wmbptt_cfg import (
        G1WMBPTTConfig,
    )

    cfg = G1WMBPTTConfig()
    assert cfg.wm.action_clip == 20.0
    assert cfg.wm.goal == "explicit"
    assert cfg.policy.input_keys[-1] == ("wm", "proprio_history")
    assert cfg.wm.task_model.endswith(":G1TrackingTask")


def test_ppo_reads_the_flat_feature_groups():
    from imitationlab.envs.isaaclab.tasks.manager_based.imitation.config.g1.agents.ppo_cfg import (
        G1PPOConfig,
    )

    cfg = G1PPOConfig()
    assert cfg.policy.input_keys == [("policy", "features")]
    assert cfg.value_function.input_keys == [("critic", "features")]
    # The packed-data recipe (experiments/recipes/g1/ppo.args) adds normalization.
    assert not cfg.policy.normalize_input
    assert cfg.policy.num_cells == [512, 256, 128]
    assert cfg.ppo.actor_learning_rate is None
    assert cfg.loss.gamma == ImitationG1EnvCfg().reset_discount


def test_latent_live_observations_share_the_rollout_task(monkeypatch):
    from imitationlab.envs.isaaclab.tasks.manager_based.imitation.config.g1.agents.ppo_cfg import (
        G1PPOConfig,
    )
    from imitationlab.envs.isaaclab.tasks.manager_based.imitation.config.g1.agents.wmbptt_cfg import (
        G1WMBPTTConfig,
    )

    # The encoder file is read only to pin its hash; the switch is the path.
    monkeypatch.setattr(ImitationG1EnvCfg, "_bind_skill_encoder", lambda self: None)
    cfg = ImitationG1EnvCfg(skill_encoder_path="encoder.pt")
    cfg.resolve()
    assert cfg.observations.policy.features.func is g1.latent_policy_features
    assert cfg.observations.critic.features.func is g1.latent_critic_features
    assert cfg.data.resolved_reference_prefetch_mode() == "next"
    assert cfg.sim.physics.num_substeps == 4
    ppo = G1PPOConfig()
    wm = G1WMBPTTConfig()
    wm.bind_env_config(cfg)
    assert ppo.policy.input_keys == [("policy", "features")]
    assert ppo.value_function.input_keys == [("critic", "features")]
    assert wm.wm.goal == "z" and wm.wm.task_model.endswith(":G1LatentTrackingTask")
    env = SimpleNamespace(
        num_envs=2, device="cpu", g1_skill=SimpleNamespace(command_dim=10)
    )
    assert g1.latent_policy_features(env).shape == (2, 1000)
    assert g1.latent_critic_features(env).shape == (2, 194)
    env.g1_task = SimpleNamespace(
        policy_obs_from_parts=MagicMock(return_value=torch.ones(2, 1000))
    )
    env._g1_proprio = torch.zeros(2, 10, 99)
    env.g1_cursor = lambda: (torch.tensor([0, 1]), torch.tensor([4, 8]))
    env.g1_state = lambda: torch.zeros(2, 71)
    assert torch.equal(g1.latent_policy_features(env), torch.ones(2, 1000))
    args = env.g1_task.policy_obs_from_parts.call_args.args
    assert torch.equal(
        args[-1], torch.tensor([4, 8])
    )  # task owns the +1, no double shift


def test_the_readout_settings_leave_training_alone():
    readout = _readout_cfg()
    assert readout.pinned_reset
    assert not (
        readout.reset_noise or readout.observation_noise or readout.randomize_context
    )
    training = ImitationG1EnvCfg()
    assert training.reset_noise and training.observation_noise
    assert training.randomize_context and not training.pinned_reset
    assert training.terminations.tracking_failed is not None


def _pinned_env(custom_fn):
    selection = SimpleNamespace(custom_fn=custom_fn, start_frame=0)
    return SimpleNamespace(
        cfg=SimpleNamespace(
            pinned_reset=True,
            command_interface=SimpleNamespace(
                reference=SimpleNamespace(selection=selection)
            ),
        ),
        device="cpu",
        _g1_explicit_starts=None,
        g1_library=SimpleNamespace(length=torch.tensor([5, 6, 7, 8])),
        g1_reset_sampler=MagicMock(),
    )


def test_pinned_reset_uses_the_selection_instead_of_the_sampler():
    table = torch.tensor([3, 1, 2])
    env = _pinned_env(lambda env_ids, count: table[env_ids])

    ranks, frames, bins = G1RLEnv._g1_episode_starts(env, torch.tensor([2, 0]))

    assert ranks.tolist() == [2, 3]
    assert frames.tolist() == [0, 0]
    assert bins is None
    env.g1_reset_sampler.sample.assert_not_called()


def test_pinned_reset_refuses_to_fall_back_to_the_sampler():
    env = _pinned_env(None)

    with pytest.raises(ValueError, match="trajectory-ranks"):
        G1RLEnv._g1_episode_starts(env, torch.tensor([0]))


def test_a_readout_fails_on_the_training_termination_only():
    """SuperMimic's rule: the task's own `tracking_failed` is the one failure term."""
    cfg = _readout_cfg()
    cfg.resolve()
    terms = {
        name: term
        for name, term in vars(cfg.terminations).items()
        if term is not None and not name.startswith("_")
    }
    failures = [name for name, term in terms.items() if not term.time_out]
    assert failures == ["tracking_failed"]
    assert terms["tracking_failed"].func is g1.tracking_failed
    assert terms["reference_finished"].time_out is True


def test_the_action_term_pins_the_canonical_joint_order():
    """Newton and PhysX enumerate joints differently; the action term pins the order."""
    action = ImitationG1EnvCfg().actions.joint_pos
    assert list(action.joint_names) == list(G1_ISAACLAB_JOINT_NAMES)
    assert action.preserve_order
