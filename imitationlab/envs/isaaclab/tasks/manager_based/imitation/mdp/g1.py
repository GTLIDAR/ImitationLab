"""Manager adapters of the ``imitation-g1-v1`` MDP (the SuperMimic replica).

Design reference: zita-ch/supermimic_vibed, commit 0525a23 (MIT).
The environment evaluates the shared scalar heads on live link poses/twists;
the world-model task evaluates them on differentiable forward kinematics.
"""

from __future__ import annotations

import torch


def _zeros(env, width: int):
    # ObservationManager discovers shapes before the data plane is finalized.
    return torch.zeros(env.num_envs, width, device=env.device)


def root_history(env):
    return (
        env._g1_states[..., :13].flatten(1)
        if hasattr(env, "_g1_states")
        else _zeros(env, 130)
    )


def joint_history(env):
    return (
        env._g1_states[..., 13:].flatten(1)
        if hasattr(env, "_g1_states")
        else _zeros(env, 580)
    )


def action_history(env):
    return (
        env._g1_actions.flatten(1) if hasattr(env, "_g1_actions") else _zeros(env, 290)
    )


def proprio_history(env):
    return (
        env._g1_proprio.flatten(1) if hasattr(env, "_g1_proprio") else _zeros(env, 990)
    )


def reference_cursor(env):
    return torch.stack(
        (
            env.trajectory_manager.env_traj_rank.to(env.device),
            env._current_reference_local_step,
            env.episode_length_buf,
        ),
        -1,
    ).float()


def sim_context(env):
    return env._g1_context if hasattr(env, "_g1_context") else _zeros(env, 33)


def push_flag(env):
    return _zeros(env, 1)


def policy_features(env):
    if not hasattr(env, "g1_task"):
        return _zeros(env, 1720)
    rank, frame = env.g1_cursor()
    return torch.cat(
        (env._g1_proprio.flatten(1), env.g1_library.goal(rank, frame, 10)), -1
    )


def critic_features(env):
    if not hasattr(env, "g1_task"):
        return _zeros(env, 914)
    rank, frame = env.g1_cursor()
    return env.g1_task.critic_obs(env.g1_state(), env._g1_context, rank, frame)


def replica_reward(env):
    rank, frame = env.g1_cursor()
    reward, parts = env.g1_mdp.reward(
        env.g1_state(),
        env.action_manager.action,
        env.action_manager.prev_action,
        rank,
        frame,
        body_state=env.g1_body_state(),
    )
    env._g1_reward_parts = parts
    # RewardManager multiplies each term by the 0.02 s control timestep.
    return reward / env.step_dt


def tracking_failed(env):
    rank, frame = env.g1_cursor()
    return env.g1_mdp.termination(env.g1_state(), rank, frame, hard=True)


def reference_finished(env):
    rank, frame = env.g1_cursor()
    return frame >= env.g1_library.length[rank] - 1


def latent_policy_features(env):
    if not hasattr(env, "g1_task"):
        return _zeros(env, 990 + env.g1_skill.command_dim)
    rank, frame = env.g1_cursor()
    return env.g1_task.policy_obs_from_parts(
        env._g1_proprio, env.g1_state(), rank, frame
    )


def latent_critic_features(env):
    if not hasattr(env, "g1_task"):
        return _zeros(env, 184 + env.g1_skill.command_dim)
    return critic_features(env)
