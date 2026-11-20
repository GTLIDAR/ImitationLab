#!/usr/bin/env python3
"""Check world-model reward, termination and replay history against a live env.

Uses the selected task's registered ``wm.task_model``. Rewards include terminal
transitions whenever the environment publishes post-step ``final_obs``. Empty, nonfinite or insufficiently
correlated comparisons fail rather than silently qualifying a task.

    pixi run -e isaaclab python scripts/tracking/validate_world_model_task.py \
        --task imitation-g1-v1 --assert-kitless --num_envs 256 --steps 200 \
        physics=newton_mjwarp <env.data overrides>
"""

from __future__ import annotations

import argparse
import json
import sys

from imitationlab.cli_args import add_kebab_aliases
from imitationlab.evaluation.world_model.validation import (
    comparison_stats,
    qualification_failures,
)


def build_parser():
    from imitationlab.runners.launch_compat import add_launcher_args

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", default="imitation-g1-v1")
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--action_std", type=float, default=0.3)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--min_correlation", type=float, default=0.9996)
    parser.add_argument("--min_termination_agreement", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--skill_encoder", default=None)
    parser.add_argument("--assert-kitless", action="store_true")
    parser.add_argument(
        "--check_noise_inference",
        action="store_true",
        help="Also qualify frozen collecting-policy replay with a disposable WMBPTT agent.",
    )
    parser.add_argument(
        "--check_environment_contract",
        action="store_true",
        help="Also verify live reset, randomization, action and timeout contracts.",
    )
    parser.add_argument(
        "--replay_checkpoint",
        help="Restore this WMBPTT checkpoint before the frozen replay check.",
    )
    add_launcher_args(parser)
    return add_kebab_aliases(parser, hide_snake_case=False)


def main() -> int:
    command = list(sys.argv)
    if "--assert-kitless" in sys.argv:
        from imitationlab.runners.runtime_bootstrap import install_kit_import_guard

        install_kit_import_guard()
    from isaaclab_tasks.utils import setup_preset_cli

    args, hydra_args = setup_preset_cli(build_parser(), sys.argv[1:])
    if args.replay_checkpoint and not args.check_noise_inference:
        raise ValueError("--replay_checkpoint requires --check_noise_inference.")
    args.headless = True
    sys.argv = [sys.argv[0], *hydra_args]
    import gymnasium as gym
    import imitationlab.envs.isaaclab.tasks  # noqa: F401
    import isaaclab_tasks  # noqa: F401
    import torch
    from isaaclab.app import launch_simulation
    from isaaclab_tasks.utils import resolve_task_config
    from imitationlab.runners.launch_compat import needs_kit

    env_cfg, agent_cfg = resolve_task_config(args.task, "rlopt_wmbptt_cfg_entry_point")
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    kit = needs_kit(env_cfg, args)  # resolves the physics preset before launching
    if args.assert_kitless and kit:
        raise RuntimeError("Selected config requires Kit; pass physics=newton_mjwarp.")
    with launch_simulation(env_cfg, args):
        from imitationlab.algorithms.common.world_model.replay import SequenceReplay
        from imitationlab.algorithms.common.world_model.task import load_task

        gym_env = gym.make(args.task, cfg=env_cfg)
        env = gym_env.unwrapped
        torch.manual_seed(args.seed)
        kwargs = dict(agent_cfg.wm.task_kwargs)
        if args.skill_encoder:
            kwargs.update(goal_mode="z", skill_encoder=args.skill_encoder)
        task = load_task(
            str(agent_cfg.wm.task_model), env, torch.device(env.device), **kwargs
        )
        post_step_terminal = getattr(env, "terminal_observation_is_post_step", False)
        rm = env.reward_manager
        term_index = {n: i for i, n in enumerate(rm._term_names)}
        replay = SequenceReplay(
            env.num_envs,
            args.steps + 2,
            task.state_dim,
            task.action_dim,
            task.context_dim,
            env.device,
        )
        obs, _ = env.reset()
        sums, skill_err = {}, {}
        hard_agree = hard_total = hist_err = hist_count = terminal_count = 0
        for _ in range(args.steps):
            rows = task.read_obs(obs[task.obs_group])
            action = args.action_std * torch.randn(
                env.num_envs, task.action_dim, device=env.device
            )
            replay.add(
                rows["state"],
                action,
                rows["hist_actions"][:, -1],
                rows["context"],
                rows["rank"],
                rows["frame"],
                rows["episode_step"],
                rows["push"],
            )
            obs, reward, terminated, truncated, extras = env.step(action)
            nxt = task.read_obs(obs[task.obs_group])
            same = (nxt["episode_step"] == rows["episode_step"] + 1) & ~nxt["push"]
            next_state = nxt["state"].clone()
            if post_step_terminal and "final_obs" in extras:
                final = extras["final_obs"]
                ids = final["_env_ids"].long()
                terminal_rows = task.read_obs(final["obs"][task.obs_group])
                next_state[ids] = terminal_rows["state"]
                same[ids] = ~terminal_rows["push"]
            terminal_count += int((same & terminated).sum())
            analytic_reward, parts = task.reward(
                next_state,
                action,
                rows["hist_actions"][:, -1],
                rows["context"],
                rows["rank"],
                rows["frame"],
            )
            if post_step_terminal:
                env_parts = env._g1_reward_parts
                for name, value in parts.items():
                    if name in env_parts:
                        sums.setdefault(name, []).append(
                            torch.stack((value[same], env_parts[name][same]), -1)
                        )
                sums.setdefault("total", []).append(
                    torch.stack((analytic_reward[same], reward[same]), -1)
                )
            else:
                for name, value in parts.items():
                    if name in term_index:
                        env_value = rm._step_reward[:, term_index[name]]
                        sums.setdefault(name, []).append(
                            torch.stack((value[same] / task.dt, env_value[same]), -1)
                        )
            hard = task.termination(
                next_state, rows["context"], rows["rank"], rows["frame"], hard=True
            )
            hard_agree += int(((hard > 0.5) == terminated)[same].sum())
            hard_total += int(same.sum())
            if replay.steps >= task.history:
                tau = torch.full((env.num_envs,), replay.steps - 1, device=env.device)
                ids = torch.arange(env.num_envs, device=env.device)
                rebuilt = replay.history(tau, ids, task.history)
                hist_err = max(
                    hist_err,
                    float((rebuilt["hist_states"] - rows["hist_states"]).abs().max()),
                    float((rebuilt["hist_actions"] - rows["hist_actions"]).abs().max()),
                )
                hist_count += env.num_envs
            if post_step_terminal and getattr(task, "skill", None) is not None:
                expected = task.goal(nxt["state"], nxt["rank"], nxt["frame"])
                for group in ("policy", "critic"):
                    actual = obs[group]["features"][:, -task.skill.command_dim :]
                    delta = (actual - expected).abs()
                    error = (
                        float(delta.max())
                        if bool(torch.isfinite(delta).all())
                        else float("inf")
                    )
                    key = f"{group}_command_max_abs"
                    skill_err[key] = max(skill_err.get(key, 0.0), error)
            elif getattr(task, "skill", None) is not None:
                hl = env.current_expert_macro_transition_batch(
                    horizon_steps=task.skill.frames
                )["hl"]
                env_frames = torch.cat(
                    (hl["state"].unsqueeze(1), hl["future_window"]), 1
                )[:, : task.skill.frames]
                mine = task.skill_frames(nxt["state"], nxt["rank"], nxt["frame"])
                diff = (mine - env_frames).abs()
                z_env, z_mine = task.skill(env_frames)[:, :-2], task.skill(mine)[:, :-2]
                for key, value in (
                    ("joint_pos_max", diff[..., :29].max()),
                    ("anchor_pos_max", diff[..., 29:32].max()),
                    ("anchor_rot6d_max", diff[..., 32:38].max()),
                    ("z_max_abs", (z_mine - z_env).abs().max()),
                    (
                        "z_rel_l2",
                        (
                            (z_mine - z_env).norm(dim=-1)
                            / z_env.norm(dim=-1).clamp_min(1e-9)
                        ).max(),
                    ),
                ):
                    skill_err[key] = max(skill_err.get(key, 0.0), float(value))
        report = {
            "protocol": {
                "command": command,
                "task": args.task,
                "num_envs": env.num_envs,
                "steps": args.steps,
                "seed": args.seed,
                "action_std": args.action_std,
                "step_dt": env.step_dt,
                "resolved_data_cfg": env.cfg.data.to_dict(),
                "reference_prefetch_mode": env.expert_data_plane.reference_prefetch_mode,
                "task_model": str(agent_cfg.wm.task_model),
            },
            "terms": {
                name: comparison_stats(torch.cat(chunks))
                for name, chunks in sums.items()
            },
            "termination_agreement": hard_agree / max(1, hard_total),
            "termination_samples": hard_total,
            "termination_scope": "all transitions"
            if post_step_terminal
            else "survivors (legacy final_obs)",
            "terminal_samples": terminal_count,
            "require_terminal_samples": post_step_terminal,
            "history_max_abs_err": hist_err,
            "history_samples": hist_count,
            "skill_goal_max_err": skill_err,
            "describe": task.describe(),
            "missing_reward_terms": sorted(set(env._g1_reward_parts) - set(sums))
            if post_step_terminal
            else [],
        }
        report["failures"] = qualification_failures(
            report,
            tolerance=args.tolerance,
            min_correlation=args.min_correlation,
            min_termination_agreement=args.min_termination_agreement,
        )
        if args.check_environment_contract:
            from imitationlab.evaluation.world_model.g1_env_validation import (
                qualify_g1_environment,
            )

            report["environment_contract"] = qualify_g1_environment(env)
        if args.check_noise_inference:
            from pathlib import Path
            from imitationlab.envs.isaaclab.envs.torchrl_adapter import (
                IsaacLabWrapper,
                attach_terminal_obs_reader,
            )
            from imitationlab.envs.isaaclab.tasks.manager_based.imitation.command_interface import (
                bind_command_interface,
            )
            from torchrl.envs import Compose, RewardSum, StepCounter, TransformedEnv
            from imitationlab.algorithms import WMBPTT
            from imitationlab.evaluation.world_model.validation import (
                frozen_policy_replay_check,
            )

            bind_command_interface(agent_cfg, env_cfg)
            agent_cfg.env.num_envs = env.num_envs
            agent_cfg.env.env_name = args.task
            agent_cfg.device = str(env.device)
            agent_cfg.logger.backend = ""
            agent_cfg.logger.log_dir = str(
                Path(args.output_json or "outputs/g1_validation/live.json")
                .resolve()
                .parent
                / "replay_probe"
            )
            agent_cfg.wm.compile = False
            agent_cfg.wm.bf16 = False
            wrapped = TransformedEnv(
                attach_terminal_obs_reader(IsaacLabWrapper(gym_env)),
                Compose(RewardSum(), StepCounter(500)),
            )
            agent = WMBPTT(env=wrapped, config=agent_cfg)
            checkpoint = None
            if args.replay_checkpoint:
                import hashlib

                checkpoint = Path(args.replay_checkpoint).resolve()
                agent.load_model(str(checkpoint))
                with checkpoint.open("rb") as handle:
                    checkpoint = {
                        "path": str(checkpoint),
                        "sha256": hashlib.file_digest(handle, "sha256").hexdigest(),
                    }
            report["noise_inference"] = frozen_policy_replay_check(agent)
            report["noise_inference"]["checkpoint"] = checkpoint
            if not report["noise_inference"]["passed"]:
                report["failures"].append("frozen collecting-policy replay failed")
        print(json.dumps(report, indent=2, default=str))
        if args.output_json:
            from pathlib import Path

            output = Path(args.output_json)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2, default=str) + "\n")
        env.close()
    return bool(report["failures"])


if __name__ == "__main__":
    raise SystemExit(main())
