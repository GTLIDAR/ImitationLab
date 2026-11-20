#!/usr/bin/env python3
"""Held-out k-step prediction error of a WMBPTT checkpoint's world model.

Rolls the checkpoint's deterministic policy (plus the training action noise,
which is how the replay was collected) in a fresh env with its own seed,
records every transition, then rolls the saved dynamics model open loop on the
logged actions from every valid start. Reports, for k in ``--ks``:

* pelvis position error (mm), pelvis orientation error (deg), joint position
  error (deg, mean |.|), the 14 tracked bodies' position error through forward
  kinematics (mm, mean over bodies, root-relative and world);
* the normalized error the training loss uses (squared, per increment unit);
* the same numbers for the constant-velocity baseline (the untrained model's
  prior: s_{t+1} = s_t + mean increment) for scale.

The task model is the checkpoint's ``wm.task_model`` (``imitation-g1-v1``), and its sizes come from that task. The
policy acts as deployed (``WMPolicy``, with the reference setpoint of a
``wm.actor_residual`` checkpoint). A ``wm.wm_structured`` checkpoint's model
outputs are integrated semi-implicitly, its normalized error divides by the
plain increments' scale (``inc_norm``), and its baseline integrates the mean
output, which is the semi-implicit drift plus the mean correction.

    pixi run -e isaaclab python scripts/tracking/eval_world_model_kstep.py \
        --checkpoint <model_step_N.pt> --num_envs 1024 --steps 400 \
        [--goal_encoder <local encoder .pt>] <env overrides>

Launched directly (not through a pixi task) it needs ``TERM=xterm
OMNI_KIT_ACCEPT_EULA=YES`` in the environment.
"""

from __future__ import annotations

import argparse
import json
import math
import sys

from imitationlab.cli_args import add_kebab_aliases
from imitationlab.runners.launch_compat import add_launcher_args

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--task", default="imitation-g1-v1")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--warmup_steps", type=int, default=100)
parser.add_argument("--action_noise_std", type=float, default=None)
parser.add_argument("--ks", default="1,8,32")
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--output_json", default=None)
parser.add_argument(
    "--goal_encoder",
    default=None,
    help="override the checkpoint's wm.goal_encoder path (a z-goal checkpoint trained on a cluster "
    "records the cluster path)",
)
add_launcher_args(parser)
add_kebab_aliases(parser, hide_snake_case=False)


def main() -> int:
    from isaaclab_tasks.utils import setup_preset_cli

    args, hydra_args = setup_preset_cli(parser, sys.argv[1:])
    args.headless = True
    sys.argv = [sys.argv[0], *hydra_args]
    import gymnasium as gym
    import imitationlab.envs.isaaclab.tasks  # noqa: F401
    import isaaclab_tasks  # noqa: F401
    import torch
    from isaaclab.app import launch_simulation
    from isaaclab_tasks.utils import resolve_task_config

    env_cfg, _ = resolve_task_config(args.task, "rlopt_wmbptt_cfg_entry_point")
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    with launch_simulation(env_cfg, args):
        from imitationlab.envs.isaaclab.task_model.g1_kinematics import (
            G1_TRACKED_BODY_NAMES,
            quat_wxyz_to_matrix,
        )
        from imitationlab.envs.isaaclab.task_model.g1_task import unpack
        from imitationlab.algorithms.common.world_model.learner import WMPolicy
        from imitationlab.algorithms.common.world_model.config import WMBPTTConfig
        from imitationlab.algorithms.common.world_model.networks import (
            DynamicsModel,
            build_actor,
        )
        from imitationlab.algorithms.common.world_model.replay import SequenceReplay
        from imitationlab.algorithms.common.world_model.task import load_task

        env = gym.make(args.task, cfg=env_cfg).unwrapped
        dev = env.device
        torch.manual_seed(args.seed)
        ck = torch.load(args.checkpoint, map_location=dev, weights_only=False)
        cfg = ck["wm_config"]
        # Same task construction as WMBPTT.__init__: the goal mode and skill
        # encoder set the actor's observation width, so a z-goal checkpoint
        # needs its encoder (the recorded path, or --goal_encoder).
        task_kwargs = dict(cfg.get("task_kwargs") or {})
        goal_encoder = args.goal_encoder or cfg.get("goal_encoder") or None
        if cfg.get("goal", "explicit") != "explicit" or goal_encoder:
            task_kwargs.update(
                goal_mode=str(cfg.get("goal", "explicit")), skill_encoder=goal_encoder
            )
        task_model = cfg.get("task_model", WMBPTTConfig.task_model)
        task = load_task(
            task_model,
            env,
            dev,
            sensor_noise=bool(cfg.get("sensor_noise", True)),
            **task_kwargs,
        )
        structured = bool(cfg.get("wm_structured", False))
        dyn = DynamicsModel(
            task.wm_input_dim,
            task.increment_dim,
            cfg["wm_hidden"],
            structured=structured,
        ).to(dev)
        dyn.load_state_dict(ck["dynamics_state_dict"])
        if structured:
            task.bind_structured()
            integrate, error_norm = task.structured_integrate, dyn.inc_norm
        else:
            integrate, error_norm = task.integrate, dyn.out_norm
        meta = ck.get("policy_meta") or {}
        actor = build_actor(
            str(meta.get("actor_arch", "flat")),
            task.policy_obs_dim,
            task.action_dim,
            cfg["actor_hidden"],
            1.0,
            cfg.get("actor_activation", "elu"),
            goal_dim=meta.get("actor_goal_dim"),
            latent_dim=int(meta.get("actor_latent_dim", 32)),
        ).to(dev)
        prefix = "module.actor."
        actor.load_state_dict(
            {
                k[len(prefix) :]: v
                for k, v in ck["policy_state_dict"].items()
                if k.startswith(prefix)
            }
        )
        residual = bool(meta.get("actor_residual", False))
        if residual:
            task.bind_reference_setpoint()
        policy = WMPolicy(task, actor, bool(cfg.get("bf16", True)), residual)
        noise = (
            cfg["action_noise_std"]
            if args.action_noise_std is None
            else args.action_noise_std
        )
        ks = [int(k) for k in args.ks.split(",")]
        K = max(ks)
        rp = SequenceReplay(
            env.num_envs,
            args.steps + 1,
            task.state_dim,
            task.action_dim,
            task.context_dim,
            dev,
        )
        obs, _ = env.reset()
        with torch.no_grad():
            for step in range(args.warmup_steps + args.steps + 1):
                rows = task.read_obs(obs["wm"])
                a = policy(*(obs[group][key] for group, key in task.policy_in_keys))
                a = (a + noise * torch.randn_like(a)).clamp(
                    -cfg["action_clip"], cfg["action_clip"]
                )
                if step >= args.warmup_steps:
                    rp.add(
                        rows["state"],
                        a,
                        rows["hist_actions"][:, -1],
                        rows["context"],
                        rows["rank"],
                        rows["frame"],
                        rows["episode_step"],
                        rows["push"],
                    )
                obs, *_ = env.step(a)
            starts = torch.arange(0, args.steps - K, 4, device=dev)
            tau = starts.repeat_interleave(env.num_envs)
            envs = torch.arange(env.num_envs, device=dev).repeat(len(starts))
            out = {
                "checkpoint": args.checkpoint,
                "frames": ck.get("cumulative_env_frames"),
                "hours": float(ck.get("cumulative_train_seconds", 0.0)) / 3600,
                "num_envs": args.num_envs,
                "steps": args.steps,
                "action_noise_std": noise,
                "task_model": task_model,
                "wm_structured": structured,
                "actor_residual": residual,
                "model": {},
                "mean_increment": {},
            }
            scale = torch.maximum(error_norm.std(), task.increment_floor)
            acc = {name: {k: [] for k in ks} for name in ("model", "mean_increment")}
            for chunk in torch.split(torch.arange(len(tau), device=dev), 8192):
                win = rp.window(tau[chunk], envs[chunk], K + 1)
                mask = torch.cumprod(win["valid"].float(), 1)
                ctx = win["context"][:, 0]
                for name in ("model", "mean_increment"):
                    s = win["state"][:, 0]
                    for j in range(K):
                        if name == "model":
                            inc = dyn(
                                task.wm_input(s, win["action"][:, j], ctx),
                                bf16=bool(cfg.get("bf16", True)),
                            )
                        else:
                            inc = dyn.out_norm.mean.expand(s.shape[0], -1)
                        s = integrate(s, inc)
                        k = j + 1
                        if k not in ks:
                            continue
                        m = mask[:, j] > 0
                        if not m.any():
                            continue
                        true = win["state"][m, k]
                        pred = s[m]
                        e = task.state_error(pred, true)
                        tp = unpack(true)
                        pp = unpack(pred)
                        bt, _ = task.kin(
                            tp[0],
                            quat_wxyz_to_matrix(tp[1]),
                            tp[4],
                            G1_TRACKED_BODY_NAMES,
                        )
                        bp, _ = task.kin(
                            pp[0],
                            quat_wxyz_to_matrix(pp[1]),
                            pp[4],
                            G1_TRACKED_BODY_NAMES,
                        )
                        acc[name][k].append(
                            torch.stack(
                                (
                                    e[:, 0:3].norm(dim=-1) * 1000,
                                    e[:, 3:6].norm(dim=-1) * 180 / math.pi,
                                    e[:, 12:41].abs().mean(-1) * 180 / math.pi,
                                    (bp - bt).norm(dim=-1).mean(-1) * 1000,
                                    (
                                        (bp - pp[0].unsqueeze(1))
                                        - (bt - tp[0].unsqueeze(1))
                                    )
                                    .norm(dim=-1)
                                    .mean(-1)
                                    * 1000,
                                    (e / scale).square().mean(-1),
                                ),
                                -1,
                            )
                        )
            cols = (
                "root_pos_mm",
                "root_rot_deg",
                "joint_pos_deg",
                "body_pos_world_mm",
                "body_pos_local_mm",
                "normalized",
            )
            for name in acc:
                for k in ks:
                    if not acc[name][k]:
                        continue
                    vals = torch.cat(acc[name][k])
                    out[name][f"k{k}"] = {
                        c: float(vals[:, i].mean()) for i, c in enumerate(cols)
                    }
                    out[name][f"k{k}"].update(
                        {
                            f"{c}_p90": float(vals[:, i].quantile(0.9))
                            for i, c in enumerate(cols[:5])
                        }
                    )
                    out[name][f"k{k}"]["windows"] = int(vals.shape[0])
        print(json.dumps(out, indent=1))
        if args.output_json:
            with open(args.output_json, "w") as fh:
                json.dump(out, fh, indent=1)
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
