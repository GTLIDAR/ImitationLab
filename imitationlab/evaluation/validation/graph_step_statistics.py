"""G1 env control steps driven directly: statistics, a sync check and timing.

WMBPTT steps the env itself, so these modes build the env as the training
entrypoint does and drive it with a fixed state-feedback policy, with no agent:
each step targets the reference joint positions of the next frame, plus
Gaussian noise of ``--noise`` in action units from its own CUDA generator, so
the env's random streams are the env's alone.

- ``run`` records per-step device statistics (no host synchronization inside
  the loop): mean reward and reward parts, reset and termination fractions,
  episode length at reset, the live MPJPE metrics, the reset sampler's mean bin
  return, and histograms of the new start position in its clip, of the new
  clip (one bin per clip) and of the new clip's length. Every LAFAN1 clip is
  longer than the last length edge (3,200 frames), so there the length
  histogram has one occupied bin and compares nothing; ``compare-groups``
  reports each histogram's ``bins_with_mass``. ``compare`` is
  ``step_statistics.compare_statistics``: block
  means with z-scores within one pair of runs. The adaptive reset sampler makes
  each run one realization of a slow transient (its probability mass reaches
  the hard bins at a run-specific step), so within-run block errors understate
  the spread between runs: two eager seeds already fail it.
  ``compare-groups`` is the test to use: several runs per arm, each run's mean
  after ``--skip`` steps as one sample, Welch's t and an exact permutation p
  between arms; for each histogram, the total variation between the arms' mean
  distributions with its permutation p, beside the pairwise spread within arms.
- ``sync`` steps until the graph step has captured (one step for the masked
  step), then runs ``--steps`` steps
  under ``torch.cuda.set_sync_debug_mode("error")``, reading what WMBPTT reads
  (the ``wm`` group, reward, dones, the log and the masked terminal rows), and
  finally checks that reading the indexed ``final_obs`` does raise there.
- ``terminal`` checks ``final_obs`` each step: its ids are the ended envs, its
  episode step is the pre-step length plus one, and (masked step) the rows of
  ``final_obs_masked`` that did not reset equal the returned observation.
- ``writes`` checks, with live physics, that each step's reset reached the
  simulator (Newton/MuJoCo-Warp only). On the rows the step reset: the robot's
  root pose and joint positions and velocities equal the written reset state
  (``_g1_states[:, -1]``); the raw ``state_0.body_q`` root equals the
  ``joint_q`` root, so forward kinematics ran; MuJoCo-Warp's
  ``qacc_warmstart`` is zero, so the solver state was reset. After every step
  Newton's world reset mask is clear. The bitwise tests replay recorded
  physics, which overwrites these arrays after each ``sim.step``, so they
  cannot see this. The record also gives the liveness evidence: how far the
  reset moved the root, and the share of kept rows with a nonzero warm start.
- ``time`` reports the mean wall time of ``env.step`` over ``--steps`` steps
  after ``--warmup`` steps, with the GPU synchronized at both ends, the policy
  inside that loop, and then, alone, each captured graph's replay and the
  physics step (``sim.step``).

Usage (repository root, isaaclab env)::

    python -m imitationlab.evaluation.validation.graph_step_statistics run --out eager.json \\
        --task imitation-g1-v1 --num_envs 4096 --steps 600 -- <overrides>
    python -m imitationlab.evaluation.validation.graph_step_statistics compare eager.json masked.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from imitationlab.cli_args import add_kebab_aliases
from imitationlab.evaluation.validation.step_statistics import (
    LENGTH_BIN_EDGES,
    START_BINS,
    compare_statistics,
)
from imitationlab.paths import REPO_ROOT

__all__ = ["compare_groups", "main", "reference_policy"]

_POLICY_SEED = 9001
SEGMENT_STEPS = 100


def reference_policy(env: Any, noise: float, generator: Any) -> Any:
    """Raw actions whose joint targets are the next reference frame's, plus noise."""
    import torch

    term = env.action_manager.get_term("joint_pos")
    rank, frame = env.g1_cursor()
    target = env.g1_library.ref_state(rank, frame + 1)[:, 13:42]
    action = (target - term._offset) / term._scale
    if noise > 0.0:
        action = action + noise * torch.randn(
            action.shape, device=action.device, generator=generator
        )
    return action


class _Recorder:
    """Per-step statistics accumulated on the device."""

    def __init__(self, env: Any, steps: int) -> None:
        import torch

        self.env = env
        device = env.device
        self.parts: list[str] | None = None
        self.terms = list(env.termination_manager._term_names)
        self.rows: list[torch.Tensor] = []
        segments = -(-int(steps) // SEGMENT_STEPS)
        self.starts = torch.zeros(
            segments, START_BINS, device=device, dtype=torch.float64
        )
        self.lengths = torch.zeros(
            segments, len(LENGTH_BIN_EDGES) - 1, device=device, dtype=torch.float64
        )
        self.edges = torch.tensor(LENGTH_BIN_EDGES[1:-1], device=device)
        # One bin per clip: every LAFAN1 clip is longer than the last length
        # edge, so the length histogram alone cannot tell clips apart.
        self.clips = torch.zeros(
            segments,
            int(env.trajectory_manager.num_trajectories),
            device=device,
            dtype=torch.float64,
        )

    def names(self) -> list[str]:
        return (
            ["reward"]
            + [f"part/{name}" for name in self.parts or []]
            + ["reset_fraction"]
            + [f"done/{name}" for name in self.terms]
            + ["_len_sum", "_len_count", "mpjpe_l_mm", "mpjpe_g_mm"]
            + ["sampler/bin_G_mean"]
        )

    def record(self, pre_length: Any) -> None:
        import torch

        env = self.env
        if self.parts is None:
            self.parts = sorted(env._g1_reward_parts)
        mask = env.reset_buf.to(torch.bool)
        weight = mask.to(torch.float32)
        tm = env.trajectory_manager
        clip_len = tm.length.index_select(0, tm.env_traj_rank)
        position = tm.env_step.to(torch.float32) / (clip_len - 1).clamp(min=1).to(
            torch.float32
        )
        start_bin = (position * START_BINS).long().clamp(0, START_BINS - 1)
        segment = len(self.rows) // SEGMENT_STEPS
        self.starts[segment].index_add_(0, start_bin, weight.to(torch.float64))
        self.lengths[segment].index_add_(
            0, torch.bucketize(clip_len, self.edges), weight.to(torch.float64)
        )
        self.clips[segment].index_add_(
            0, tm.env_traj_rank.to(device=self.clips.device), weight.to(torch.float64)
        )
        metrics = env._reference_term.metrics
        parts = env._g1_reward_parts
        self.rows.append(
            torch.cat(
                (
                    env.reward_buf.float().mean().reshape(1),
                    torch.stack([parts[n].float().mean() for n in self.parts]),
                    weight.mean().reshape(1),
                    env.termination_manager._term_dones.float().mean(dim=0),
                    ((pre_length + 1).float() * weight).sum().reshape(1),
                    weight.sum().reshape(1),
                    metrics["mpjpe_l_mm"].float().mean().reshape(1),
                    metrics["mpjpe_g_mm"].float().mean().reshape(1),
                    env.g1_reset_sampler.bin_G.float().mean().reshape(1),
                )
            )
        )

    def finish(self) -> dict[str, Any]:
        import torch

        stacked = torch.stack(self.rows).cpu()
        return {
            "series": {n: stacked[:, i].tolist() for i, n in enumerate(self.names())},
            # Totals, for `compare`; per 100-step segment, for `compare-groups`.
            "hist": {
                "reset_start_position": self.starts.sum(0).cpu().tolist(),
                "reset_clip_length": self.lengths.sum(0).cpu().tolist(),
                "reset_clip": self.clips.sum(0).cpu().tolist(),
            },
            "segment_hist": {
                "reset_start_position": self.starts.cpu().tolist(),
                "reset_clip_length": self.lengths.cpu().tolist(),
                "reset_clip": self.clips.cpu().tolist(),
            },
            "segment_steps": SEGMENT_STEPS,
        }


@contextlib.contextmanager
def _env(args: argparse.Namespace, overrides: list[str]):
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "tracking"))
    from imitationlab.runners.runtime_bootstrap import install_kit_import_guard

    install_kit_import_guard()
    from imitationlab.runners.launch_compat import add_launcher_args, needs_kit
    from isaaclab_tasks.utils import setup_preset_cli

    parser = argparse.ArgumentParser()
    add_launcher_args(parser)
    launcher_args, hydra_args = setup_preset_cli(parser, ["--headless", *overrides])
    sys.argv = [sys.argv[0], *hydra_args]

    import gymnasium as gym
    import imitationlab.envs.isaaclab.tasks  # noqa: F401
    import isaaclab_tasks  # noqa: F401
    import torch
    from isaaclab.app import launch_simulation
    from isaaclab_tasks.utils import resolve_task_config

    env_cfg, _ = resolve_task_config(args.task, "wmbptt_cfg_entry_point")
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    if needs_kit(env_cfg, launcher_args):
        raise RuntimeError("This driver needs the kitless Newton/MJWarp runtime.")
    with launch_simulation(env_cfg, launcher_args):
        torch.manual_seed(args.seed)
        env = gym.make(args.task, cfg=env_cfg).unwrapped
        try:
            yield env
        finally:
            env.close()


def _describe(env: Any, args: argparse.Namespace) -> dict[str, Any]:
    from imitationlab.envs.isaaclab.envs.collect_fastpath import describe

    return {
        "task": args.task,
        "num_envs": int(env.num_envs),
        "seed": int(args.seed),
        "noise": float(args.noise),
        "step_mode": str(getattr(env, "_step_mode", "eager")),
        "wm_only_observations": bool(getattr(env.cfg, "wm_only_observations", False)),
        "collect_fastpath": describe(),
    }


def _run(args: argparse.Namespace, overrides: list[str]) -> None:
    import torch

    with _env(args, overrides) as env:
        generator = torch.Generator(device=env.device).manual_seed(_POLICY_SEED)
        recorder = _Recorder(env, args.steps)
        env.reset(seed=args.seed)
        resets = torch.zeros((), device=env.device)
        with torch.no_grad():
            for _ in range(args.steps):
                pre_length = env.episode_length_buf.clone()
                env.step(reference_policy(env, args.noise, generator))
                recorder.record(pre_length)
                resets += env.reset_buf.sum()
        record = _describe(env, args)
        record.update(recorder.finish())
        record["resets"] = float(resets)
        record["steps"] = int(args.steps)
    Path(args.out).write_text(json.dumps(record))


def _sync(args: argparse.Namespace, overrides: list[str]) -> None:
    import torch

    with _env(args, overrides) as env:
        generator = torch.Generator(device=env.device).manual_seed(_POLICY_SEED)
        env.reset(seed=args.seed)
        stepper = None
        graph = getattr(env, "_step_mode", "eager") == "graph"
        with torch.no_grad():
            for _ in range(20):
                env.step(reference_policy(env, args.noise, generator))
                stepper = getattr(env, "_masked_stepper", None)
                if not graph or getattr(stepper, "graphs", None) is not None:
                    break
            if graph and getattr(stepper, "graphs", None) is None:
                raise RuntimeError("The graph step did not capture in 20 steps.")
            resets = torch.zeros((), device=env.device)
            seen = torch.zeros((), device=env.device)
            torch.cuda.synchronize()
            torch.cuda.set_sync_debug_mode("error")
            try:
                for _ in range(args.steps):
                    obs, reward, terminated, truncated, extras = env.step(
                        reference_policy(env, args.noise, generator)
                    )
                    for value in obs["wm"].values():
                        seen += value.float().mean()
                    masked = extras["final_obs_masked"]
                    resets += masked["_env_mask"].sum()
                    for value in masked["obs"]["wm"].values():
                        seen += value.float().mean()
                    seen += reward.mean() + (terminated | truncated).float().mean()
                    for value in extras["log"].values():
                        seen += value.float().mean()
                print(
                    f"[INFO] {args.steps} {env._step_mode} steps ran without a host "
                    "synchronization.",
                    flush=True,
                )
                refused = False
                try:
                    extras["final_obs"]["_env_ids"]
                except RuntimeError as exc:
                    refused = "synchroniz" in str(exc)
            finally:
                torch.cuda.set_sync_debug_mode("default")
        record = _describe(env, args)
        record.update(
            steps=int(args.steps),
            resets=float(resets),
            captured_at_step=getattr(stepper, "captured_at_step", None),
            indexed_final_obs_refused=refused,
            finite=bool(torch.isfinite(seen)),
        )
    Path(args.out).write_text(json.dumps(record))


def _terminal(args: argparse.Namespace, overrides: list[str]) -> None:
    """Check the published terminal rows against what the step must have reached."""
    import torch

    with _env(args, overrides) as env:
        generator = torch.Generator(device=env.device).manual_seed(_POLICY_SEED)
        env.reset(seed=args.seed)
        counts = {
            "rows": 0,
            "id_mismatch_steps": 0,
            "length_mismatches": 0,
            "kept_row_mismatches": 0,
            "masked_steps": 0,
        }
        with torch.no_grad():
            for _ in range(args.steps):
                pre_length = env.episode_length_buf.clone()
                obs, _, terminated, truncated, extras = env.step(
                    reference_policy(env, args.noise, generator)
                )
                done = (terminated | truncated).nonzero().squeeze(-1)
                final = extras.get("final_obs")
                ids = (
                    torch.empty(0, dtype=torch.long, device=env.device)
                    if final is None
                    else final["_env_ids"]
                )
                counts["rows"] += int(ids.numel())
                counts["id_mismatch_steps"] += int(not torch.equal(ids, done))
                if ids.numel():
                    cursor = final["obs"]["wm"]["reference_cursor"]
                    counts["length_mismatches"] += int(
                        (cursor[:, 2] != (pre_length[ids] + 1).float()).sum()
                    )
                masked = extras.get("final_obs_masked")
                if masked is not None:
                    counts["masked_steps"] += 1
                    keep = ~masked["_env_mask"]
                    for key, value in obs["wm"].items():
                        counts["kept_row_mismatches"] += int(
                            (masked["obs"]["wm"][key][keep] != value[keep])
                            .any(-1)
                            .sum()
                        )
        record = _describe(env, args)
        record.update(steps=int(args.steps), **counts)
    print(json.dumps(record), flush=True)
    Path(args.out).write_text(json.dumps(record))


def _tensor(value: Any) -> Any:
    return value.torch if hasattr(value, "torch") else value


def _writes(args: argparse.Namespace, overrides: list[str]) -> None:
    """Live physics: the simulator holds what each step's reset wrote."""
    import torch
    import warp as wp
    from isaaclab_newton.physics.newton_manager import NewtonManager

    with _env(args, overrides) as env:
        generator = torch.Generator(device=env.device).manual_seed(_POLICY_SEED)
        env.reset(seed=args.seed)
        n, device = int(env.num_envs), env.device
        robot = env.robot
        joint_ids = env._g1_joint_ids
        origins = env.scene.env_origins
        solver_data = getattr(NewtonManager._solver, "mjw_data", None)
        names = (
            "root_position",
            "root_orientation",
            "joint_position",
            "joint_velocity",
            "fk_root",
            "reset_warmstart",
        )
        worst = {name: torch.zeros((), device=device) for name in names}
        totals = {
            name: torch.zeros((), device=device, dtype=torch.float64)
            for name in (
                "reset_rows",
                "pending_masks",
                "reset_root_move",
                "kept_root_move",
                "kept_rows",
                "kept_warm_rows",
            )
        }

        def worst_on(name: str, error: Any, rows: Any) -> None:
            value = torch.where(rows, error.float(), 0.0).amax()
            torch.maximum(worst[name], value, out=worst[name])

        stepper = None
        with torch.no_grad():
            for _ in range(args.steps):
                before = _tensor(robot.data.root_link_pose_w)[:, :3].clone()
                env.step(reference_policy(env, args.noise, generator))
                stepper = getattr(env, "_masked_stepper", None)
                reset = stepper.reset_mask if stepper is not None else env.reset_buf
                reset = reset.to(torch.bool)
                kept = ~reset
                # The raw arrays first: a robot.data read may run forward().
                for mask in (
                    NewtonManager._world_reset_mask,
                    NewtonManager._fk_reset_mask,
                ):
                    if mask is not None:
                        totals["pending_masks"] += wp.to_torch(mask).sum()
                state = NewtonManager.backend.state_0
                joint_q = wp.to_torch(state.joint_q).reshape(n, -1)
                body_q = wp.to_torch(state.body_q).reshape(n, -1, 7)
                worst_on(
                    "fk_root", (body_q[:, 0, :3] - joint_q[:, :3]).abs().amax(-1), reset
                )
                if solver_data is not None:
                    warm = wp.to_torch(solver_data.qacc_warmstart).reshape(n, -1)
                    warm = warm.abs().amax(-1)
                    worst_on("reset_warmstart", warm, reset)
                    totals["kept_warm_rows"] += (kept & (warm > 0)).sum()
                written = env._g1_states[:, -1]
                quat = written[:, 3:7]  # wxyz; the sim pose is xyzw
                pose = torch.cat(
                    (written[:, :3] + origins, quat[:, 1:], quat[:, :1]), -1
                )
                root = _tensor(robot.data.root_link_pose_w)
                worst_on(
                    "root_position", (root[:, :3] - pose[:, :3]).abs().amax(-1), reset
                )
                worst_on(
                    "root_orientation",
                    (root[:, 3:7] - pose[:, 3:]).abs().amax(-1),
                    reset,
                )
                q = _tensor(robot.data.joint_pos).index_select(1, joint_ids)
                qd = _tensor(robot.data.joint_vel).index_select(1, joint_ids)
                worst_on(
                    "joint_position", (q - written[:, 13:42]).abs().amax(-1), reset
                )
                worst_on(
                    "joint_velocity", (qd - written[:, 42:71]).abs().amax(-1), reset
                )
                move = (root[:, :3] - before).norm(dim=-1).double()
                totals["reset_rows"] += reset.sum()
                totals["kept_rows"] += kept.sum()
                totals["reset_root_move"] += torch.where(reset, move, 0.0).sum()
                totals["kept_root_move"] += torch.where(kept, move, 0.0).sum()
        host = {name: float(value) for name, value in totals.items()}
        record = _describe(env, args)
        record.update(
            steps=int(args.steps),
            captured_at_step=getattr(stepper, "captured_at_step", None),
            max_error={name: float(value) for name, value in worst.items()},
            solver_warmstart_checked=solver_data is not None,
            reset_rows=host["reset_rows"],
            pending_masks=host["pending_masks"],
            reset_root_move_mean_m=host["reset_root_move"]
            / max(host["reset_rows"], 1.0),
            kept_root_move_mean_m=host["kept_root_move"] / max(host["kept_rows"], 1.0),
            kept_warmstart_nonzero_share=host["kept_warm_rows"]
            / max(host["kept_rows"], 1.0),
        )
    print(json.dumps(record), flush=True)
    Path(args.out).write_text(json.dumps(record))


def _time(args: argparse.Namespace, overrides: list[str]) -> None:
    import torch

    with _env(args, overrides) as env:
        generator = torch.Generator(device=env.device).manual_seed(_POLICY_SEED)
        env.reset(seed=args.seed)
        resets = torch.zeros((), device=env.device)
        with torch.no_grad():
            for _ in range(args.warmup):
                env.step(reference_policy(env, args.noise, generator))
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(args.steps):
                env.step(reference_policy(env, args.noise, generator))
                resets += env.reset_buf.sum()
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
        stepper = getattr(env, "_masked_stepper", None)
        record = _describe(env, args)
        record.update(
            steps=int(args.steps),
            warmup=int(args.warmup),
            ms_per_step=1e3 * seconds / args.steps,
            resets_per_step=float(resets) / args.steps,
            captured_at_step=getattr(stepper, "captured_at_step", None),
            max_memory_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
        )

        # Parts of the measured loop, each timed alone and last, because these
        # calls move the simulation without the rest of the step.
        def timed(fn) -> float:
            torch.cuda.synchronize()
            begin = time.perf_counter()
            with torch.no_grad():
                for _ in range(args.steps):
                    fn()
            torch.cuda.synchronize()
            return 1e3 * (time.perf_counter() - begin) / args.steps

        record["policy_ms_per_step"] = timed(
            lambda: reference_policy(env, args.noise, generator)
        )
        graphs = getattr(stepper, "graphs", None)
        if graphs is not None:
            for name, graph in graphs.items():
                record[f"graph_{name}_ms"] = timed(graph.replay)
        record["physics_ms_per_step"] = timed(lambda: env.sim.step(render=False))
    print(json.dumps(record), flush=True)
    Path(args.out).write_text(json.dumps(record))


def _splits(n_a: int, n_b: int):
    from itertools import combinations

    runs = range(n_a + n_b)
    for chosen in combinations(runs, n_a):
        rest = [i for i in runs if i not in chosen]
        yield list(chosen), rest


def _permutation_p(values: list[Any], n_a: int, statistic) -> float:
    """Share of the arm labellings whose statistic is at least the observed one."""
    observed = statistic(values[:n_a], values[n_a:])
    count = total = 0
    for a, b in _splits(n_a, len(values) - n_a):
        total += 1
        count += (
            statistic([values[i] for i in a], [values[i] for i in b])
            >= observed - 1e-12
        )
    return count / total


def _welch_t(a: list[float], b: list[float]) -> float:
    import math

    def mean_var(x):
        m = sum(x) / len(x)
        return m, sum((v - m) ** 2 for v in x) / max(1, len(x) - 1)

    ma, va = mean_var(a)
    mb, vb = mean_var(b)
    se = math.sqrt(va / len(a) + vb / len(b))
    return (mb - ma) / se if se > 0 else (0.0 if mb == ma else float("inf"))


def _tv(p: list[float], q: list[float]) -> float:
    return 0.5 * sum(abs(x - y) for x, y in zip(p, q))


def _mean_dist(rows: list[list[float]]) -> list[float]:
    return [sum(col) / len(rows) for col in zip(*rows)]


def compare_groups(
    a: list[dict[str, Any]], b: list[dict[str, Any]], *, skip: int = 300
) -> dict[str, Any]:
    """Arm A against arm B, one sample per run, after ``skip`` steps."""
    n_a = len(a)
    runs = [*a, *b]
    if n_a < 2 or len(b) < 2:
        raise ValueError("compare-groups needs at least two runs per arm.")
    rows = []
    names = [n for n in a[0]["series"] if not n.startswith("_")]
    samples: dict[str, list[float]] = {}
    for name in names:
        samples[name] = [
            sum(r["series"][name][skip:]) / len(r["series"][name][skip:]) for r in runs
        ]
    samples["episode_length_at_reset"] = [
        sum(r["series"]["_len_sum"][skip:])
        / max(1.0, sum(r["series"]["_len_count"][skip:]))
        for r in runs
    ]

    def gap(x, y):
        return abs(sum(y) / len(y) - sum(x) / len(x))

    for name, values in samples.items():
        rows.append(
            {
                "name": name,
                "a": sum(values[:n_a]) / n_a,
                "b": sum(values[n_a:]) / (len(values) - n_a),
                "t": _welch_t(values[:n_a], values[n_a:]),
                "p": _permutation_p(values, n_a, gap),
            }
        )
    hists = []
    for name in a[0]["segment_hist"]:
        dists = []
        for r in runs:
            first = skip // int(r["segment_steps"])
            counts = [sum(col) for col in zip(*r["segment_hist"][name][first:])]
            total = sum(counts)
            dists.append([c / total for c in counts] if total > 0 else counts)

        def between(x, y):
            return _tv(_mean_dist(x), _mean_dist(y))

        within = [
            _tv(dists[i], dists[j])
            for group in (range(n_a), range(n_a, len(runs)))
            for i in group
            for j in group
            if i < j
        ]
        hists.append(
            {
                "name": name,
                "tv": between(dists[:n_a], dists[n_a:]),
                "within_tv": sum(within) / len(within),
                "p": _permutation_p(dists, n_a, between),
                # One occupied bin means TV 0 for any two arms: no evidence.
                "bins_with_mass": sum(
                    1 for col in zip(*dists) if any(v > 0 for v in col)
                ),
            }
        )
    return {"scalars": rows, "hists": hists, "runs": [n_a, len(b)], "skip": skip}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    overrides: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, overrides = argv[:split], argv[split + 1 :]
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "sync", "time", "terminal", "writes"):
        mode = sub.add_parser(name, help=f"{name} mode (hydra overrides after --)")
        mode.add_argument("--out", required=True)
        mode.add_argument("--task", default="imitation-g1-v1")
        mode.add_argument("--num_envs", type=int, default=4096)
        mode.add_argument("--steps", type=int, default=600)
        mode.add_argument("--seed", type=int, default=0)
        mode.add_argument("--noise", type=float, default=0.3)
        if name == "time":
            # Past the reset sampler's transient (wiki/graph-env-step.md).
            mode.add_argument("--warmup", type=int, default=300)
    groups = sub.add_parser("compare-groups", help="compare two arms of several runs")
    groups.add_argument("--a", nargs="+", required=True)
    groups.add_argument("--b", nargs="+", required=True)
    groups.add_argument("--skip", type=int, default=300)
    groups.add_argument("--out", default="")
    cmp = sub.add_parser("compare", help="compare two run records")
    cmp.add_argument("a")
    cmp.add_argument("b")
    cmp.add_argument("--block", type=int, default=24)
    cmp.add_argument("--skip-blocks", type=int, default=1)
    cmp.add_argument("--z", type=float, default=4.0)
    cmp.add_argument("--p", type=float, default=1e-3)
    args = add_kebab_aliases(parser).parse_args(argv)
    modes = {
        "run": _run,
        "sync": _sync,
        "time": _time,
        "terminal": _terminal,
        "writes": _writes,
    }
    if args.command in modes:
        modes[args.command](args, overrides)
        return 0
    if args.command == "compare-groups":
        load = lambda paths: [json.loads(Path(x).read_text()) for x in paths]  # noqa: E731
        result = compare_groups(load(args.a), load(args.b), skip=args.skip)
        print(f"runs {result['runs']}, steps after {result['skip']}")
        print(f"{'scalar':40s} {'a':>11s} {'b':>11s} {'t':>7s} {'perm p':>7s}")
        for row in result["scalars"]:
            print(
                f"{row['name']:40s} {row['a']:11.5g} {row['b']:11.5g} "
                f"{row['t']:7.2f} {row['p']:7.3f}"
            )
        print(
            f"{'histogram':40s} {'TV a-b':>11s} {'TV within':>11s} {'perm p':>7s} "
            f"{'bins':>5s}"
        )
        for row in result["hists"]:
            print(
                f"{row['name']:40s} {row['tv']:11.4f} {row['within_tv']:11.4f} "
                f"{row['p']:7.3f} {row['bins_with_mass']:5d}"
            )
        if args.out:
            Path(args.out).write_text(json.dumps(result))
        return 0
    a = json.loads(Path(args.a).read_text())
    b = json.loads(Path(args.b).read_text())
    result = compare_statistics(a, b, block=args.block, skip_blocks=args.skip_blocks)
    failed = False
    print(f"{'scalar':40s} {'a':>11s} {'b':>11s} {'z':>7s}")
    for row in result["scalars"]:
        flag = abs(row["z"]) > args.z
        failed |= flag
        print(
            f"{row['name']:40s} {row['a']:11.5g} {row['b']:11.5g} "
            f"{row['z']:7.2f}{'  <--' if flag else ''}"
        )
    print(f"{'histogram':40s} {'TV':>11s} {'p':>11s}")
    for row in result["hists"]:
        flag = row["p"] < args.p
        failed |= flag
        print(
            f"{row['name']:40s} {row['tv']:11.4f} {row['p']:11.3g}"
            f"{'  <--' if flag else ''}"
        )
    print(f"max |z| {max(abs(r['z']) for r in result['scalars']):.2f}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
