"""Bitwise equivalence of rollout collection with and without the fast paths.

``run`` builds the env and agent exactly as ``scripts/tracking/train.py`` does
(same argv, same checkpoint resume), then collects ``--iters`` batches with the
agent's own collector instead of training, and writes one SHA-256 prefix per
rollout leaf per env step (observations, actions, rewards, done flags, policy
outputs, trajectory ids) plus per-batch hashes of the reference cursors and the
return-based reset sampler state. ``compare`` reports the first divergence.

Two run-to-run nondeterminism sources are removed so that two processes can be
compared bit for bit; both arms get the same treatment:

- Physics. The Newton/MJWarp contact solve is not deterministic across
  processes (measured on the combo50b recipe: joint velocities differ by up to
  0.24 rad/s after one control step from the same state). ``--physics
  record:PATH`` stores the float arrays the scene reads after every
  ``sim.step`` (Newton ``state_0`` and the contact sensors' outputs);
  ``--physics replay:PATH`` overwrites them with the recording after each of
  its own ``sim.step`` calls, so everything downstream of physics sees the
  recorded trace. Self-check: two eager arms, one recording and one replaying,
  are identical over 240 steps.
- CUDA scans and atomics. ``torch.cumsum`` on a large float vector,
  ``torch.multinomial(replacement=True)`` (which builds its CDF with that
  scan) and float ``index_add_`` give different results across processes.
  The run enables ``torch.use_deterministic_algorithms(warn_only=True)`` and
  replaces ``torch.multinomial`` with an inverse-CDF draw on a sequential
  float64 CPU cumsum; production runs keep the original kernels.

Usage (from the repository root, isaaclab env)::

    python -m imitationlab.evaluation.validation.collect_equivalence run \\
        --out eager.json --iters 10 --physics record:/tmp/phys.pt -- <train.py argv>
    ISAACLAB_IMITATION_COLLECT_FASTPATH=1 IMITATIONLAB_COLLECT_FASTPATH=1 \\
    python -m imitationlab.evaluation.validation.collect_equivalence run \\
        --out fast.json --iters 10 --physics replay:/tmp/phys.pt -- <train.py argv>
    python -m imitationlab.evaluation.validation.collect_equivalence compare eager.json fast.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import sys
import time
from pathlib import Path
from typing import Any

from imitationlab.paths import REPO_ROOT

__all__ = ["compare_records", "main"]


def _hash(tensor: Any) -> str:
    import torch

    tensor = tensor.detach()
    if tensor.dtype == torch.bool:
        tensor = tensor.to(torch.uint8)
    return hashlib.sha256(tensor.contiguous().cpu().numpy().tobytes()).hexdigest()[:16]


def _install_deterministic_kernels() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.use_deterministic_algorithms(True, warn_only=True)
    original = torch.multinomial

    def multinomial(input, num_samples, replacement=False, *, generator=None, out=None):
        if not replacement or input.ndim != 1 or out is not None:
            return original(input, num_samples, replacement, generator=generator)
        cdf = torch.cumsum(input.detach().double().cpu(), 0)
        cdf = (cdf / cdf[-1]).to(input.device)
        u = torch.rand(
            num_samples, device=input.device, dtype=torch.float64, generator=generator
        )
        return torch.searchsorted(cdf, u, right=True).clamp_max(input.numel() - 1)

    torch.multinomial = multinomial


def _physics_arrays() -> list[tuple[str, Any]]:
    """Float arrays the scene reads after a physics step, in a stable order."""
    import warp as wp
    from isaaclab_newton.physics.newton_manager import NewtonManager

    def floats(obj: Any):
        for name, value in sorted(vars(obj).items()):
            if (
                isinstance(value, wp.array)
                and value.size > 0
                and not name.startswith("_")
                and wp.to_torch(value).dtype.is_floating_point
            ):
                yield name, value

    out = [(f"state_0.{n}", v) for n, v in floats(NewtonManager.backend.state_0)]
    sensors = sorted(
        NewtonManager._newton_contact_sensors.items(), key=lambda kv: str(kv[0])
    )
    for key, sensor in sensors:
        out += [(f"contact[{key}].{n}", v) for n, v in floats(sensor)]
    for i, sensor in enumerate(NewtonManager._newton_imu_sensors):
        out += [(f"imu[{i}].{n}", v) for n, v in floats(sensor)]
    return out


def _install_physics_trace(env: Any, spec: str):
    """Record or replay every ``sim.step`` result; returns a finish callback."""
    import torch
    import warp as wp

    mode, path = spec.split(":", 1)
    if mode not in {"record", "replay"}:
        raise ValueError(f"--physics must be record:PATH or replay:PATH, got {spec!r}.")
    state: dict[str, Any] = {"step": 0, "names": None, "frames": []}
    if mode == "replay":
        blob = torch.load(path, weights_only=False)
        state["names"], state["frames"] = blob["names"], blob["frames"]
    step = env.sim.step

    def traced_step(*args: Any, **kwargs: Any) -> Any:
        result = step(*args, **kwargs)
        arrays = _physics_arrays()
        names = [name for name, _ in arrays]
        if mode == "record":
            state["names"] = state["names"] or names
            state["frames"].append(
                [wp.to_torch(v).detach().clone().cpu() for _, v in arrays]
            )
        else:
            if names != state["names"]:
                raise RuntimeError("Physics array layout differs from the recording.")
            if state["step"] >= len(state["frames"]):
                raise RuntimeError("The replay ran past the recorded physics trace.")
            for (_, value), recorded in zip(arrays, state["frames"][state["step"]]):
                wp.to_torch(value).copy_(recorded)
        state["step"] += 1
        return result

    env.sim.step = traced_step

    def finish() -> dict[str, Any]:
        if mode == "record":
            torch.save({"names": state["names"], "frames": state["frames"]}, path)
        return {"physics_mode": mode, "physics_steps": state["step"]}

    return finish


def _collect(agent: Any, args: argparse.Namespace) -> None:
    import torch
    from torchrl.envs.utils import ExplorationType

    base = agent.env
    while hasattr(base, "base_env"):
        base = base.base_env
    env = base._env.unwrapped
    finish = _install_physics_trace(env, args.physics) if args.physics else None
    agent.collector.exploration_type = (
        ExplorationType.DETERMINISTIC
        if args.exploration == "deterministic"
        else ExplorationType.RANDOM
    )
    agent.validate_training()
    metadata = agent.init_metadata()
    record: dict[str, Any] = {"steps": [], "batches": [], "collect_ms": []}
    for k in range(args.iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        rollout = agent.collect(metadata, k).rollout
        torch.cuda.synchronize()
        record["collect_ms"].append((time.perf_counter() - start) * 1e3)
        keys = sorted(
            (key if isinstance(key, tuple) else (key,))
            for key in rollout.keys(True, True)
        )
        for t in range(rollout.shape[-1]):
            frame = rollout[..., t]
            record["steps"].append(
                {"/".join(key): _hash(frame.get(key)) for key in keys}
            )
        tm = env.trajectory_manager
        batch = {"traj_rank": _hash(tm.env_traj_rank), "env_step": _hash(tm.env_step)}
        sampler = getattr(env.reference_command, "_return_reset_sampler", None)
        if sampler is not None:
            batch["return_bins"] = _hash(sampler.normalized_return)
            batch["return_memory"] = _hash(sampler.memory)
            batch["env_bins"] = _hash(sampler.env_bin)
        truncated = rollout.get(("next", "truncated")).reshape(-1)
        terminated = rollout.get(("next", "terminated")).reshape(-1)
        batch["time_outs"] = int((truncated & ~terminated).sum())
        batch["dones"] = int(rollout.get(("next", "done")).sum())
        record["batches"].append(batch)
    record["num_envs"] = int(env.num_envs)
    if finish is not None:
        record.update(finish())
    Path(args.out).write_text(json.dumps(record))
    agent.collector.shutdown()


def _run(args: argparse.Namespace, train_argv: list[str]) -> None:
    if not args.nondeterministic:
        _install_deterministic_kernels()
    import imitationlab.algorithms.ppo.ppo as ppo_module

    ppo_module.PPO.train = lambda agent: _collect(agent, args)
    train = REPO_ROOT / "scripts" / "tracking" / "train.py"
    sys.path.insert(0, str(train.parent))
    sys.argv = [str(train), *train_argv]
    runpy.run_path(str(train), run_name="__main__")


def compare_records(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """Human-readable differences between two ``run`` records (empty if equal)."""
    problems: list[str] = []
    if len(a["steps"]) != len(b["steps"]):
        problems.append(f"step counts differ: {len(a['steps'])} vs {len(b['steps'])}")
    for t, (sa, sb) in enumerate(zip(a["steps"], b["steps"])):
        keys = sorted(set(sa) | set(sb))
        diff = [key for key in keys if sa.get(key) != sb.get(key)]
        if diff:
            problems.append(
                f"step {t}: {len(diff)}/{len(keys)} leaves differ, e.g. {diff[:8]}"
            )
            break
    for i, (ba, bb) in enumerate(zip(a["batches"], b["batches"])):
        diff = [key for key in sorted(set(ba) | set(bb)) if ba.get(key) != bb.get(key)]
        if diff:
            problems.append(f"batch {i}: {diff}")
            break
    return problems


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    train_argv: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, train_argv = argv[:split], argv[split + 1 :]
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser(
        "run", help="collect and hash a rollout (argv after -- goes to train.py)"
    )
    run.add_argument("--out", required=True)
    run.add_argument("--iters", type=int, default=10)
    run.add_argument(
        "--exploration", choices=("deterministic", "random"), default="deterministic"
    )
    run.add_argument("--physics", default="", help="record:PATH or replay:PATH")
    run.add_argument(
        "--nondeterministic",
        action="store_true",
        help="keep the stock CUDA scan/multinomial kernels (timing runs only)",
    )
    cmp = sub.add_parser("compare", help="compare two run records")
    cmp.add_argument("a")
    cmp.add_argument("b")
    args = parser.parse_args(argv)
    if args.command == "run":
        _run(args, train_argv)
        return 0
    a = json.loads(Path(args.a).read_text())
    b = json.loads(Path(args.b).read_text())
    problems = compare_records(a, b)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"IDENTICAL: {len(a['steps'])} env steps x {len(a['steps'][0])} leaves")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
