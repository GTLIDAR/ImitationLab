"""Bitwise equivalence of two G1 env configurations, without an agent.

``collect_equivalence`` needs a checkpoint and a TorchRL collector, and the G1 env runs
neither: WMBPTT steps the env directly. ``run`` builds the env as the training
entrypoint does, drives it with actions from a seeded CPU generator, and writes
one SHA-256 prefix per tensor per control step:

- ``obs/<group>/<term>``: the returned observations;
- ``final/<group>/<term>``, ``final/_env_ids``: the batched terminal rows;
- ``out/*``: reward, terminated, truncated;
- ``g1/*``: cursors, histories, context and the logged reward parts;
- ``sampler/*``: every tensor of the reset sampler;
- ``rng/*``: the CPU and CUDA generator states.

``imported`` names the file of each package under test, so a caller that points
``PYTHONPATH`` at an older tree can check that the run imported it.

``compare`` reports the first difference. A run with fewer observation groups is
compared on the leaves both records hold, and ``--require`` names the prefixes
that must be present in both, so an empty intersection cannot pass.

Physics replay and the deterministic scan and multinomial kernels are
``collect_equivalence``'s; see its docstring for why both are needed.

Usage (from the repository root, isaaclab env)::

    python -m imitationlab.evaluation.validation.graph_step_equivalence run --out a.json \\
        --steps 200 --physics record:/tmp/phys.pt -- <hydra overrides>
    python -m imitationlab.evaluation.validation.graph_step_equivalence run --out b.json \\
        --steps 200 --physics replay:/tmp/phys.pt -- env.wm_only_observations=true \\
        <hydra overrides>
    python -m imitationlab.evaluation.validation.graph_step_equivalence compare a.json b.json \\
        --require obs/wm --require final/wm
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from imitationlab.evaluation.validation.collect_equivalence import (
    _hash,
    _install_deterministic_kernels,
    _install_physics_trace,
)

__all__ = ["compare_records", "imported_packages", "main"]

_SEED = 137
_ACTION_STD = 0.3


def _leaves(value: Any, prefix: str) -> Iterable[tuple[str, Any]]:
    import torch

    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, Mapping) or hasattr(value, "items"):
        for key, item in value.items():
            yield from _leaves(item, f"{prefix}/{key}")


def _record_step(
    env: Any, obs: Any, extras: Mapping[str, Any] | None
) -> dict[str, str]:
    import torch

    row = {name: _hash(tensor) for name, tensor in _leaves(obs, "obs")}
    final = (extras or {}).get("final_obs")
    if final is not None:
        row["final/_env_ids"] = _hash(final["_env_ids"])
        row.update({n: _hash(t) for n, t in _leaves(final["obs"], "final")})
    rank, frame = env.g1_cursor()
    row["g1/rank"], row["g1/frame"] = _hash(rank), _hash(frame)
    row["g1/episode_step"] = _hash(env.episode_length_buf)
    for name in ("_g1_states", "_g1_actions", "_g1_proprio", "_g1_context"):
        row[f"g1/{name}"] = _hash(getattr(env, name))
    row.update({f"g1/reward/{k}": _hash(v) for k, v in env._g1_reward_parts.items()})
    for name, value in sorted(vars(env.g1_reset_sampler).items()):
        if isinstance(value, torch.Tensor):
            row[f"sampler/{name}"] = _hash(value)
    row["rng/cpu"] = _hash(torch.get_rng_state())
    row["rng/cuda"] = _hash(torch.cuda.get_rng_state(env.device))
    return row


def _watch_failure_frames(env: Any, counts: dict[str, int]) -> None:
    """Count episode ends whose bookkeeping disagrees with the live cursor.

    ``reset_fail_share`` derives the ending frame from the start frame and the
    step count. The cursor at a reset holds the frame after it.

    The eager reset ends episodes with ``end_episodes`` (indices), the masked
    and graph steps with ``end_episodes_masked`` (the reset mask), so both are
    watched; ``frame_checks`` counts the calls checked. A CUDA graph replays
    its kernels without calling Python, and a host read would break its
    capture, so in the graph arm only the steps before the capture are
    checked. That arm must equal the masked arm bit for bit on every tensor
    the check reads.
    """
    import torch

    sampler = env.g1_reset_sampler
    if not getattr(sampler, "fail_share", 0.0):
        return

    def count(ended: Any, derived: Any, cursor: Any) -> None:
        counts["frame_checks"] += 1
        counts["ended_rows"] += int(ended.sum())
        counts["frame_mismatches"] += int((ended & (derived != cursor)).sum())

    end_episodes = sampler.end_episodes

    def checked(env_ids: Any, terminated: Any, **kwargs: Any) -> Any:
        _, cursor = env.g1_cursor()
        count(
            sampler.ep_step[env_ids] > 0,
            sampler.start_frame[env_ids] + sampler.ep_step[env_ids],
            cursor[env_ids],
        )
        return end_episodes(env_ids, terminated, **kwargs)

    sampler.end_episodes = checked
    end_masked = getattr(sampler, "end_episodes_masked", None)
    if end_masked is None:  # a base tree from before the masked forms
        return

    def checked_masked(mask: Any, terminated: Any) -> Any:
        if not torch.cuda.is_current_stream_capturing():
            _, cursor = env.g1_cursor()
            count(
                mask & (sampler.ep_step > 0),
                sampler.start_frame + sampler.ep_step,
                cursor,
            )
        return end_masked(mask, terminated)

    sampler.end_episodes_masked = checked_masked


def _drive(env: Any, args: argparse.Namespace) -> dict[str, Any]:
    import torch

    finish = _install_physics_trace(env, args.physics) if args.physics else None
    watched = {"ended_rows": 0, "frame_mismatches": 0, "frame_checks": 0}
    _watch_failure_frames(env, watched)
    actions = _ACTION_STD * torch.randn(
        args.steps,
        env.num_envs,
        29,
        generator=torch.Generator(device="cpu").manual_seed(_SEED + 50001),
    )
    obs, _ = env.reset(seed=_SEED)
    record: dict[str, Any] = {"steps": [], "num_envs": int(env.num_envs)}
    record["reset"] = _record_step(env, obs, None)
    counts = {"resets": 0, "terminated": 0, "truncated": 0}
    with torch.no_grad():
        for action in actions:
            obs, reward, terminated, truncated, extras = env.step(action.to(env.device))
            row = _record_step(env, obs, extras)
            row["out/reward"] = _hash(reward)
            row["out/terminated"] = _hash(terminated)
            row["out/truncated"] = _hash(truncated)
            record["steps"].append(row)
            counts["terminated"] += int(terminated.sum())
            counts["truncated"] += int(truncated.sum())
            final = extras.get("final_obs")
            counts["resets"] += 0 if final is None else int(final["_env_ids"].numel())
    record.update(counts)
    record.update(watched)
    record["prefetch_mode"] = env.expert_data_plane.reference_prefetch_mode
    bin_fail = getattr(env.g1_reset_sampler, "bin_fail", None)
    record["bin_fail_mass"] = 0.0 if bin_fail is None else float(bin_fail.sum())
    if finish is not None:
        record.update(finish())
    return record


def imported_packages() -> dict[str, str]:
    """The file each package under test came from.

    A ``PYTHONPATH`` entry that misses falls through to the editable install
    without an error, so a run cannot infer its tree from what it was told.
    """
    import imitationlab
    import imitationlab.envs.isaaclab

    return {
        package.__name__: str(Path(package.__file__).resolve())
        for package in (imitationlab, imitationlab.envs.isaaclab)
    }


def _run(args: argparse.Namespace, overrides: list[str]) -> None:
    from imitationlab.runners.runtime_bootstrap import install_kit_import_guard

    install_kit_import_guard()
    _install_deterministic_kernels()
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
    env_cfg.seed = _SEED
    if needs_kit(env_cfg, launcher_args):
        raise RuntimeError("This harness needs the kitless Newton/MJWarp runtime.")
    with launch_simulation(env_cfg, launcher_args):
        torch.manual_seed(_SEED)
        env = gym.make(args.task, cfg=env_cfg).unwrapped
        try:
            record = _drive(env, args)
        finally:
            env.close()
    record["imported"] = imported_packages()
    Path(args.out).write_text(json.dumps(record))


def compare_records(
    a: dict[str, Any], b: dict[str, Any], require: Iterable[str] = ()
) -> list[str]:
    """Differences between two ``run`` records (empty if equal).

    Leaves held by both records must match. Without ``require`` the two records
    must also hold the same leaves; with it, each named prefix must be present
    in both, and leaves one record lacks are tolerated.
    """
    required = tuple(require)
    problems: list[str] = []
    if len(a["steps"]) != len(b["steps"]):
        problems.append(f"step counts differ: {len(a['steps'])} vs {len(b['steps'])}")
    seen: set[str] = set()
    for t, (sa, sb) in enumerate(
        zip([a["reset"], *a["steps"]], [b["reset"], *b["steps"]])
    ):
        common = set(sa) & set(sb)
        seen |= common
        if not required and set(sa) != set(sb):
            problems.append(f"step {t}: leaves differ: {sorted(set(sa) ^ set(sb))[:8]}")
            break
        diff = sorted(key for key in common if sa[key] != sb[key])
        if diff:
            problems.append(f"step {t}: {len(diff)}/{len(common)} differ: {diff[:8]}")
            break
    for prefix in required:
        if not any(key.startswith(prefix) for key in seen):
            problems.append(f"no compared leaf starts with {prefix!r}")
    return problems


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    overrides: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, overrides = argv[:split], argv[split + 1 :]
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser(
        "run", help="step the env and hash it (hydra overrides after --)"
    )
    run.add_argument("--out", required=True)
    run.add_argument("--task", default="imitation-g1-v1")
    run.add_argument("--num-envs", type=int, default=256)
    run.add_argument("--steps", type=int, default=200)
    run.add_argument("--physics", default="", help="record:PATH or replay:PATH")
    cmp = sub.add_parser("compare", help="compare two run records")
    cmp.add_argument("a")
    cmp.add_argument("b")
    cmp.add_argument("--require", action="append", default=[])
    args = parser.parse_args(argv)
    if args.command == "run":
        _run(args, overrides)
        return 0
    a = json.loads(Path(args.a).read_text())
    b = json.loads(Path(args.b).read_text())
    problems = compare_records(a, b, args.require)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"IDENTICAL: {len(a['steps'])} env steps, {len(a['steps'][0])} leaves in a")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
