"""The public ImitationLab command line. Help never starts the simulator."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

from imitationlab import __version__
from imitationlab.runners.errors import UserError, reported_by
from imitationlab.runners.registry import (
    DEFAULT_TASK,
    interface_arg,
    is_eager_task,
)

COMMANDS = {
    "train": "Train a G1 tracking policy.",
    "pretrain": "Pretrain an FSQ or spectral skill command encoder.",
    "evaluate": "Evaluate a checkpoint with its saved task and network settings.",
    "export": "Export a checkpoint and its encoder as a native bundle.",
    "data": "Prepare, inspect, and pack motion data.",
    "hardware": "Show recommended execution sizes for this machine.",
    "tune": "Measure training throughput and save the fastest valid sizes.",
    "benchmark": "Measure tracking quality against training time.",
}


def _execute(module: str, argv: list[str], dry_run: bool = False) -> int:
    command = [sys.executable, "-m", module, *argv]
    if dry_run:
        print(shlex.join(command))
        return 0
    os.execv(sys.executable, command)
    return 0


def _pretrain(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="imitationlab pretrain",
        description="Pretrain an FSQ or spectral skill command encoder.",
    )
    parser.add_argument(
        "--interface", required=True, type=interface_arg, choices=("fsq", "spectral")
    )
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output-dir", dest="output_dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--hardware",
        choices=("auto", "off", "small", "standard", "large", "xlarge"),
        default="auto",
    )
    args, rest = parser.parse_known_args(argv)
    with reported_by(parser):
        return _run_pretrain(parser, args, rest)


def _run_pretrain(parser: argparse.ArgumentParser, args, rest: list[str]) -> int:
    from imitationlab.paths import recipes_root
    from imitationlab.runners.artifacts import data_overrides

    hardware_args = []
    if not args.dry_run:
        os.environ.pop("IMITATIONLAB_PRETRAIN_HARDWARE", None)
    if args.hardware != "off" and not any(
        x == "--checkpoint" or x.startswith("--checkpoint=") for x in rest
    ):
        from imitationlab.runners.hardware import explain, plan_for_argv

        plan = plan_for_argv(
            [*data_overrides(args.data), *rest],
            interface=args.interface,
            stage="pretrain",
            profile=args.hardware,
        )
        print(explain(plan), file=sys.stderr)
        if plan.errors and not args.dry_run:
            parser.error(" ".join(plan.errors))
        if not plan.errors:
            hardware_args = plan.overrides
            if not args.dry_run:
                for key, value in plan.environment.items():
                    os.environ.setdefault(key, value)
                os.environ["IMITATIONLAB_PRETRAIN_HARDWARE"] = json.dumps(
                    plan.to_dict()
                )
    command = [
        "--args_file",
        str(recipes_root() / "encoders" / f"{args.interface}.args"),
    ]
    if args.output_dir:
        command.extend(("--output_dir", str(args.output_dir)))
    command.extend(data_overrides(args.data))
    return _execute(
        "imitationlab.runners.pretrain", [*command, *hardware_args, *rest], args.dry_run
    )


def _evaluate(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="imitationlab evaluate",
        description="Evaluate with the checkpoint's saved task and network settings.",
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--data", type=Path, help="Override the saved motion bank with evaluation data."
    )
    parser.add_argument(
        "--num-envs", "--num_envs", dest="num_envs", type=int, default=4
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/evaluation/result.json"),
        help="Result JSON path (default: %(default)s).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing result file at --output.",
    )
    parser.add_argument("--policy", choices=("ppo", "supermimic", "svg", "diffsr"))
    parser.add_argument("--dry-run", action="store_true")
    args, rest = parser.parse_known_args(argv)
    with reported_by(parser):
        return _run_evaluate(args, rest)


def _run_evaluate(args, rest: list[str]) -> int:
    from imitationlab.runners.artifacts import (
        MissingRunConfiguration,
        checkpoint_run_spec,
        data_overrides,
        flag_value,
        resolve_checkpoint,
        restore_runtime_environment,
    )

    if args.output.exists() and not args.overwrite and not args.dry_run:
        raise UserError(
            f"Output already exists: {args.output}; pass --overwrite or choose "
            "another --output."
        )
    checkpoint = resolve_checkpoint(args.checkpoint)
    try:
        saved = checkpoint_run_spec(checkpoint)
    except MissingRunConfiguration:
        if not args.policy or not args.data:
            raise
        saved = {"argv": [], "task": DEFAULT_TASK}
    previous = saved.get("argv", [])
    algo = (
        ("PPO" if saved["policy"] == "ppo" else "WMBPTT")
        if saved.get("policy")
        else flag_value(previous, "--algo", flag_value(previous, "--algorithm", "PPO"))
    )
    if args.policy:
        algo = "PPO" if args.policy == "ppo" else "WMBPTT"
    task = saved.get("task") or flag_value(previous, "--task", DEFAULT_TASK)
    seed = args.seed if args.seed is not None else saved.get("seed")
    if seed is None:
        seed = flag_value(previous, "--seed", 42)
    overrides = [
        item
        for item in previous
        if "=" in item
        and not item.startswith("--")
        and not item.startswith("agent.logger.")
    ]
    if args.policy:
        from imitationlab.runners.launch import POLICIES

        if saved.get("policy") and saved["policy"] != args.policy:
            raise UserError(
                f"Checkpoint learner is {saved['policy']}, not {args.policy}."
            )
        overrides.extend(POLICIES[args.policy][1])
    if args.data:
        overrides = [
            item
            for item in overrides
            if not item.startswith(
                (
                    "env.data.manifest=",
                    "env.data.reference_arrays_dir=",
                    "env.data.persist_id=",
                )
            )
        ]
    # Put Hydra assignments before unknown argparse flags such as nargs='+'
    # trajectory ranks, whose parser would otherwise try to consume them.
    extra_overrides = [
        item for item in rest if "=" in item and not item.startswith("--")
    ]
    extra_flags = [item for item in rest if item not in extra_overrides]
    evaluation_settings = []
    if is_eager_task(task):
        evaluation_settings = [
            "env.step_mode=eager",
            "env.wm_only_observations=false",
            "env.reset_fail_share=0.0",
            'env.reset_exclude_names_file=""',
            "env.persist_reset_stats=false",
            "env.reward_kernel_scale=1.0",
        ]
    entry = flag_value(previous, "--agent")
    if entry:
        extra_flags = ["--agent_entry_point", entry, *extra_flags]
    command = [
        "--checkpoint",
        str(checkpoint),
        "--task",
        task,
        "--algo",
        algo,
        "--headless",
        "--num_envs",
        str(args.num_envs),
        "--steps",
        str(args.steps),
        "--seed",
        str(seed),
        "--output_json",
        str(args.output),
        "physics=newton_mjwarp",
        *overrides,
        *evaluation_settings,
        "env.pinned_reset=true",
        "env.reset_noise=false",
        "env.observation_noise=false",
        "env.randomize_context=false",
        *data_overrides(args.data),
        *extra_overrides,
        *extra_flags,
    ]
    if not args.dry_run:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        restore_runtime_environment(saved)
    return _execute(
        "imitationlab.evaluation.evaluate_checkpoint", command, args.dry_run
    )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    spellings = {
        "--num-envs": "--num_envs",
        "--max-iterations": "--max_iterations",
        "--max-train-seconds": "--max_train_seconds",
        "--args-file": "--args_file",
    }
    argv = [
        spellings.get(value.split("=", 1)[0], value.split("=", 1)[0])
        + ("=" + value.split("=", 1)[1] if "=" in value else "")
        if value.startswith("--")
        else value
        for value in argv
    ]
    parser = argparse.ArgumentParser(
        prog="imitationlab",
        description="Motion data, command encoders, and imitation policy learning.",
        epilog="commands:\n"
        + "\n".join(f"  {name:<10} {text}" for name, text in COMMANDS.items())
        + "\n\nRun 'imitationlab <command> --help' for the options of a command.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"ImitationLab {__version__}"
    )
    parser.add_argument(
        "command",
        choices=tuple(COMMANDS),
        metavar="<command>",
        help="Command to run; see the list below.",
    )
    if not argv or argv[0].startswith("-"):
        parser.parse_args(argv)
        return 0
    command = parser.parse_args(argv[:1]).command
    try:
        return _dispatch(command, argv[1:])
    except UserError as exc:
        # Subcommands report their own input errors; this is the last resort
        # for one raised outside a subcommand parser.
        print(f"imitationlab {command}: error: {exc}", file=sys.stderr)
        return 2


def _dispatch(command: str, argv: list[str]) -> int:
    if command == "benchmark":
        from imitationlab.runners.benchmark import main as benchmark

        return benchmark(argv)
    if command in ("hardware", "tune"):
        from imitationlab.runners.tuning import main as hardware_cli

        return hardware_cli(argv, calibrate=command == "tune")
    if command == "train":
        from imitationlab.runners.launch import main as train

        train(argv)
        return 0
    if command == "pretrain":
        return _pretrain(argv)
    if command == "evaluate":
        return _evaluate(argv)
    if command == "export":
        from imitationlab.runners.export import main as export

        return export(argv)
    from imitationlab.data_tools.cli import main as data

    return data(argv)


if __name__ == "__main__":
    raise SystemExit(main())
