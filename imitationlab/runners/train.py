"""Runtime-aware ImitationLab training entrypoint for Isaac Lab 3."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from imitationlab.cli_args import add_kebab_aliases, canonical_argv
from imitationlab.runners.provenance.config_keys import assert_no_unknown_config_keys
from imitationlab.runners.registry import resolve_agent_cfg_entry_point
from imitationlab.runners.runtime_bootstrap import (
    assert_kit_not_loaded,
    config_contains_type_name,
    install_kit_import_guard,
)

logger = logging.getLogger(__name__)

ALGORITHMS = (
    "PPO",
    "WMBPTT",
)


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI after an optional strict no-Kit guard is installed."""
    from imitationlab.runners.launch_compat import add_launcher_args

    parser = argparse.ArgumentParser(
        description="Train an ImitationLab agent with Isaac Lab."
    )
    parser.add_argument(
        "--video",
        action="store_true",
        default=False,
        help="Record videos during training.",
    )
    parser.add_argument(
        "--video_length",
        type=int,
        default=200,
        help="Video length in environment steps.",
    )
    parser.add_argument(
        "--video_interval",
        type=int,
        default=2000,
        help="Steps between video recordings.",
    )
    parser.add_argument(
        "--video_width", type=int, default=None, help="Optional render width override."
    )
    parser.add_argument(
        "--video_height",
        type=int,
        default=None,
        help="Optional render height override.",
    )
    parser.add_argument(
        "--num_envs", type=int, default=None, help="Number of simulated environments."
    )
    parser.add_argument(
        "--task", type=str, default=None, help="Registered Isaac Lab task name."
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="rlopt_cfg_entry_point",
        help="ImitationLab agent configuration entry point.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Environment and agent seed."
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=None,
        help="Metric cadence in environment steps.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Checkpoint to resume."
    )
    parser.add_argument(
        "--max_iterations", type=int, default=None, help="Training rollout iterations."
    )
    parser.add_argument(
        "--max_train_seconds",
        type=float,
        default=None,
        help=(
            "Wall-clock training budget in seconds (agent.trainer.max_train_seconds). "
            "The clock starts after the first collected batch, so env/agent "
            "construction and the first-step cold start are outside it. At the "
            "budget the run stops at the next iteration boundary, writes a final "
            "checkpoint named by cumulative env frames, and exits 0. The frame "
            "budget still applies; whichever ends first stops the run."
        ),
    )
    parser.add_argument(
        "--profile_iterations",
        action="store_true",
        default=False,
        help=(
            "Synchronize CUDA at phase boundaries and report timings for every "
            "training iteration."
        ),
    )
    parser.add_argument("--export_io_descriptors", action="store_true", default=False)
    parser.add_argument(
        "--algo",
        "--algorithm",
        dest="algorithm",
        type=str.upper,
        default="PPO",
        choices=ALGORITHMS,
        help="ImitationLab algorithm; it must match a task config entry point.",
    )
    parser.add_argument(
        "--assert-kitless",
        action="store_true",
        help="Require Newton and fail if Isaac Sim or Omniverse Kit is imported.",
    )
    add_launcher_args(parser)
    if "--distributed" not in parser._option_string_actions:
        parser.add_argument("--distributed", action="store_true")
    parser.add_argument(
        "--distributed_backend",
        choices=("nccl", "gloo"),
        default="nccl",
        help=(
            "torch.distributed backend for --distributed. nccl (default) needs one "
            "GPU per worker; gloo lets workers share a GPU (local functional tests)."
        ),
    )
    parser.add_argument(
        "--require_distributed_smoke",
        help="Require completed multi-worker smoke health records before startup.",
    )
    parser.add_argument(
        "--max_training_seconds",
        type=float,
        default=None,
        help="Distributed qualification wall-clock cap, checked between iterations.",
    )
    # setup_preset_cli chooses the help formatter, so both spellings are listed.
    return add_kebab_aliases(parser, hide_snake_case=False)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    from isaaclab_tasks.utils import setup_preset_cli

    parser = _build_parser()
    args_cli, hydra_args = setup_preset_cli(parser, canonical_argv(argv, parser))
    sys.argv = [sys.argv[0]] + hydra_args
    return args_cli


def _validate_newton_robot_asset(env_cfg: object) -> None:
    """Fail before launch when a kit-less Newton robot USD is unavailable."""
    if not config_contains_type_name(env_cfg, "NewtonCfg"):
        return
    scene = getattr(env_cfg, "scene", None)
    robot = getattr(scene, "robot", None)
    spawn = getattr(robot, "spawn", None)
    usd_path = getattr(spawn, "usd_path", None)
    if not isinstance(usd_path, str) or "://" in usd_path:
        return
    if not Path(usd_path).is_file():
        raise FileNotFoundError(
            "The kit-less Newton path requires the G1 USD root layer, which "
            f"does not exist: {usd_path}. The repo packages the official "
            "Unitree asset via git-lfs; run `git lfs pull` to materialize it."
        )


def run(argv: list[str] | None = None, *, require_running_kit: bool = False) -> int:
    """Resolve configuration, own the simulation context, and run training."""
    if argv is None:
        argv = sys.argv[1:]
    # --args_file PATH: a file of tokens, expanded in place (imitationlab/runners/args_file.py).
    from imitationlab.runners.args_file import expand_args_files

    argv = expand_args_files(argv)
    launch_argv = list(argv)
    strict_kitless = "--assert-kitless" in argv
    if strict_kitless:
        install_kit_import_guard()

    args_cli = _parse_args(argv)
    args_cli.launch_argv = launch_argv
    if args_cli.task is None:
        raise ValueError("--task is required for ImitationLab training.")
    # IMITATION_EMULATE_GPU_GIB: hold the rest of a large GPU before anything else
    # allocates, so the run sees a small card's memory (gpu_emulation.py).
    from imitationlab.runners.gpu_emulation import apply_from_env as _emulate_gpu

    _emulate_gpu()

    # Task registration is intentionally after the optional Kit-first bootstrap.
    import imitationlab.envs.isaaclab.tasks  # noqa: F401
    import isaaclab_tasks  # noqa: F401
    from isaaclab.app import launch_simulation
    from isaaclab.utils import has_kit
    from imitationlab.runners.launch_compat import needs_kit as config_needs_kit
    from isaaclab_tasks.utils import resolve_task_config

    args_cli.agent = resolve_agent_cfg_entry_point(
        args_cli.task, args_cli.algorithm, args_cli.agent
    )
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent)
    # An override on a key the config does not declare is accepted by the Hydra
    # merge as a plain attribute and then read by nobody.
    assert_no_unknown_config_keys(agent_cfg)
    assert_no_unknown_config_keys(env_cfg, prefix="env")
    needs_kit = config_needs_kit(env_cfg, args_cli)
    _validate_newton_robot_asset(env_cfg)

    if strict_kitless:
        if needs_kit or not config_contains_type_name(env_cfg, "NewtonCfg"):
            raise RuntimeError(
                "--assert-kitless requires a resolved NewtonCfg with no Kit cameras or Kit visualizer. "
                "Pass physics=newton_mjwarp and a kit-less renderer."
            )
        assert_kit_not_loaded()
        print(
            "[INFO] Strict kit-less Newton runtime validated before simulation startup."
        )

    if require_running_kit and not has_kit():
        raise RuntimeError(
            "The PhysX bootstrap did not start SimulationApp before ImitationLab configuration loading."
        )
    if os.environ.get("ISAACLAB_SPLIT_RUNTIME") == "1" and needs_kit and not has_kit():
        raise RuntimeError(
            "The split runtime requires PhysX to start through scripts/tracking/train_physx.py "
            "with /isaac-sim/python.sh."
        )

    from imitationlab.runners.distributed.runtime import configure_runtime

    if getattr(args_cli, "distributed", False) and needs_kit:
        raise ValueError("Distributed qualification currently requires kit-less Newton")
    configure_runtime(args_cli, env_cfg, agent_cfg)
    try:
        with launch_simulation(env_cfg, args_cli):
            from imitationlab.runners.train_impl import train

            train(env_cfg, agent_cfg, args_cli)
    finally:
        if getattr(args_cli, "distributed", False):
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()

    if strict_kitless:
        assert_kit_not_loaded()
        print("[INFO] Strict kit-less invariant held through ImitationLab shutdown.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI wrapper that preserves a nonzero exit code through Isaac shutdown."""
    try:
        return run(argv)
    except Exception:
        logger.exception("Unhandled exception during ImitationLab training.")
        logging.shutdown()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
