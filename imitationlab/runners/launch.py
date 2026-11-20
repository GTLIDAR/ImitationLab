#!/usr/bin/env python3
"""Train a G1 tracker with independent command and policy choices."""

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

from imitationlab.cli_args import TRAIN_MODULES, canonical_argv, snake_case_options
from imitationlab.paths import recipes_root
from imitationlab.runners.artifacts import (
    MissingRunConfiguration,
    PACKED_DATA_DEFAULTS,
    checkpoint_run_spec,
    data_overrides,
    describe_run,
    flag_value,
    resolve_checkpoint,
    restore_runtime_environment,
)
from imitationlab.runners.errors import reported_by
from imitationlab.runners.registry import (
    DEFAULT_TASK,
    canonical_interface,
    interface_arg,
)


POLICIES = {
    "ppo": ("PPO", ()),
    "supermimic": ("WMBPTT", ("agent.wm.actor_source=imagined",)),
    "svg": ("WMBPTT", ("agent.wm.actor_source=real",)),
    "diffsr": (
        "WMBPTT",
        (
            "agent.wm.actor_source=imagined",
            "agent.wm.dynamics_model=diffsr",
            "agent.wm.wm_opt=adam",
        ),
    ),
}
ENCODERS = {"spectral": "deterministic", "fsq": "sonic_fsq"}
TRAINING_CONFIGS = recipes_root() / "g1"


def uses_packed_motions(overrides: list[str]) -> bool:
    if any(arg == "--args_file" or arg.startswith("--args_file=") for arg in overrides):
        from imitationlab.runners.args_file import expand_args_files

        overrides = expand_args_files(overrides, environ={})
    key = "env.data.reference_arrays_dir="
    value = next(
        (arg[len(key) :] for arg in reversed(overrides) if arg.startswith(key)), ""
    )
    return value.strip("'\"").lower() not in ("", "null", "none")


def _saved_selection(spec: dict) -> dict:
    previous = spec["argv"]
    overrides = dict(
        item.split("=", 1)
        for item in previous
        if item.startswith(("env.skill_encoder_", "agent.wm.")) and "=" in item
    )
    algo_flags = [
        item.replace("--algorithm", "--algo", 1)
        if item == "--algorithm" or item.startswith("--algorithm=")
        else item
        for item in previous
    ]
    policy = spec.get("policy")
    if policy is None:
        policy = "ppo"
        if flag_value(algo_flags, "--algo", "PPO").upper() == "WMBPTT":
            policy = "supermimic"
            if overrides.get("agent.wm.dynamics_model", "").strip("'\"") == "diffsr":
                policy = "diffsr"
            elif overrides.get("agent.wm.actor_source", "").strip("'\"") == "real":
                policy = "svg"

    encoder = (spec.get("encoder") or {}).get("path")
    if encoder is None:
        encoded_path = overrides.get("env.skill_encoder_path", "").strip("'\"")
        if encoded_path.lower() not in ("", "null", "none"):
            encoder = encoded_path
    interface = spec.get("interface")
    if interface is not None:
        interface = canonical_interface(interface)
    else:
        mode = overrides.get("env.skill_encoder_mode", "").strip("'\"")
        interface = (
            ("fsq" if mode == ENCODERS["fsq"] else "spectral")
            if encoder
            else "explicit"
        )
    return {
        "task": spec.get("task") or flag_value(previous, "--task"),
        "policy": policy,
        "interface": interface,
        "encoder": Path(encoder).expanduser().resolve() if encoder else None,
    }


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="imitationlab train",
        description=__doc__,
        epilog="Additional arguments are passed to the Isaac Lab trainer.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--interface",
        type=interface_arg,
        choices=("explicit", *ENCODERS),
        help="Command interface. Defaults to explicit; spectral and fsq need --encoder.",
    )
    parser.add_argument(
        "--policy",
        choices=POLICIES,
        help="Policy learner. Defaults to supermimic.",
    )
    parser.add_argument(
        "--encoder", type=Path, help="Pretrained spectral or FSQ encoder checkpoint."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the command and exit."
    )
    parser.add_argument("--task", help=f"Task ID (default: {DEFAULT_TASK}).")
    parser.add_argument(
        "--hardware",
        choices=("auto", "off", "small", "standard", "large", "xlarge"),
        default="auto",
        help="Choose execution sizes from local resources (default: auto); off keeps recipe sizes.",
    )
    parser.add_argument(
        "--tuning",
        type=Path,
        help="Use a validated selection produced by imitationlab tune.",
    )
    parser.add_argument(
        "--data", type=Path, help="Motion manifest or packed reference bank."
    )
    parser.add_argument(
        "--checkpoint", type=Path, help="Resume using the saved run settings."
    )
    args, overrides = parser.parse_known_args(argv)
    # The trainer declares its options in snake case; accept kebab case too.
    overrides = canonical_argv(overrides, snake_case_options(*TRAIN_MODULES))
    with reported_by(parser):
        _launch(parser, args, overrides)


def _launch(parser: argparse.ArgumentParser, args, overrides: list[str]) -> None:
    try:
        overrides = [*data_overrides(args.data), *overrides]
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))

    if args.checkpoint is not None:
        if args.tuning or args.hardware not in ("auto", "off"):
            parser.error(
                "Resume restores the saved execution settings; use explicit size overrides instead of a new hardware selection."
            )
        checkpoint = resolve_checkpoint(args.checkpoint)
        try:
            spec = checkpoint_run_spec(checkpoint)
        except MissingRunConfiguration:
            if args.policy is None or args.data is None:
                raise
        else:
            for name, saved in _saved_selection(spec).items():
                requested = getattr(args, name)
                if isinstance(requested, Path):
                    requested = requested.expanduser().resolve()
                if requested is not None and requested != saved:
                    parser.error(
                        f"--{name} {str(requested)!r} conflicts with the checkpoint's "
                        f"saved {name} {str(saved)!r}."
                    )
            previous = spec["argv"]
            # A resumed segment must load the user's chosen checkpoint, not an older one.
            filtered = []
            skip = False
            for item in previous:
                if skip:
                    skip = False
                    continue
                if item == "--checkpoint":
                    skip = True
                    continue
                if not item.startswith("--checkpoint="):
                    filtered.append(item)
            command = [
                sys.executable,
                "-m",
                "imitationlab.runners.train",
                *filtered,
                "--checkpoint",
                str(checkpoint),
                *(
                    ["--seed", str(spec["seed"])]
                    if spec.get("seed") is not None
                    else []
                ),
                *overrides,
            ]
            if args.dry_run:
                print(shlex.join(command))
                return
            os.environ["IMITATIONLAB_RUN_SPEC"] = json.dumps(spec)
            restore_runtime_environment(spec)
            os.execv(sys.executable, command)
            return
        overrides = ["--checkpoint", str(checkpoint), *overrides]

    args.interface = args.interface or "explicit"
    args.policy = args.policy or "supermimic"
    args.task = args.task if args.task is not None else DEFAULT_TASK

    goal_args = []
    if args.interface in ENCODERS:
        if args.encoder is None:
            parser.error(f"--interface {args.interface} requires --encoder.")
        # Hydra treats characters such as '=' and spaces as syntax in bare values.
        goal_args = [
            f"env.skill_encoder_path={json.dumps(str(args.encoder.expanduser().resolve()))}",
            f"env.skill_encoder_mode={ENCODERS[args.interface]}",
        ]
    elif args.encoder is not None:
        parser.error("--encoder requires --interface spectral or fsq.")

    algorithm, policy_args = POLICIES[args.policy]
    if algorithm == "WMBPTT":
        policy_args = (
            "agent.wm.actor_estimator=pathwise",
            "agent.wm.critic_source=imagined",
            *policy_args,
        )
    defaults = []
    if (
        args.interface != "explicit"
        and args.task == DEFAULT_TASK
        and uses_packed_motions(overrides)
    ):
        # Latent commands keep eager stepping, but need the same bounded-memory
        # reference storage as explicit commands for full BONES-SEED banks.
        defaults = [
            *PACKED_DATA_DEFAULTS,
            "env.library_host_mapped=true",
            "env.library_env_window=64",
        ]
    if args.interface == "explicit" and uses_packed_motions(overrides):
        defaults = ["--args_file", str(TRAINING_CONFIGS / "packed.args")]
    if args.task == DEFAULT_TASK or (
        args.interface == "explicit" and uses_packed_motions(overrides)
    ):
        # Command interfaces and data storage do not change learner defaults.
        # In particular, latent learners must not fall back to 500 world-model
        # warmup iterations and consume a short run without updating the actor.
        if algorithm == "WMBPTT":
            defaults += ["--args_file", str(TRAINING_CONFIGS / "world_model.args")]
        if args.policy == "supermimic":
            defaults += ["--args_file", str(TRAINING_CONFIGS / "supermimic.args")]
    if args.checkpoint is None and args.task == DEFAULT_TASK and args.policy == "ppo":
        defaults += ["--args_file", str(TRAINING_CONFIGS / "ppo.args")]
    hardware_args = []
    hardware_plan = None
    if args.hardware != "off" or args.tuning:
        from imitationlab.runners.hardware import explain, plan_for_argv

        if args.task == DEFAULT_TASK and int(os.environ.get("WORLD_SIZE", "1")) == 1:
            hardware_plan = plan_for_argv(
                overrides,
                policy=args.policy,
                interface=args.interface,
                profile=args.hardware,
            )
            if args.tuning:
                from imitationlab.runners.tuning import load_selection

                hardware_plan = load_selection(
                    args.tuning, hardware_plan, overrides, args.encoder
                )
            print(explain(hardware_plan), file=sys.stderr)
            if hardware_plan.errors and not args.dry_run:
                parser.error(" ".join(hardware_plan.errors))
            if not hardware_plan.errors:
                hardware_args = hardware_plan.overrides
                if not args.dry_run:
                    for key, value in hardware_plan.environment.items():
                        os.environ.setdefault(key, value)
        elif args.tuning or args.hardware != "auto":
            parser.error(
                f"Hardware selection currently supports single-GPU {DEFAULT_TASK} runs."
            )
    command = [
        sys.executable,
        "-m",
        "imitationlab.runners.train",
        "--task",
        args.task,
        "--algo",
        algorithm,
        "--headless",
        "physics=newton_mjwarp",
        *goal_args,
        *defaults,
        # Learner settings override the shared defaults. CLI overrides come last.
        *policy_args,
        *hardware_args,
        *overrides,
    ]
    if args.dry_run:
        print(shlex.join(command))
    else:
        try:
            spec = describe_run(
                command[3:],
                policy=args.policy,
                interface=args.interface,
                task=args.task,
                encoder=str(args.encoder.expanduser().resolve())
                if args.encoder
                else None,
            )
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        spec["hardware_selection"] = (
            hardware_plan.to_dict() if hardware_plan else {"mode": "off_or_preserved"}
        )
        os.environ["IMITATIONLAB_RUN_SPEC"] = json.dumps(spec)
        os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
