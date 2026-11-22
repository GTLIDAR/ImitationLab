"""Inspect a hardware recommendation or calibrate disposable local training runs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading

from imitationlab.cli_args import (
    PRETRAIN_MODULES,
    TRAIN_MODULES,
    canonical_argv,
    snake_case_options,
)
from imitationlab.runners.registry import (
    INTERFACES,
)
from imitationlab.runners.errors import UserError, reported_by
from imitationlab.runners.registry import DEFAULT_TASK
from imitationlab.runners.args_file import expand_args_files
from imitationlab.runners.artifacts import (
    RUNTIME_ENVIRONMENT,
    data_overrides,
    file_sha256,
    resolve_checkpoint,
)
from imitationlab.runners.hardware import (
    PROFILE_NAMES,
    HardwarePlan,
    bank_from_argv,
    choose_plan,
    detect_hardware,
    explain,
    selected_device,
)


def source_fingerprint() -> str:
    from imitationlab.paths import recipes_root

    package = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for root, pattern in ((package, "*.py"), (recipes_root(), "*.args")):
        for path in sorted(root.rglob(pattern)):
            if "tests" not in path.parts:
                digest.update(str(path.relative_to(root)).encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()


def context_signature(
    plan: HardwarePlan, argv: list[str], encoder: Path | None
) -> dict:
    # Logging destination, seed, and stopping budgets do not define the workload.
    excluded_flags = {"--seed", "--device", "--max_iterations", "--max_train_seconds"}
    normalized, skip = [], False
    environment = dict(os.environ)
    for item in expand_args_files(argv, environ=environment):
        if skip:
            skip = False
            continue
        flag = item.split("=", 1)[0]
        if flag in excluded_flags:
            skip = "=" not in item
            continue
        if item.startswith(
            (
                "env.data.manifest=",
                "env.data.reference_arrays_dir=",
                "env.data.persist_id=",
                "env.data.persist_dir=",
                "env.data.cache_dir=",
                "agent.logger.",
                "agent.save_interval=",
                "agent.collector.total_frames=",
                "agent.trainer.max_train_seconds=",
            )
        ):
            continue
        normalized.append(item)
    packages = {}
    for name in ("torch", "torchrl", "tensordict", "isaaclab", "newton", "mujoco-warp"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    gpu = plan.hardware
    return {
        "schema_version": 1,
        "source_sha256": source_fingerprint(),
        "packages": packages,
        "environment": {
            key: environment.get(key)
            for key in (*RUNTIME_ENVIRONMENT, "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "hardware": {
            "gpu": gpu.gpu_name,
            "vram_total": gpu.vram_total,
            "compute_capability": list(gpu.compute_capability or ()),
            "torch": gpu.torch_version,
            "cuda": gpu.cuda_version,
            "cpus": gpu.cpu_count,
            "ram_total": gpu.ram_total,
        },
        "policy": plan.policy,
        "interface": plan.interface,
        "bank_sha256": plan.bank.manifest_sha256,
        "encoder_sha256": file_sha256(encoder) if encoder else None,
        "overrides": normalized,
    }


def load_selection(
    path: Path, plan: HardwarePlan, argv: list[str], encoder: Path | None
) -> HardwarePlan:
    try:
        saved = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise UserError(f"Cannot read tuning selection {path}: {exc}") from exc
    context = saved.get("context")
    if context != context_signature(plan, argv, encoder):
        raise UserError(
            "Tuning selection does not match this hardware, source, software, data, encoder, or workload overrides. Recalibrate or omit --tuning."
        )
    winner = saved.get("winner")
    if not isinstance(winner, dict):
        raise UserError("Tuning selection has no valid winning profile.")
    profile = winner.get("profile")
    if profile not in PROFILE_NAMES:
        raise UserError("Tuning selection has no valid winning profile.")
    selected = choose_plan(
        plan.hardware,
        plan.bank,
        policy=plan.policy,
        interface=plan.interface,
        profile=profile,
    )
    if selected.errors or selected.overrides != winner.get("overrides"):
        raise UserError(
            "Available memory no longer supports the measured configuration; use auto sizing or recalibrate."
        )
    fps = winner.get("frames_per_second", 0)
    if not isinstance(fps, (float, int)) or not math.isfinite(fps) or fps <= 0:
        raise UserError("Tuning selection contains an invalid throughput measurement.")
    selected.calibration = {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "frames_per_second": fps,
    }
    selected.reasons.append(
        f"Using calibrated {profile}: {fps:.0f} measured frames/s (throughput only)."
    )
    return selected


def validate_checkpoint(path: Path) -> dict:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    tensor_count = 0

    def visit(value):
        nonlocal tensor_count
        if isinstance(value, torch.Tensor):
            tensor_count += 1
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError("Non-finite checkpoint tensor")
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(payload)

    def steps(value):
        result = []
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "step" and isinstance(item, (int, float, torch.Tensor)):
                    result.append(float(item))
                else:
                    result.extend(steps(item))
        return result

    optimizers = payload.get("optimizers") or {
        "ppo": payload.get("optimizer_state_dict", {})
    }
    counts = {name: max(steps(value), default=0) for name, value in optimizers.items()}
    if not tensor_count or not all(value > 0 for value in counts.values()):
        raise ValueError(
            f"Missing optimizer updates in calibration checkpoint: {counts}"
        )
    return {
        "sha256": file_sha256(path),
        "optimizer_steps": counts,
        "finite_tensors": tensor_count,
    }


class _Stopped(BaseException):
    """The runner itself received a termination signal."""

    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum


def _raise_stopped(signum, _frame) -> None:
    raise _Stopped(signum)


def run_trial(command: list[str], env: dict, log: Path, timeout: float) -> int:
    # The trial runs in its own session so a timeout can stop its whole process
    # tree. That also detaches it from signals sent to this runner, so forward
    # termination here: otherwise stopping the runner leaves training on the GPU.
    previous = {}
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGTERM, signal.SIGHUP):
            previous[signum] = signal.signal(signum, _raise_stopped)
    try:
        with log.open("w") as stream:
            stream.write(json.dumps(command) + "\n")
            stream.flush()
            process = subprocess.Popen(
                command,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                return process.wait(timeout=timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt, _Stopped) as stop:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                if isinstance(stop, KeyboardInterrupt):
                    raise
                if isinstance(stop, _Stopped):
                    raise SystemExit(128 + stop.signum) from None
                return 124
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def main(argv: list[str] | None = None, *, calibrate: bool = False) -> int:
    parser = argparse.ArgumentParser(
        prog="imitationlab tune" if calibrate else "imitationlab hardware",
        allow_abbrev=False,
        description="Measure local training throughput after warmup."
        if calibrate
        else "Inspect local hardware and recommended execution sizes without starting Isaac Lab.",
    )
    parser.add_argument(
        "--policy",
        choices=("ppo", "supermimic", "svg", "diffsr"),
        default="supermimic",
        help="Policy learner (default: %(default)s).",
    )
    parser.add_argument(
        "--interface",
        choices=INTERFACES,
        default="explicit",
        help="Command interface (default: %(default)s).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=calibrate,
        help="Motion manifest or packed reference bank.",
    )
    parser.add_argument(
        "--encoder",
        type=Path,
        help="Pretrained spectral or FSQ encoder checkpoint, for a latent interface.",
    )
    parser.add_argument(
        "--stage",
        choices=("train", "pretrain"),
        default="train",
        help="Workload to size (default: %(default)s). Calibration supports train only.",
    )
    parser.add_argument(
        "--hardware",
        choices=("auto", "off", *PROFILE_NAMES),
        default="auto",
        help="Execution-size profile (default: auto, chosen from local resources).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the recommendation as JSON (imitationlab hardware).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="New directory for trial logs and selection.json (default: outputs/tuning/<UTC time>)."
        if calibrate
        else "Also write the recommendation JSON to this file.",
    )
    if calibrate:
        parser.add_argument(
            "--seconds",
            type=float,
            default=30,
            help="Measured seconds per candidate after warmup (default: %(default)s).",
        )
        parser.add_argument(
            "--warmup-iterations",
            type=int,
            default=5,
            help="Training iterations discarded before measuring (default: %(default)s).",
        )
        parser.add_argument(
            "--timeout",
            type=float,
            default=900,
            help="Wall-clock limit per candidate, in seconds (default: %(default)s).",
        )
        parser.add_argument(
            "--candidates",
            nargs="+",
            choices=PROFILE_NAMES,
            help="Profiles to measure (default: every profile that fits this machine).",
        )
    args, rest = parser.parse_known_args(argv)
    # Workload options are the trainer's or the pretrainer's, declared in snake
    # case; accept kebab case too.
    rest = canonical_argv(rest, snake_case_options(*TRAIN_MODULES, *PRETRAIN_MODULES))
    with reported_by(parser):
        return _run(parser, args, rest, calibrate=calibrate)


def _run(
    parser: argparse.ArgumentParser, args, rest: list[str], *, calibrate: bool
) -> int:
    workload = [*data_overrides(args.data), *rest]
    expanded = expand_args_files(workload, environ={})
    hardware = detect_hardware(selected_device(expanded))
    bank = bank_from_argv(expanded)
    plan = choose_plan(
        hardware,
        bank,
        policy=args.policy,
        interface=args.interface,
        stage=args.stage,
        profile=args.hardware,
    )
    if not calibrate:
        output = json.dumps(plan.to_dict(), indent=2)
        print(
            output
            if args.json
            else explain(plan) + "\nOverrides: " + " ".join(plan.overrides)
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output + "\n")
        return 1 if plan.errors else 0
    if args.stage != "train" or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("Calibration currently supports single-GPU policy training.")
    if (
        args.seconds <= 0
        or not math.isfinite(args.seconds)
        or args.warmup_iterations < 1
        or args.timeout <= 0
        or not math.isfinite(args.timeout)
    ):
        parser.error("Seconds, timeout, and warmup iterations must be positive.")
    if args.hardware == "off" or (args.hardware != "auto" and args.candidates):
        parser.error(
            "Use --candidates to select calibration profiles, or a single --hardware profile."
        )
    if args.interface != "explicit" and args.encoder is None:
        parser.error("Latent calibration requires --encoder.")
    if args.interface == "explicit" and args.encoder is not None:
        parser.error("--encoder requires a latent interface.")
    if bank.kind == "unknown":
        parser.error("Calibration requires an existing motion manifest or packed bank.")
    if any(
        x.startswith(
            (
                "--checkpoint",
                "--task",
                "--hardware",
                "--tuning",
                "--max_iterations",
                "--max_train_seconds",
            )
        )
        for x in expanded
    ):
        parser.error(
            "Calibration owns the task, stopping budget, and fresh initialization; remove conflicting options."
        )
    if plan.errors:
        parser.error(" ".join(plan.errors))
    output = args.output or Path("outputs/tuning") / datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "selection.json"
    if result_path.exists():
        parser.error("Use a fresh output directory to preserve the existing selection.")
    report = {
        "task": DEFAULT_TASK,
        "seed": 42,
        "context": context_signature(plan, workload, args.encoder),
        "trials": [],
        "winner": None,
    }
    eligible = []
    for name in args.candidates or (
        [args.hardware] if args.hardware != "auto" else PROFILE_NAMES
    ):
        candidate = choose_plan(
            hardware, bank, policy=args.policy, interface=args.interface, profile=name
        )
        if not candidate.errors:
            eligible.append(candidate)
    if not eligible:
        parser.error("No candidate fits available resources.")
    for candidate in eligible:
        name = candidate.profile
        trial = output / name
        trial.mkdir(exist_ok=False)
        measurements = trial / "measurement.json"
        env = dict(
            os.environ,
            TERM=os.environ.get("TERM", "xterm"),
            OMNI_KIT_ACCEPT_EULA="YES",
            PYTHONUNBUFFERED="1",
        )
        env["IMITATIONLAB_CALIBRATION"] = json.dumps(
            {
                "output": str(measurements.resolve()),
                "seconds": args.seconds,
                "warmup": args.warmup_iterations,
            }
        )
        command = [
            sys.executable,
            "-m",
            "imitationlab",
            "train",
            "--policy",
            args.policy,
            "--interface",
            args.interface,
            "--hardware",
            name,
            "--data",
            str(args.data),
            *rest,
            "--seed",
            "42",
            "agent.collector.total_frames=1000000000000",
            "agent.logger.backend=",
            "agent.save_interval=1000000000000",
            f"agent.logger.log_dir={trial.resolve() / 'training'}",
        ]
        if args.encoder:
            command += ["--encoder", str(args.encoder)]
        if args.policy != "ppo":
            command += ["agent.wm.wm_warmup_iters=0"]
        print(
            f"Calibrating {name}: {args.warmup_iterations} warmup iterations, then {args.seconds:g} measured seconds. Log: {trial / 'train.log'}",
            flush=True,
        )
        result = {"profile": name, "overrides": candidate.overrides, "command": command}
        result["exit_code"] = run_trial(command, env, trial / "train.log", args.timeout)
        try:
            if result["exit_code"]:
                raise ValueError(f"Trial exited with code {result['exit_code']}")
            timing = json.loads(measurements.read_text())
            if (
                not math.isfinite(timing["frames_per_second"])
                or timing["frames_per_second"] <= 0
            ):
                raise ValueError("Invalid timing sample")
            result.update(timing)
            result["checkpoint"] = validate_checkpoint(
                resolve_checkpoint(trial / "training")
            )
            result["status"] = "passed"
        except (ValueError, OSError, KeyError) as exc:
            result.update(status="failed", error=str(exc))
        report["trials"].append(result)
        passed = [item for item in report["trials"] if item["status"] == "passed"]
        report["winner"] = (
            max(passed, key=lambda x: x["frames_per_second"]) if passed else None
        )
        result_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(
            f"{name}: {result['status']}"
            + (
                f" ({result['frames_per_second']:.0f} frames/s)"
                if result["status"] == "passed"
                else f" — {result['error']}"
            ),
            flush=True,
        )
    if report["winner"]:
        print(
            f"Best measured candidate: {report['winner']['profile']}. Reuse with --tuning {result_path}"
        )
        return 0
    return 1
