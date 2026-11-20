"""Measure tracking quality against training time using the public workflows."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import random
import sys
import time

from imitationlab.runners.artifacts import file_sha256
from imitationlab.runners.errors import UserError
from imitationlab.runners.registry import (
    DEFAULT_TASK,
    INTERFACES,
    canonical_interface,
    interface_arg,
)
from imitationlab.runners.tuning import run_trial, source_fingerprint

POLICIES = ("ppo", "supermimic", "svg", "diffsr")
DATASETS = ("lafan1", "bones_seed")


def runtime_details() -> dict:
    versions = {}
    for name in (
        "torch",
        "torchrl",
        "tensordict",
        "isaaclab",
        "newton",
        "mujoco",
        "mujoco-warp",
    ):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "packages": versions,
        "python": sys.version,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": os.uname().nodename,
    }


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def motion_board(
    bank: Path, *, count: int = 256, seed: int = 42, steps: int = 500
) -> dict:
    manifest_path = bank / "reference_arrays_manifest.json"
    if not manifest_path.is_file():
        raise UserError(f"Packed motion bank is missing {manifest_path.name}: {bank}")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    total = len(manifest["traj_info"]["start_index"])
    if total < 1 or count < 1 or steps < 1:
        raise UserError(
            "The bank, evaluation clip count, and horizon must be positive."
        )
    ranks = sorted(random.Random(seed).sample(range(total), min(count, total)))
    board = {
        "bank_id": manifest["key"]["source"]["persist_id"],
        "bank_sha256": hashlib.sha256(raw).hexdigest(),
        "bank_clips": total,
        "bank_frames": manifest["traj_info"]["written"],
        "ranks": ranks,
        "seed": seed,
        "steps": steps,
        "control_dt": 0.02,
        "start_frame": 0,
        "action_sampling": "mode",
        "randomization": "none",
        "success": f"No task tracking failure during min({steps * 0.02:g} seconds, clip length).",
        "failure_rule": "abs(root height error) > 0.25 m or abs(root R[2,2] error) > 0.8",
        "population": "Fixed in-distribution training-motion board; not held-out generalization.",
    }
    board["sha256"] = hashlib.sha256(
        json.dumps(board, sort_keys=True).encode()
    ).hexdigest()
    return board


def tracking_point(evaluation: dict, checkpoint: dict, board: dict) -> dict:
    metadata = evaluation["metadata"]
    for key, expected in {
        "task": DEFAULT_TASK,
        "steps_requested": board["steps"],
        "seed": board["seed"],
        "reference_start_frame": 0,
        "action_sampling": "mode",
        "randomization_profile": "none",
        "early_terminations_enabled": True,
        "keep_after_done": False,
    }.items():
        if metadata[key] != expected:
            raise ValueError(f"Evaluation {key} differs from the fixed protocol.")
    if metadata["reference_arrays"]["persist_id"] != board["bank_id"]:
        raise ValueError("Evaluation bank changed.")
    aggregate = evaluation["aggregate"]
    if (
        aggregate["num_evaluated_envs"] != len(board["ranks"])
        or aggregate["valid_transition_count"] < 1
    ):
        raise ValueError("Evaluation population does not match the fixed board.")
    rows = evaluation.get("per_env", evaluation.get("per_environment", []))
    if (
        len(rows) != len(board["ranks"])
        or sorted(row["trajectory_rank"] for row in rows) != board["ranks"]
    ):
        raise ValueError("Evaluation clip ranks changed.")
    values = {
        "success": aggregate["tracking_success_rate"],
        "mpjpe_l_mm": evaluation["metrics"]["tracking_mpjpe_mm"]["mean"],
        "mpjpe_g_mm": evaluation["metrics"]["tracking_mpjpe_g_mm"]["mean"],
        "survival_seconds": aggregate["survival_steps_mean"] * board["control_dt"],
    }
    if (
        not all(math.isfinite(x) for x in values.values())
        or not 0 <= values["success"] <= 1
    ):
        raise ValueError("Non-finite or invalid tracking measurements.")
    return {
        **checkpoint,
        **values,
        "steps_executed": aggregate["steps_executed"],
        "valid_transitions": aggregate["valid_transition_count"],
        "board_sha256": board["sha256"],
    }


def first_reached(
    points: list[dict], success: float, *, local_error: float | None = None
) -> dict | None:
    previous = None
    for point in sorted(points, key=lambda x: x["train_seconds"]):
        # Evaluation rates can originate in float32 (38/40 becomes 0.949999988).
        # Accept representation error, without interpolating between samples.
        reached = point["success"] >= success or math.isclose(
            point["success"], success, rel_tol=0, abs_tol=1e-7
        )
        if reached and (local_error is None or point["mpjpe_l_mm"] <= local_error):
            return {
                "seconds": point["train_seconds"],
                "previous_sample_seconds": previous,
                "end_to_end_seconds": point["end_to_end_seconds"],
            }
        previous = point["train_seconds"]
    return None


def child(command: list[str], log: Path, timeout: float) -> dict:
    env = dict(
        os.environ,
        TERM=os.environ.get("TERM") or "xterm",
        OMNI_KIT_ACCEPT_EULA="YES",
        PYTHONUNBUFFERED="1",
    )
    for key in (
        "IMITATIONLAB_CALIBRATION",
        "IMITATIONLAB_RUN_SPEC",
        "IMITATIONLAB_PRETRAIN_HARDWARE",
    ):
        env.pop(key, None)
    begin = time.monotonic()
    code = run_trial(command, env, log, timeout)
    return {
        "command": command,
        "exit_code": code,
        "wall_seconds": time.monotonic() - begin,
        "monotonic_start": begin,
    }


def finite_checkpoint(path: Path) -> dict:
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)
    tensors = 0

    def visit(value):
        nonlocal tensors
        if isinstance(value, torch.Tensor):
            tensors += 1
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(f"Non-finite checkpoint tensor: {path}")
        elif isinstance(value, dict):
            for v in value.values():
                visit(v)
        elif isinstance(value, (list, tuple)):
            for v in value:
                visit(v)

    visit(state)
    if not tensors:
        raise ValueError(f"Checkpoint has no tensors: {path}")
    return state


def validate_training_checkpoint(path: Path, policy: str) -> dict:
    """Require learner updates, not just a fitted world model or finite weights."""
    from imitationlab.runners.tuning import validate_checkpoint

    validation = validate_checkpoint(path)
    required = {"ppo"} if policy == "ppo" else {"wm", "actor", "critic"}
    missing = required.difference(validation["optimizer_steps"])
    if missing:
        raise ValueError(f"Missing optimizer update evidence: {sorted(missing)}")
    return validation


def _report_failure(result: dict, result_path: Path) -> int:
    """Print a failed case's error to stderr and return the exit status."""
    if result["status"] == "completed":
        return 0
    print(
        f"imitationlab benchmark: error: {result.get('error', 'unknown failure')} "
        f"(recorded in {result_path})",
        file=sys.stderr,
    )
    return 1


def pretrain(args, rest: list[str]) -> int:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "pretrain.json"
    if result_path.exists():
        raise UserError(
            "Use a new output directory; existing encoder results are immutable."
        )
    result = {
        "status": "running",
        "dataset": args.dataset,
        "interface": args.interface,
        "seed": args.seed,
        "source_sha256": source_fingerprint(),
        "runtime": runtime_details(),
        "board": motion_board(args.data),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(result_path, result)
    command = [
        sys.executable,
        "-m",
        "imitationlab",
        "pretrain",
        "--assert-kitless",
        "--interface",
        args.interface,
        "--data",
        str(args.data),
        "--output-dir",
        str(output),
        "--num_updates",
        str(args.updates),
        "--seed",
        str(args.seed),
        "--wandb_mode",
        "disabled",
        "--log_interval",
        "100",
        *rest,
    ]
    try:
        result["process"] = child(command, output / "pretrain.log", args.timeout)
        if result["process"]["exit_code"]:
            raise ValueError(
                f"Encoder exited {result['process']['exit_code']}; see pretrain.log"
            )
        path = output / "checkpoints/encoder.pt"
        state = finite_checkpoint(path)
        if state["update"] != args.updates:
            raise ValueError("Encoder did not complete the requested updates.")
        result.update(
            status="completed",
            encoder=str(path),
            encoder_sha256=file_sha256(path),
            updates=state["update"],
            config=state["config"],
        )
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        result.update(status="failed", error=str(exc))
    write_json(result_path, result)
    print(
        json.dumps({k: result[k] for k in ("dataset", "interface", "status")}, indent=2)
    )
    return _report_failure(result, result_path)


def frozen_encoder_record(
    directory: Path, *, interface: str, board: dict, seed: int, source_sha256: str
) -> dict:
    """Validate an explicitly identified frozen encoder before policy training."""
    record_path = directory / "pretrain.json"
    if not record_path.is_file():
        raise UserError(f"Encoder run has no pretrain.json: {directory}")
    record = json.loads(record_path.read_text())
    if (
        record["status"] != "completed"
        or canonical_interface(record["interface"]) != canonical_interface(interface)
        or record["board"]["bank_sha256"] != board["bank_sha256"]
        or record["seed"] != seed
        or record["source_sha256"] != source_sha256
    ):
        raise UserError(
            "Encoder interface, training bank, seed, or source does not match. "
            "To reuse frozen weights from another run, explicitly pin their "
            "--encoder-seed and --encoder-source-sha256."
        )
    if file_sha256(Path(record["encoder"])) != record["encoder_sha256"]:
        raise UserError("Encoder weights changed after pretraining.")
    return record


def run(args, rest: list[str]) -> int:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "result.json"
    if result_path.exists():
        raise UserError(
            "Use a new output directory; existing benchmark results are immutable."
        )
    schedule = [float(x) for x in args.milestones]
    if (
        not math.isfinite(args.seconds)
        or not schedule
        or schedule[0] <= 0
        or schedule[-1] > args.seconds
        or not all(math.isfinite(x) for x in schedule)
        or any(a >= b for a, b in zip(schedule, schedule[1:]))
    ):
        raise UserError(
            "Milestones must be positive and no later than the training budget."
        )
    eval_seed = args.seed if args.eval_seed is None else args.eval_seed
    board = motion_board(
        args.data, count=args.eval_clips, steps=args.eval_steps, seed=eval_seed
    )
    pretrain_seconds = 0.0
    encoder = None
    if args.interface != "explicit":
        if not args.encoder_run:
            raise UserError(
                "A latent interface needs --encoder-run with a completed pretrain.json."
            )
        record = frozen_encoder_record(
            args.encoder_run,
            interface=args.interface,
            board=board,
            seed=args.seed if args.encoder_seed is None else args.encoder_seed,
            source_sha256=args.encoder_source_sha256 or source_fingerprint(),
        )
        encoder = Path(record["encoder"])
        pretrain_seconds = record["process"]["wall_seconds"]
    elif (
        args.encoder_run or args.encoder_seed is not None or args.encoder_source_sha256
    ):
        raise UserError("Explicit commands do not use an encoder.")
    result = {
        "status": "running",
        "task": DEFAULT_TASK,
        "dataset": args.dataset,
        "policy": args.policy,
        "interface": args.interface,
        "seed": args.seed,
        "source_sha256": source_fingerprint(),
        "runtime": runtime_details(),
        "board": board,
        "budget_seconds": args.seconds,
        "milestones": schedule,
        "pretrain_seconds": pretrain_seconds,
        "encoder_pretraining": {
            k: record[k] for k in ("seed", "source_sha256", "encoder_sha256", "updates")
        }
        if encoder
        else None,
        "points": [],
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": os.uname().nodename,
    }
    write_json(result_path, result)
    command = [
        sys.executable,
        "-m",
        "imitationlab",
        "train",
        "--policy",
        args.policy,
        "--interface",
        args.interface,
        "--data",
        str(args.data),
        "--seed",
        str(args.seed),
        "--assert-kitless",
        *rest,
        "--max-train-seconds",
        str(args.seconds),
        "agent.collector.total_frames=1000000000000",
        "agent.save_interval=1000000000000",
        f"agent.trainer.checkpoint_seconds={json.dumps(schedule)}",
        "agent.trainer.gpu_telemetry=true",
        "agent.logger.backend=",
        f"agent.logger.log_dir={output / 'training'}",
    ]
    if encoder:
        command += ["--encoder", str(encoder)]
    try:
        result["training"] = child(command, output / "train.log", args.timeout)
        if result["training"]["exit_code"]:
            raise ValueError(
                f"Training exited {result['training']['exit_code']}; see train.log"
            )
        runs = list((output / "training").glob("*/run.json"))
        if len(runs) != 1:
            raise ValueError("Expected one training run.")
        run_dir = runs[0].parent
        result["run"] = json.loads(runs[0].read_text())
        for key, expected in (
            ("policy", args.policy),
            ("interface", args.interface),
            ("task", DEFAULT_TASK),
            ("seed", args.seed),
        ):
            if result["run"][key] != expected:
                raise ValueError(f"Resolved {key} differs from the requested run.")
        if result["run"]["data"]["persist_id"] != board["bank_id"]:
            raise ValueError("Resolved training bank identity differs from the board.")
        if encoder and result["run"]["encoder"]["sha256"] != record["encoder_sha256"]:
            raise ValueError("Training loaded different encoder weights.")
        clock = json.loads((run_dir / "training_clock.json").read_text())
        startup = clock["monotonic_start"] - result["training"]["monotonic_start"]
        result["startup_seconds"] = startup
        checkpoints = {}
        for pattern in ("policy_step_*.pt", "model_step_*.pt"):
            for path in (run_dir / "models").glob(pattern):
                checkpoints[int(path.stem.rsplit("_", 1)[1])] = path
        if not checkpoints:
            raise ValueError("No milestone checkpoints were saved.")
        # A finite world model does not prove that policy learning happened.
        # The final full checkpoint must contain actual optimizer updates,
        # including the actor (a warmup-only run is not a learning curve).
        result["checkpoint_validation"] = validate_training_checkpoint(
            checkpoints[max(checkpoints)], args.policy
        )
        for frame, path in sorted(checkpoints.items()):
            state = finite_checkpoint(path)
            seconds = float(state["cumulative_train_seconds"])
            point = {
                "checkpoint": str(path),
                "checkpoint_sha256": file_sha256(path),
                "frames": frame,
                "train_seconds": seconds,
                "startup_seconds": startup,
                "pretrain_seconds": pretrain_seconds,
                "end_to_end_seconds": pretrain_seconds + startup + seconds,
            }
            del state
            destination = output / "evaluation" / f"{path.stem}.json"
            destination.parent.mkdir(exist_ok=True)
            evaluation = [
                sys.executable,
                "-m",
                "imitationlab",
                "evaluate",
                "--assert-kitless",
                "--checkpoint",
                str(path),
                "--data",
                str(args.data),
                "--num-envs",
                str(len(board["ranks"])),
                "--steps",
                str(args.eval_steps),
                "--seed",
                str(board["seed"]),
                "--output",
                str(destination),
                "--trajectory_ranks",
                *map(str, board["ranks"]),
            ]
            process = child(
                evaluation, destination.with_suffix(".log"), args.eval_timeout
            )
            if process["exit_code"]:
                raise ValueError(
                    f"Evaluation exited {process['exit_code']}; see {destination.with_suffix('.log')}"
                )
            point = tracking_point(json.loads(destination.read_text()), point, board)
            point["evaluation_seconds"] = process["wall_seconds"]
            result["points"].append(point)
            write_json(result_path, result)
        if result["points"][-1]["train_seconds"] < args.seconds:
            raise ValueError("The requested training-time budget was not reached.")
        result["thresholds"] = {
            str(level): first_reached(result["points"], level)
            for level in (0.5, 0.9, 0.95)
        }
        result["thresholds"]["0.9_and_40mm"] = first_reached(
            result["points"], 0.9, local_error=40
        )
        result["status"] = "completed"
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        result.update(status="failed", error=str(exc))
    write_json(result_path, result)
    print(
        json.dumps(
            {k: result[k] for k in ("dataset", "policy", "interface", "status")},
            indent=2,
        )
    )
    return _report_failure(result, result_path)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "report":
        from imitationlab.evaluation.learning_curves import main as report

        return report(argv[1:], prog="imitationlab benchmark report")
    parser = argparse.ArgumentParser(
        prog="imitationlab benchmark", description=__doc__, allow_abbrev=False
    )
    actions = parser.add_subparsers(dest="action", required=True)
    actions.add_parser(
        "report", help="Render aggregate tables and curves from recorded cases."
    )
    for name in ("run", "pretrain"):
        p = actions.add_parser(
            name,
            allow_abbrev=False,
            help="Train a policy and evaluate its milestone checkpoints."
            if name == "run"
            else "Pretrain a command encoder for later runs.",
        )
        p.add_argument(
            "--dataset",
            choices=DATASETS,
            required=True,
            help="Dataset label recorded with the result.",
        )
        p.add_argument(
            "--data", type=Path, required=True, help="Packed reference bank directory."
        )
        p.add_argument(
            "--interface",
            type=interface_arg,
            choices=INTERFACES if name == "run" else INTERFACES[1:],
            default="explicit" if name == "run" else None,
            required=name == "pretrain",
            help="Command interface (default: explicit)."
            if name == "run"
            else "Encoder to pretrain.",
        )
        p.add_argument(
            "--output",
            type=Path,
            required=True,
            help="New directory for the logs and the result record.",
        )
        p.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Policy seed (default: %(default)s)."
            if name == "run"
            else "Encoder seed (default: %(default)s).",
        )
        p.add_argument(
            "--timeout",
            type=float,
            default=7200,
            help="Wall-clock limit for the training process, in seconds (default: %(default)s)."
            if name == "run"
            else "Wall-clock limit for pretraining, in seconds (default: %(default)s).",
        )
    p = actions.choices["pretrain"]
    p.add_argument(
        "--updates",
        type=int,
        default=2000,
        help="Encoder gradient updates (default: %(default)s).",
    )
    p = actions.choices["run"]
    p.add_argument(
        "--policy", choices=POLICIES, required=True, help="Policy learner to train."
    )
    p.add_argument(
        "--encoder-run",
        type=Path,
        help="Output directory of a completed benchmark pretrain; required for a latent interface.",
    )
    p.add_argument(
        "--encoder-seed",
        type=int,
        help="Expected frozen encoder seed; defaults to the policy seed.",
    )
    p.add_argument(
        "--encoder-source-sha256",
        help="Expected frozen encoder source fingerprint; defaults to current source.",
    )
    p.add_argument(
        "--seconds",
        type=float,
        default=1800,
        help="Policy-training time budget, in seconds (default: %(default)s).",
    )
    p.add_argument(
        "--milestones",
        type=float,
        nargs="+",
        default=[60, 120, 300, 600, 900, 1200, 1800],
        help="Increasing training times, in seconds, at which a checkpoint is saved and evaluated.",
    )
    p.add_argument(
        "--eval-clips",
        type=int,
        default=256,
        help="Clips sampled from the bank for evaluation (default: %(default)s).",
    )
    p.add_argument(
        "--eval-seed",
        type=int,
        help="Fixed board/evaluation seed, independent of the policy seed; defaults to --seed.",
    )
    p.add_argument(
        "--eval-steps",
        type=int,
        default=500,
        help="Control steps per evaluation rollout (default: %(default)s).",
    )
    p.add_argument(
        "--eval-timeout",
        type=float,
        default=900,
        help="Wall-clock limit per checkpoint evaluation, in seconds (default: %(default)s).",
    )
    args, rest = parser.parse_known_args(argv)
    action = pretrain if args.action == "pretrain" else run
    try:
        return action(args, rest)
    except UserError as exc:
        actions.choices[args.action].error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
