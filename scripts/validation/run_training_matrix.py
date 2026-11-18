#!/usr/bin/env python3
"""Run the local G1 learner/interface smoke matrix on two authorized motion banks.

Run with the Isaac Lab Pixi environment. Artifacts and incremental results stay
under --output; this command never downloads or publishes motion data.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

POLICIES = ("ppo", "supermimic", "svg", "diffsr")
INTERFACES = ("explicit", "fsq", "spectral")
SMALL_WORLD_MODEL = [
    "agent.wm.min_replay_steps=64",
    "agent.wm.wm_batch_size=32",
    "agent.wm.imagine_batch=32",
    "agent.wm.horizon=4",
    "agent.wm.wm_hidden=[64,64,64]",
    "agent.wm.actor_hidden=[64,64,64]",
    "agent.wm.critic_hidden=[64,64]",
    "agent.wm.wm_warmup_iters=0",
]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tensor_check(path: Path) -> dict:
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)
    count = 0

    def visit(value):
        nonlocal count
        if isinstance(value, torch.Tensor):
            count += 1
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(f"Non-finite checkpoint tensor in {path}")
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)

    visit(state)
    if not count:
        raise ValueError(f"No tensors in {path}")

    def optimizer_steps(value):
        steps = []
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "step" and isinstance(item, (int, float, torch.Tensor)):
                    steps.append(float(item))
                else:
                    steps.extend(optimizer_steps(item))
        return steps

    optimizers = state.get("optimizers", {})
    if "optimizer_state_dict" in state:
        optimizers = {"optimizer": state["optimizer_state_dict"]}
    updates = {
        name: max(optimizer_steps(opt), default=0) for name, opt in optimizers.items()
    }
    if updates and not all(value > 0 for value in updates.values()):
        raise ValueError(f"An optimizer did not update in {path}: {updates}")
    return {
        "sha256": sha256(path),
        "finite_tensors": count,
        "iteration": state.get("iteration", state.get("update")),
        "frames": state.get("cumulative_env_frames"),
        "optimizer_steps": updates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lafan1", type=Path, required=True)
    parser.add_argument("--bones-seed", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/compatibility-matrix")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep verified passes and retry incomplete cases.",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists() and not args.resume:
        parser.error("Use a new output directory to preserve previous results.")
    env = dict(
        os.environ, TERM="xterm", OMNI_KIT_ACCEPT_EULA="YES", PYTHONUNBUFFERED="1"
    )
    env.pop("TORCHDYNAMO_DISABLE", None)
    command = [sys.executable, "-m", "imitationlab"]
    report = {
        "schema_version": 1,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "task": "imitation-g1-v1",
        "seed": args.seed,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "train_envs": 16,
            "max_iterations_argument": 2,
            "wm_warmup_iters": 0,
            "pretrain_updates": 2,
            "eval_envs": 4,
            "eval_steps": 100,
            "eval_ranks": [0, 1, 2, 3],
            "compiler_disabled": False,
        },
        "datasets": {},
        "pretraining": [],
        "cases": [],
    }
    if args.resume:
        previous = json.loads(result_path.read_text())
        if (
            previous["source_commit"] != report["source_commit"]
            or previous["seed"] != args.seed
        ):
            parser.error("Resume requires the same source commit and seed.")
        report.update(
            {key: previous[key] for key in ("datasets", "pretraining", "cases")}
        )
        report["prior_attempts"] = previous.get("prior_attempts", [])

    def save():
        temporary = result_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(result_path)

    def run(label, argv):
        print(f"START {label}", flush=True)
        start = time.monotonic()
        log_path = output / f"{label}.log"
        with log_path.open("w") as log:
            log.write(json.dumps(argv) + "\n")
            log.flush()
            try:
                completed = subprocess.run(
                    argv,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                )
                code = completed.returncode
            except subprocess.TimeoutExpired:
                code = 124
        result = {
            "exit_code": code,
            "seconds": round(time.monotonic() - start, 2),
            "command": argv,
            "log": str(log_path),
        }
        print(f"DONE {label}: exit={code} seconds={result['seconds']}", flush=True)
        if code:
            print(
                "\n".join(log_path.read_text(errors="replace").splitlines()[-18:]),
                flush=True,
            )
        return result

    for dataset, bank in (("bones_seed", args.bones_seed), ("lafan1", args.lafan1)):
        bank = bank.resolve()
        manifest = bank / "reference_arrays_manifest.json"
        info = json.loads(manifest.read_text())
        if dataset in report["datasets"] and report["datasets"][dataset][
            "manifest_sha256"
        ] != sha256(manifest):
            raise ValueError("Resume requires the original motion bank identity")
        report["datasets"][dataset] = {
            "path": str(bank),
            "persist_id": info["key"]["source"]["persist_id"],
            "manifest_sha256": sha256(manifest),
            "clips": len(info["traj_info"]["start_index"]),
            "transitions": info["traj_info"]["written"],
        }
        save()
        encoders = {}
        for interface in INTERFACES[1:]:
            label = f"{dataset}-pretrain-{interface}"
            destination = output / "encoders" / dataset / interface
            existing = next(
                (
                    item
                    for item in report["pretraining"]
                    if item["dataset"] == dataset and item["interface"] == interface
                ),
                None,
            )
            if existing and existing.get("status") == "passed":
                encoder = destination / "checkpoints" / "encoder.pt"
                if tensor_check(encoder)["sha256"] != existing["checkpoint"]["sha256"]:
                    raise ValueError("Encoder changed since previous attempt")
                encoders[interface] = encoder
                continue
            if existing:
                report.setdefault("prior_attempts", []).append(existing)
                report["pretraining"].remove(existing)
            argv = command + [
                "pretrain",
                "--hardware",
                "off",
                "--interface",
                interface,
                "--data",
                str(bank),
                "--output-dir",
                str(destination),
                "--num-envs",
                "4",
                "--batch_size",
                "16",
                "--num_updates",
                "2",
                "--eval_batches",
                "1",
                "--log_interval",
                "1",
                "--seed",
                str(args.seed),
                "--wandb_mode",
                "disabled",
                "--assert-kitless",
                "--encoder_hidden_dims",
                "64",
                "64",
                "--diffsr_feature_dim",
                "64",
                "--diffsr_embed_dim",
                "64",
                "--diffsr_f_hidden_dims",
                "64",
                "64",
                "--diffsr_g_hidden_dims",
                "64",
                "64",
                "--diffsr_mu_hidden_dims",
                "64",
                "64",
            ]
            result = {"dataset": dataset, "interface": interface, **run(label, argv)}
            try:
                if result["exit_code"]:
                    raise ValueError("Encoder pretraining failed")
                encoder = destination / "checkpoints" / "encoder.pt"
                result["checkpoint"] = tensor_check(encoder)
                result["training_checkpoint"] = tensor_check(
                    destination / "checkpoints" / "latest.pt"
                )
                if result["checkpoint"]["iteration"] != 2:
                    raise ValueError("Encoder did not reach two updates")
                encoders[interface] = encoder
                result["status"] = "passed"
            except (ValueError, OSError) as exc:
                result.update(status="failed", error=str(exc))
            report["pretraining"].append(result)
            save()

        for interface in INTERFACES:
            for policy in POLICIES:
                label = f"{dataset}-{interface}-{policy}"
                existing = next(
                    (
                        item
                        for item in report["cases"]
                        if (item["dataset"], item["interface"], item["policy"])
                        == (dataset, interface, policy)
                    ),
                    None,
                )
                if existing and existing.get("status") == "passed":
                    try:
                        checked = tensor_check(Path(existing["checkpoint"]["path"]))
                        if checked["sha256"] != existing["checkpoint"]["sha256"]:
                            raise ValueError(
                                "Checkpoint changed since previous attempt"
                            )
                        existing["checkpoint"].update(checked)
                        continue
                    except (ValueError, OSError) as exc:
                        existing.update(status="failed", error=str(exc))
                if existing:
                    report.setdefault("prior_attempts", []).append(existing)
                    report["cases"].remove(existing)
                result = {
                    "dataset": dataset,
                    "interface": interface,
                    "policy": policy,
                    "status": "running",
                }
                report["cases"].append(result)
                save()
                if interface != "explicit" and interface not in encoders:
                    result.update(status="blocked", error="Encoder pretraining failed")
                    save()
                    continue
                destination = output / "training" / label
                if destination.exists():
                    destination = (
                        output / "training" / f"{label}-retry-{time.time_ns()}"
                    )
                argv = command + [
                    "train",
                    "--hardware",
                    "off",
                    "--task",
                    "imitation-g1-v1",
                    "--policy",
                    policy,
                    "--interface",
                    interface,
                    "--data",
                    str(bank),
                    "--num-envs",
                    "16",
                    "--max-iterations",
                    "2",
                    "--seed",
                    str(args.seed),
                    "--assert-kitless",
                    "agent.logger.backend=",
                    "agent.save_interval=1",
                    f"agent.logger.log_dir={destination}",
                ]
                if interface != "explicit":
                    argv += ["--encoder", str(encoders[interface])]
                if policy != "ppo":
                    argv += SMALL_WORLD_MODEL
                result["train"] = run(f"{label}-train", argv)
                try:
                    if result["train"]["exit_code"]:
                        raise ValueError("Training process failed")
                    runs = list(destination.glob("*/run.json"))
                    if len(runs) != 1:
                        raise ValueError("Expected exactly one recorded run")
                    run_dir = runs[0].parent
                    spec = json.loads(runs[0].read_text())
                    for field, expected in (
                        ("task", "imitation-g1-v1"),
                        ("policy", policy),
                        ("interface", interface),
                        ("seed", args.seed),
                    ):
                        if spec[field] != expected:
                            raise ValueError(f"Resolved {field} does not match request")
                    if (
                        spec["data"]["persist_id"]
                        != report["datasets"][dataset]["persist_id"]
                    ):
                        raise ValueError(
                            "Resolved data identity does not match request"
                        )
                    checkpoint = max(
                        run_dir.glob("models/model_step_*.pt"),
                        key=lambda p: int(p.stem.rsplit("_", 1)[1]),
                    )
                    result["checkpoint"] = tensor_check(checkpoint)
                    result["checkpoint"]["path"] = str(checkpoint)
                    result["checkpoint"]["frames"] = int(
                        checkpoint.stem.rsplit("_", 1)[1]
                    )
                    result["run"] = spec
                    if spec.get("encoder") and spec["encoder"]["sha256"] != sha256(
                        encoders[interface]
                    ):
                        raise ValueError("Tracker loaded an unexpected encoder")
                    evaluation_path = output / f"{label}-evaluation.json"
                    # A retried case must not read, or be refused by, the
                    # result of the attempt it replaces.
                    evaluation_path.unlink(missing_ok=True)
                    argv = command + [
                        "evaluate",
                        "--checkpoint",
                        str(checkpoint),
                        "--data",
                        str(bank),
                        "--num-envs",
                        "4",
                        "--steps",
                        "100",
                        "--seed",
                        str(args.seed),
                        "--output",
                        str(evaluation_path),
                        "--trajectory_ranks",
                        "0",
                        "1",
                        "2",
                        "3",
                    ]
                    result["evaluate"] = run(f"{label}-evaluate", argv)
                    if result["evaluate"]["exit_code"]:
                        raise ValueError("Checkpoint evaluation process failed")
                    evaluation = json.loads(evaluation_path.read_text())
                    aggregate = evaluation["aggregate"]
                    if (
                        aggregate["steps_executed"] < 1
                        or aggregate["valid_transition_count"] < 1
                    ):
                        raise ValueError("Evaluation produced no valid transitions")
                    for metric in evaluation["metrics"].values():
                        if metric["count"] and not math.isfinite(metric["mean"]):
                            raise ValueError("Non-finite evaluation metric")
                    result["evaluation"] = aggregate
                    result["status"] = "passed"
                except (ValueError, OSError, KeyError) as exc:
                    result.update(status="failed", error=str(exc))
                save()
                print(f"CASE {label}: {result['status']}", flush=True)
    return (
        0
        if all(x["status"] == "passed" for x in report["pretraining"] + report["cases"])
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
