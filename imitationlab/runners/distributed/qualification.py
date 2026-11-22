"""Bounded two-GPU startup, checkpoint and resume qualification in one job."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from imitationlab.paths import REPO_ROOT


def read_health(root: Path, workers: int):
    rows = [
        json.loads(path.read_text()) for path in root.rglob("distributed_health.json")
    ]
    if len(rows) != workers or {r["rank"] for r in rows} != set(range(workers)):
        raise RuntimeError(
            f"Expected exactly {workers} worker health records under {root}"
        )
    if any(r["world_size"] != workers or r["max_model_difference"] != 0 for r in rows):
        raise RuntimeError("Worker state synchronization failed")
    if len({r["cumulative_env_frames"] for r in rows}) != 1:
        raise RuntimeError("Worker frame counts differ")
    if len({tuple(r["learning_rates"]) for r in rows}) != 1:
        raise RuntimeError("Worker learning rates differ")
    if any(r["device"] != f"cuda:{r['rank']}" for r in rows):
        raise RuntimeError("Workers did not select distinct GPUs")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--qualification-output", type=Path, required=True)
    parser.add_argument("--qualification-seconds", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=2)
    args, training_args = parser.parse_known_args(argv)
    root = args.qualification_output
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"Qualification output must be fresh: {root}")
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    launcher = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={args.workers}",
        "--max-restarts=0",
        "--module",
        "imitationlab.runners.train",
    ]

    def run_stage(name, extra):
        command = (
            launcher
            + training_args
            + [
                "--distributed",
                "--assert-kitless",
                f"agent.logger.log_dir={root / name}",
            ]
            + extra
        )
        (root / f"{name}_command.json").write_text(json.dumps(command, indent=2) + "\n")
        print(f"[QUALIFICATION] Starting {name}", flush=True)
        # Stream all workers into Slurm's log, preserving failure exit codes.
        subprocess.run(command, cwd=REPO_ROOT, check=True, env=os.environ.copy())
        return read_health(root / name, args.workers)

    # A separate process tests both allocated devices and releases its CUDA
    # contexts before torchrun starts. This distinguishes runtime/device access
    # failures from simulator or distributed-policy failures.
    probe = """import os, torch
print('CUDA_VISIBLE_DEVICES=', os.environ.get('CUDA_VISIBLE_DEVICES'), flush=True)
print('CUDA_DEVICE_ORDER=', os.environ.get('CUDA_DEVICE_ORDER'), flush=True)
print('device_count=', torch.cuda.device_count(), flush=True)
for index in range(int(os.environ['QUALIFICATION_WORKERS'])):
    device = f'cuda:{index}'
    value = torch.ones(4, device=device).sum().item()
    print(device, torch.cuda.get_device_name(index), value, flush=True)
"""
    probe_env = {**os.environ, "QUALIFICATION_WORKERS": str(args.workers)}
    try:
        with (root / "cuda_probe.log").open("w") as log:
            subprocess.run(
                [sys.executable, "-c", probe],
                check=True,
                env=probe_env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
    finally:
        print((root / "cuda_probe.log").read_text(), flush=True)
    smoke = run_stage("smoke", ["--max-iterations", "8"])
    frames = smoke[0]["cumulative_env_frames"]
    remaining = args.qualification_seconds - (time.monotonic() - started) - 300
    if remaining < 60:
        raise RuntimeError(
            "Startup exhausted qualification time; no time to test resume"
        )
    resume = run_stage(
        "resume",
        ["--checkpoint", str(root / "smoke"), "--max-training-seconds", str(remaining)],
    )
    if any(
        r["initial_frames"] != frames or r["cumulative_env_frames"] <= frames
        for r in resume
    ):
        raise RuntimeError("Resume did not advance the global frame counter")
    report = {
        "status": "passed",
        "workers": args.workers,
        "smoke": smoke,
        "resume": resume,
        "elapsed_seconds": time.monotonic() - started,
    }
    (root / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"[QUALIFICATION] PASSED: {root / 'qualification.json'}", flush=True)
    return 0
