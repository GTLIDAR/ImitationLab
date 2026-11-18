"""Portable run descriptions and recovery of historical launch settings."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
from pathlib import Path

from imitationlab.runners.args_file import expand_args_files
from imitationlab.runners.errors import UserError

RUNTIME_ENVIRONMENT = (
    "ISAACLAB_IMITATION_COLLECT_FASTPATH",
    "IMITATIONLAB_COLLECT_FASTPATH",
    "RLOPT_COLLECT_FASTPATH",
    "PYTORCH_CUDA_ALLOC_CONF",
    "PYTORCH_ALLOC_CONF",
    "TORCHDYNAMO_DISABLE",
    "TORCHINDUCTOR_COMPILE_THREADS",
)

# Packed banks can be larger than device memory. Keep the full arrays mapped on
# the host and transfer sampled rows; explicit user overrides still take priority.
PACKED_DATA_DEFAULTS = (
    "env.data.runtime_cache_device=cpu",
    "env.data.reference_arrays_resident=false",
    "env.data.reference_arrays_warm_workers=0",
)


class MissingRunConfiguration(UserError):
    """A historical standalone checkpoint has no recoverable run description."""


def resolve_checkpoint(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        candidates = [
            p
            for p in path.rglob("model_step_*.pt")
            if p.is_file() and re.fullmatch(r"model_step_\d+\.pt", p.name)
        ]
        if not candidates:
            candidates = [p for p in path.rglob("model.pt") if p.is_file()]
        if not candidates:
            raise UserError(f"No training checkpoint found under {path}")
        path = max(candidates, key=lambda p: p.stat().st_mtime_ns)
    if not path.is_file():
        raise UserError(f"Checkpoint does not exist: {path}")
    return path


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def data_overrides(value: str | Path | None) -> list[str]:
    if value is None:
        return []
    clear = [
        "env.data.cache_dir=null",
        "env.data.persist_dir=null",
        "env.data.clips=null",
        "env.data.takes=null",
    ]
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        manifest = path / "reference_arrays_manifest.json"
        if not manifest.is_file():
            raise UserError(f"Motion bank is missing {manifest.name}: {path}")
        identity = json.loads(manifest.read_text())["key"]["source"]["persist_id"]
        return [
            *clear,
            *PACKED_DATA_DEFAULTS,
            "env.data.manifest=null",
            f"env.data.reference_arrays_dir={json.dumps(str(path))}",
            f"env.data.persist_id={json.dumps(identity)}",
        ]
    if not path.is_file():
        raise UserError(f"Motion manifest does not exist: {path}")
    return [
        *clear,
        "env.data.reference_arrays_dir=null",
        "env.data.persist_id=null",
        f"env.data.manifest={json.dumps(str(path))}",
    ]


def flag_value(argv: list[str], flag: str, default=None):
    for index in range(len(argv) - 1, -1, -1):
        if argv[index].startswith(flag + "="):
            return argv[index].split("=", 1)[1]
        if argv[index] == flag and index + 1 < len(argv):
            return argv[index + 1]
    return default


def checkpoint_run_spec(checkpoint: str | Path) -> dict:
    """Prefer embedded metadata; old runs recover their original command."""
    path = resolve_checkpoint(checkpoint)
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload.get("imitationlab"), dict):
        return _resolve_bundle(payload["imitationlab"], path.parent)
    for root in (path.parent, path.parent.parent):
        metadata = root / "run.json"
        if metadata.is_file():
            return _resolve_bundle(json.loads(metadata.read_text()), root)
        command = root / "command.txt"
        if command.is_file():
            tokens = shlex.split(command.read_text())
            for index, token in enumerate(tokens):
                if (
                    token.endswith("/tracking/train.py")
                    or token == "imitationlab.runners.train"
                ):
                    environment = {}
                    argv = expand_args_files(tokens[index + 1 :], environ=environment)
                    resolved_seed = None
                    agent_config = root / "params" / "agent.yaml"
                    if agent_config.is_file():
                        match = re.search(
                            r"(?m)^seed: (-?\d+)\s*$", agent_config.read_text()
                        )
                        if match:
                            resolved_seed = int(match.group(1))
                    return {
                        "schema_version": 1,
                        "argv": argv,
                        "seed": resolved_seed,
                        "recovered_from": str(command),
                        "environment": {
                            k: v
                            for k, v in environment.items()
                            if k in RUNTIME_ENVIRONMENT
                        },
                    }
    raise MissingRunConfiguration(
        f"{path} has no saved run configuration. Supply --policy and --data "
        "plus its original architecture overrides, or keep command.txt/params with the checkpoint."
    )


def _resolve_bundle(spec: dict, root: Path) -> dict:
    import copy

    spec = copy.deepcopy(spec)
    spec["argv"] = [
        value.replace("${BUNDLE}", str(root)) for value in spec.get("argv", [])
    ]
    encoder = spec.get("encoder")
    if encoder:
        encoder["path"] = encoder["path"].replace("${BUNDLE}", str(root))
        if not Path(encoder["path"]).is_file():
            raise UserError(f"Checkpoint encoder is missing: {encoder['path']}")
        if encoder.get("sha256") and file_sha256(encoder["path"]) != encoder["sha256"]:
            raise UserError(f"Checkpoint encoder hash mismatch: {encoder['path']}")
    return spec


def describe_run(
    argv: list[str], *, policy: str, interface: str, task: str, encoder: str | None
) -> dict:
    environment = {}
    expanded = expand_args_files(argv, environ=environment)
    return {
        "schema_version": 1,
        "task": task,
        "policy": policy,
        "interface": interface,
        "seed": flag_value(expanded, "--seed"),
        "encoder": (
            {"path": encoder, "sha256": file_sha256(encoder)} if encoder else None
        ),
        "argv": expanded,
        "environment": {
            k: os.environ.get(k, v)
            for k, v in environment.items()
            if k in RUNTIME_ENVIRONMENT
        },
    }


def training_run_spec(args, env_cfg, agent_cfg, fastpaths: dict) -> dict:
    """Record the actual resolved task, data identity and execution mode."""
    spec = json.loads(os.environ.get("IMITATIONLAB_RUN_SPEC", "{}"))
    wm = getattr(agent_cfg, "wm", None)
    policy = (
        "ppo"
        if args.algorithm == "PPO"
        else "diffsr"
        if wm.dynamics_model == "diffsr"
        else "svg"
        if wm.actor_source == "real"
        else "supermimic"
    )
    encoder = getattr(env_cfg, "skill_encoder_path", None)
    mode = getattr(env_cfg, "skill_encoder_mode", None)
    spec.update(
        {
            "schema_version": 1,
            "task": args.task,
            "policy": policy,
            "interface": ("fsq" if mode == "sonic_fsq" else "spectral")
            if encoder
            else "explicit",
            "seed": agent_cfg.seed,
            "argv": list(args.launch_argv),
            "environment": {
                key: os.environ[key] for key in RUNTIME_ENVIRONMENT if key in os.environ
            },
            "data": {
                "manifest": str(getattr(env_cfg.data, "manifest", "") or ""),
                "reference_arrays_dir": str(
                    getattr(env_cfg.data, "reference_arrays_dir", "") or ""
                ),
                "persist_id": str(getattr(env_cfg.data, "persist_id", "") or ""),
            },
            "execution": {
                "step_mode": str(getattr(env_cfg, "step_mode", "eager")),
                "fastpaths": fastpaths,
            },
        }
    )
    if encoder:
        spec["encoder"] = {"path": str(encoder), "sha256": file_sha256(encoder)}
    if selection := spec.get("hardware_selection"):
        selection["resolved"] = {
            "num_envs": int(env_cfg.scene.num_envs),
            "wm_batch_size": getattr(wm, "wm_batch_size", None),
            "imagine_batch": getattr(wm, "imagine_batch", None),
            "library_host_mapped": getattr(env_cfg, "library_host_mapped", None),
            "library_env_window": getattr(env_cfg, "library_env_window", None),
        }
    return spec


def restore_runtime_environment(spec: dict) -> None:
    """Restore recorded performance switches without overriding the caller."""
    for key, value in spec.get("environment", {}).items():
        if key in RUNTIME_ENVIRONMENT:
            os.environ.setdefault(key, str(value))
