"""Export a checkpoint and its command encoder as a verifiable native bundle."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import tempfile
from pathlib import Path

from imitationlab.runners.artifacts import (
    checkpoint_run_spec,
    file_sha256,
    resolve_checkpoint,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, prog="imitationlab export")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args, rest = parser.parse_known_args(argv)
    if rest:
        parser.error(f"Unrecognized arguments: {' '.join(rest)}")
    target = args.output.expanduser().resolve()
    if target.exists():
        parser.error(f"Output already exists: {target}")
    source = resolve_checkpoint(args.checkpoint)
    metadata = copy.deepcopy(checkpoint_run_spec(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    import torch

    payload = torch.load(source, map_location="cpu", weights_only=False)
    with tempfile.TemporaryDirectory(prefix=".export-", dir=target.parent) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        encoder = (metadata.get("encoder") or {}).get("path")
        if not encoder:
            for arg in metadata.get("argv", []):
                if arg.startswith("env.skill_encoder_path="):
                    encoder = arg.split("=", 1)[1].strip("\"'")
        if encoder:
            shutil.copy2(encoder, staging / "encoder.pt")
            metadata["encoder"] = {
                "path": "${BUNDLE}/encoder.pt",
                "sha256": file_sha256(staging / "encoder.pt"),
            }
            metadata["argv"] = [
                item
                for item in metadata["argv"]
                if not item.startswith("env.skill_encoder_path=")
            ]
            metadata["argv"].append('env.skill_encoder_path="${BUNDLE}/encoder.pt"')
        payload["imitationlab"] = metadata
        torch.save(payload, staging / "model.pt")
        (staging / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
        checksums = {p.name: file_sha256(p) for p in staging.iterdir() if p.is_file()}
        (staging / "checksums.json").write_text(json.dumps(checksums, indent=2) + "\n")
        staging.rename(target)
    print(target)
    return 0
