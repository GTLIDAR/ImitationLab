"""Errors and provenance helpers shared by the cluster launchers."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from imitationlab.paths import REPO_ROOT


class PipelineError(RuntimeError):
    """A gate failed. Deliberately fatal: a partial run must not look complete."""


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 of a file, streamed so multi-GB checkpoints are fine."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(payload: Any) -> str:
    """Stable JSON for hashing: sorted keys, no incidental whitespace."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def git_state() -> dict[str, Any]:
    """Best-effort git description of the working tree.

    Cluster jobs run from an extracted tarball with no ``.git``, so every field
    is optional and a failure here is recorded rather than raised.
    """

    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    head = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "available": head is not None,
        "commit": head,
        "branch": _git("branch", "--show-current"),
        "dirty": None if status is None else bool(status.strip()),
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
