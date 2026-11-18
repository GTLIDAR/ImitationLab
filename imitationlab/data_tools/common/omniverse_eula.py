"""Fail early when Omniverse Kit would wait on a EULA prompt nobody can answer."""

from __future__ import annotations

import os
import sys
from typing import TextIO

EULA_URL = (
    "https://docs.omniverse.nvidia.com/platform/latest/common/"
    "NVIDIA_Omniverse_License_Agreement.html"
)


def eula_message() -> str:
    return (
        "Isaac Sim (Omniverse Kit) requires accepting the NVIDIA Omniverse License "
        f"Agreement: {EULA_URL}\n"
        "This command is not attached to a terminal, so Kit cannot ask. After "
        "reading the agreement, accept it by setting OMNI_KIT_ACCEPT_EULA=YES "
        "in the environment and run the command again."
    )


def require_eula_answerable(
    environ: dict[str, str] | None = None, stdin: TextIO | None = None
) -> None:
    """Exit with an explanation if Kit's EULA prompt would fail.

    Kit accepts the EULA from ``OMNI_KIT_ACCEPT_EULA`` or from an interactive
    prompt. Without either, it aborts with a terse error after the batch has
    started, so check before launching it.
    """
    environ = os.environ if environ is None else environ
    stdin = sys.stdin if stdin is None else stdin
    if environ.get("OMNI_KIT_ACCEPT_EULA", "").strip().upper() in {"YES", "Y"}:
        return
    if stdin is not None and stdin.isatty():
        return
    raise SystemExit(eula_message())
