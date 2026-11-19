"""Run an Isaac Sim converter process without leaving stray processes behind."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from collections.abc import Sequence


def _accepts_eula_unattended(environ) -> bool:
    return environ.get("OMNI_KIT_ACCEPT_EULA", "").strip().upper() in {"YES", "Y"}


def _stop_group(pgid: int, grace: float) -> None:
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_isaac(
    cmd: Sequence[str], *, environ=None, grace: float = 10.0
) -> subprocess.CompletedProcess:
    """Like ``subprocess.run(cmd, check=True)`` for a Kit-based converter.

    Kit can fork helpers that outlive the converter; one held 3.3 GB of GPU
    memory for hours after a batch had finished. When the EULA is accepted
    through the environment, the converter runs in its own session and any
    process left in it is stopped once the converter exits. In a terminal
    without that variable it runs in the caller's session so Kit can prompt.
    """
    environ = os.environ if environ is None else environ
    if not _accepts_eula_unattended(environ):
        return subprocess.run(list(cmd), check=True)
    process = subprocess.Popen(list(cmd), start_new_session=True)
    try:
        code = process.wait()
    finally:
        _stop_group(process.pid, grace)
        process.wait()
    if code:
        raise subprocess.CalledProcessError(code, list(cmd))
    return subprocess.CompletedProcess(list(cmd), code)
