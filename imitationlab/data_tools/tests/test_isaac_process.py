import os
import subprocess
import sys

import pytest

from imitationlab.data_tools.common.isaac_process import run_isaac

# Exits at once after forking a child that would otherwise sleep for a minute,
# like a Kit helper that outlives the converter.
LEAKY = (
    "import os, sys, time\n"
    "pid = os.fork()\n"
    "if pid == 0:\n"
    "    time.sleep(60); os._exit(0)\n"
    "print(pid, flush=True); sys.exit(int(sys.argv[1]))\n"
)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    with open(f"/proc/{pid}/stat") as stat:
        return stat.read().split()[2] != "Z"


def _run(code: int, tmp_path):
    out = tmp_path / "pid.txt"
    cmd = [
        sys.executable,
        "-c",
        f"import sys; sys.stdout = open({str(out)!r}, 'w')\n" + LEAKY,
        str(code),
    ]
    env = {**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"}
    try:
        run_isaac(cmd, environ=env, grace=2.0)
    finally:
        child = int(out.read_text().split()[0])
    return child


def test_unattended_converter_leaves_no_child_behind(tmp_path):
    child = _run(0, tmp_path)
    assert not _alive(child)


def test_failure_raises_and_still_cleans_up(tmp_path):
    with pytest.raises(subprocess.CalledProcessError):
        _run(3, tmp_path)
    child = int((tmp_path / "pid.txt").read_text().split()[0])
    assert not _alive(child)
