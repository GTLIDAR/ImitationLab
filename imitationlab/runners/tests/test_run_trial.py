import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from imitationlab.runners.tuning import run_trial

# The runner starts a child in its own session that records its PID and sleeps,
# like a training process launched by `imitationlab benchmark run`.
RUNNER = """
import json, os, sys
from pathlib import Path
from imitationlab.runners.tuning import run_trial
child = [sys.executable, "-c",
         "import os, time; print(os.getpid(), flush=True); time.sleep(120)"]
code = run_trial(child, dict(os.environ), Path(sys.argv[1]), timeout=120)
print(json.dumps({"returned": code}))
"""


def _child_pid(log: Path) -> int:
    for _ in range(200):
        lines = log.read_text().splitlines() if log.exists() else []
        if len(lines) > 1:
            return int(lines[1])
        time.sleep(0.05)
    raise AssertionError("child never started")


def _gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def test_terminating_the_runner_stops_the_trial(tmp_path):
    log = tmp_path / "trial.log"
    runner = subprocess.Popen(
        [sys.executable, "-c", RUNNER, str(log)], stdout=subprocess.PIPE, text=True
    )
    child = _child_pid(log)
    runner.send_signal(signal.SIGTERM)
    assert runner.wait(timeout=30) == 128 + signal.SIGTERM
    assert _gone(child)


def test_timeout_still_returns_124(tmp_path):
    command = [sys.executable, "-c", "import time; time.sleep(30)"]
    assert run_trial(command, dict(os.environ), tmp_path / "t.log", timeout=0.5) == 124
    assert json.loads((tmp_path / "t.log").read_text().splitlines()[0]) == command


def test_signal_handlers_are_restored(tmp_path):
    before = signal.getsignal(signal.SIGTERM)
    run_trial([sys.executable, "-c", "pass"], dict(os.environ), tmp_path / "t.log", 30)
    assert signal.getsignal(signal.SIGTERM) is before
