"""Cell selection and the one-process-per-cell argv of `eval_checkpoint_tree`."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from imitationlab.evaluation import evaluate_checkpoint_tree as tree


def _cells(*frames):
    from imitationlab.evaluation.score_tree import Cell

    return [
        Cell(frames=f, checkpoint=Path(f"c{f}.pt"), output_json=Path(f"o{f}.json"))
        for f in frames
    ]


def test_select_cells_keeps_all_without_a_filter():
    cells = _cells(1, 2, 3)
    assert tree.select_cells(cells, None) == cells


def test_select_cells_filters_by_frames():
    assert [c.frames for c in tree.select_cells(_cells(1, 2, 3), [3, 1])] == [1, 3]


def test_single_cell_argv_drops_isolation_and_keeps_evaluator_args():
    argv = [
        "eval_checkpoint_tree.py",
        "--tree",
        "t",
        "--isolate-cells",
        "--arm",
        "a",
        "--",
        "--num-envs",
        "4096",
    ]
    assert tree.single_cell_argv(argv, 250) == [
        "eval_checkpoint_tree.py",
        "--tree",
        "t",
        "--arm",
        "a",
        "--only-frames",
        "250",
        "--",
        "--num-envs",
        "4096",
    ]


def test_single_cell_argv_refuses_a_preselected_cell():
    with pytest.raises(ValueError):
        tree.single_cell_argv(
            ["x.py", "--only-frames", "1", "--isolate-cells", "--"], 2
        )


def test_run_isolated_fails_when_a_cell_writes_no_row(tmp_path, monkeypatch):
    cells = [
        type(c)(
            frames=c.frames,
            checkpoint=c.checkpoint,
            output_json=tmp_path / c.output_json.name,
        )
        for c in _cells(1, 2)
    ]
    (tmp_path / "o1.json").write_text("{}")  # cell 1 scored, cell 2 wrote nothing
    calls = []
    monkeypatch.setattr(tree.subprocess, "call", lambda cmd: calls.append(cmd) or 0)
    assert (
        tree.run_isolated(cells, ["x.py", "--isolate-cells", "--", "--seed", "0"]) == 1
    )
    # cell 2 wrote nothing, so it is retried once before counting as failed
    assert [cmd[cmd.index("--only-frames") + 1] for cmd in calls] == ["1", "2", "2"]
    assert all(cmd[0] == sys.executable for cmd in calls)


def test_run_isolated_accepts_a_row_written_before_a_teardown_crash(
    tmp_path, monkeypatch
):
    cells = [
        type(c)(
            frames=c.frames,
            checkpoint=c.checkpoint,
            output_json=tmp_path / c.output_json.name,
        )
        for c in _cells(1)
    ]

    def crash_after_writing(cmd):
        (tmp_path / "o1.json").write_text("{}")
        return 139

    monkeypatch.setattr(tree.subprocess, "call", crash_after_writing)
    assert tree.run_isolated(cells, ["x.py", "--isolate-cells", "--"]) == 0


def test_run_isolated_retries_a_startup_crash(tmp_path, monkeypatch):
    cells = [
        type(c)(
            frames=c.frames,
            checkpoint=c.checkpoint,
            output_json=tmp_path / c.output_json.name,
        )
        for c in _cells(1)
    ]
    attempts = []

    def crash_then_succeed(cmd):
        attempts.append(cmd)
        if len(attempts) == 2:
            (tmp_path / "o1.json").write_text("{}")
            return 0
        return 139

    monkeypatch.setattr(tree.subprocess, "call", crash_then_succeed)
    assert tree.run_isolated(cells, ["x.py", "--isolate-cells", "--"]) == 0
    assert len(attempts) == 2


def test_hang_guard_kills_a_process_that_outlives_its_summary(tmp_path):
    out = tmp_path / "row.json"
    script = f"import json, time; open({str(out)!r}, 'w').write(json.dumps({{'ok': 1}})); time.sleep(600)"
    start = __import__("time").monotonic()
    code = tree.call_with_hang_guard(
        [sys.executable, "-c", script], out, grace_seconds=0.5, poll_seconds=0.1
    )
    assert code == 0 and out.is_file()
    assert __import__("time").monotonic() - start < 60


def test_hang_guard_returns_the_exit_code_of_a_process_that_exits(tmp_path):
    code = tree.call_with_hang_guard(
        [sys.executable, "-c", "raise SystemExit(3)"],
        tmp_path / "none.json",
        grace_seconds=0.5,
        poll_seconds=0.1,
    )
    assert code == 3


def test_run_isolated_scores_a_fresh_summary_despite_a_nonzero_exit(
    tmp_path, monkeypatch, capsys
):
    """Isaac Sim's close timeout exits 1 after a complete summary; a stale
    summary from an earlier run does not count."""
    import os

    cells = [
        type(c)(
            frames=c.frames,
            checkpoint=c.checkpoint,
            output_json=tmp_path / c.output_json.name,
        )
        for c in _cells(1, 2)
    ]
    (tmp_path / "o2.json").write_text("{}")
    os.utime(tmp_path / "o2.json", (0, 0))

    def fake_call(cmd):
        if cmd[cmd.index("--only-frames") + 1] == "1":
            (tmp_path / "o1.json").write_text("{}")
        return 1

    monkeypatch.setattr(tree.subprocess, "call", fake_call)
    assert tree.run_isolated(cells, ["x.py", "--isolate-cells", "--"]) == 1
    out = capsys.readouterr().out
    assert "cell f1 exited 1 after writing its summary; scored" in out
    assert "cell f2 failed (exit 1)" in out and "cell(s) failed: [2]" in out


def test_thinned_frames_keeps_the_first_milestone_of_each_bucket() -> None:
    from imitationlab.evaluation.evaluate_checkpoint_tree import thinned_frames

    frames = [50, 100, 150, 200, 250, 390, 401]
    assert thinned_frames(frames, 200) == {50, 200, 401}
    # The newest milestone is kept even inside a bucket, so a final checkpoint scores.
    assert thinned_frames(frames[:6], 200) == {50, 200, 390}
    assert thinned_frames([], 200) == set()


def test_a_single_cell_keeps_the_hang_guard(monkeypatch, tmp_path):
    seen = {}

    def isolated(cells, argv, *, hang_grace_seconds):
        seen.update(cells=len(cells), grace=hang_grace_seconds)
        return 0

    monkeypatch.setattr(tree, "plan_cells", lambda *a, **k: _cells(5))
    monkeypatch.setattr(tree, "run_isolated", isolated)
    monkeypatch.setattr(
        tree.sys,
        "argv",
        [
            "x.py",
            "--tree",
            str(tmp_path),
            "--output-root",
            str(tmp_path / "out"),
            "--arm",
            "a",
            "--isolate-cells",
            "--hang-grace-seconds",
            "60",
            "--",
            "--seed",
            "0",
        ],
    )

    assert tree.main() == 0
    assert seen == {"cells": 1, "grace": 60.0}


def test_a_child_keeps_its_cell_when_a_live_run_saves_a_newer_checkpoint(
    monkeypatch, tmp_path
):
    seen = {}
    monkeypatch.setattr(tree, "plan_cells", lambda *a, **k: _cells(100, 115, 120))
    monkeypatch.setattr(
        tree, "run_isolated", lambda cells, argv, **k: seen.update(n=len(cells)) or 0
    )
    monkeypatch.setattr(
        tree.sys,
        "argv",
        [
            "x.py",
            "--tree",
            str(tmp_path),
            "--output-root",
            str(tmp_path / "out"),
            "--arm",
            "a",
            "--min-frame-gap",
            "100",
            "--only-frames",
            "115",
            "--isolate-cells",
            "--hang-grace-seconds",
            "60",
            "--",
            "--seed",
            "0",
        ],
    )

    assert tree.main() == 0
    assert seen == {"n": 1}


# ------------------------------------------------------------------ --group
def test_group_child_argv_replaces_the_parents_frames() -> None:
    argv = [
        "x.py",
        "--tree",
        "t",
        "--only-frames",
        "1",
        "2",
        "3",
        "--group",
        "2",
        "--",
        "--seed",
        "0",
    ]
    assert tree.group_child_argv(argv, [2, 3]) == [
        "x.py",
        "--tree",
        "t",
        "--group",
        "2",
        "--group-child",
        "--only-frames",
        "2",
        "3",
        "--",
        "--seed",
        "0",
    ]


def test_passthrough_ranks_reads_the_rank_list() -> None:
    assert tree.passthrough_ranks(
        ["--a", "1", "--trajectory-ranks", "4", "5", "--b"]
    ) == ["4", "5"]
    assert tree.passthrough_ranks(["--seed", "0"]) == []


def _grouped_argv(tmp_path, *extra):
    return [
        "x.py",
        "--tree",
        str(tmp_path),
        "--output-root",
        str(tmp_path / "replica{seed_suffix}"),
        "--arm",
        "a",
        "--row",
        "replica",
        "--group",
        "2",
        "--repeats",
        "2",
        *extra,
        "--",
        "--seed",
        "0",
        "--trajectory-ranks",
        "0",
        "1",
        "2",
    ]


def _repeat_cells(frames, repeats, root):
    from imitationlab.evaluation.score_tree import Cell

    return tuple(
        Cell(
            frames=f,
            checkpoint=Path(f"c{f}.pt"),
            output_json=root / f"r{r}" / f"o{f}.json",
            seed_index=r,
        )
        for f in frames
        for r in range(repeats)
    )


def test_group_mode_plans_groups_of_checkpoints(monkeypatch, tmp_path) -> None:
    seen = {}
    monkeypatch.setattr(
        tree,
        "plan_repeat_cells",
        lambda *a, **k: _repeat_cells([10, 20, 30], k["repeats"], tmp_path),
    )

    def groups(groups, argv, *, clips, hang_grace_seconds):
        seen.update(
            frames=[sorted({c.frames for c in g}) for g in groups],
            blocks=[len(g) for g in groups],
            clips=clips,
            grace=hang_grace_seconds,
        )
        return 0

    monkeypatch.setattr(tree, "run_groups", groups)
    monkeypatch.setattr(tree.sys, "argv", _grouped_argv(tmp_path))
    assert tree.main() == 0
    assert seen == {
        "frames": [[10, 20], [30]],
        "blocks": [4, 2],
        "clips": 3,
        "grace": 120.0,
    }


def test_a_group_child_runs_its_blocks_in_one_evaluator(monkeypatch, tmp_path) -> None:
    seen = {}
    monkeypatch.setattr(
        tree,
        "plan_repeat_cells",
        lambda *a, **k: _repeat_cells([10, 20, 30], 2, tmp_path),
    )
    monkeypatch.setattr(tree, "_run_evaluator", lambda args: seen.update(args=args))
    monkeypatch.setattr(
        tree.sys,
        "argv",
        _grouped_argv(tmp_path, "--group-child", "--only-frames", "10", "20"),
    )
    assert tree.main() == 0
    args = seen["args"]
    assert args[args.index("--num-envs") + 1] == "12"  # 3 clips x 4 blocks
    assert "--grouped" in args
    checkpoints = args[args.index("--checkpoints") + 1 : args.index("--output-jsons")]
    assert checkpoints == ["c10.pt", "c10.pt", "c20.pt", "c20.pt"]
    outputs = args[args.index("--output-jsons") + 1 :]
    assert outputs == [
        str(tmp_path / f"r{r}" / f"o{f}.json") for f in (10, 20) for r in range(2)
    ]
    indices = args[
        args.index("--grouped-seed-indices") + 1 : args.index("--checkpoints")
    ]
    assert indices == ["0", "1", "0", "1"]


@pytest.mark.parametrize(
    "extra",
    [["--isolate-cells"], ["--", "--num-envs", "6"], ["--", "--checkpoints", "x"]],
)
def test_group_mode_refuses_conflicting_flags(monkeypatch, tmp_path, extra) -> None:
    argv = _grouped_argv(tmp_path)
    if extra[0] == "--":
        argv = argv + extra[1:]
    else:
        argv = argv[:-7] + extra + argv[-7:]
    monkeypatch.setattr(tree.sys, "argv", argv)
    with pytest.raises(SystemExit):
        tree.main()


def test_run_groups_retries_a_group_with_a_missing_row(monkeypatch, tmp_path) -> None:
    from imitationlab.evaluation.score_tree import Cell

    group = tuple(
        Cell(frames=1, checkpoint=Path("c1.pt"), output_json=tmp_path / f"o{r}.json")
        for r in range(2)
    )
    calls = []

    def guard(cmd, outputs, grace, *, fresh_after):
        calls.append(cmd)
        outputs[0].write_text("{}")  # the second row never arrives
        return 0

    monkeypatch.setattr(tree, "call_with_hang_guard", guard)
    assert (
        tree.run_groups(
            [group], ["x.py", "--group", "1", "--"], clips=3, hang_grace_seconds=5
        )
        == 1
    )
    assert len(calls) == 2
    assert (
        calls[0][calls[0].index("--only-frames") + 1] == "1"
        and "--group-child" in calls[0]
    )


def test_run_groups_scores_a_group_whose_rows_are_all_in(
    monkeypatch, tmp_path, capsys
) -> None:
    from imitationlab.evaluation.score_tree import Cell

    group = tuple(
        Cell(frames=1, checkpoint=Path("c1.pt"), output_json=tmp_path / f"o{r}.json")
        for r in range(2)
    )

    def guard(cmd, outputs, grace, *, fresh_after):
        for path in outputs:
            path.write_text("{}")
        return 1

    monkeypatch.setattr(tree, "call_with_hang_guard", guard)
    assert tree.run_groups([group], ["x.py", "--"], clips=3, hang_grace_seconds=5) == 0
    assert "exited 1 after writing all its rows; scored" in capsys.readouterr().out


def test_hang_guard_waits_for_every_row_of_a_group(tmp_path) -> None:
    import time

    rows = [tmp_path / "a.json", tmp_path / "b.json"]
    rows[1].write_text("{}")
    import os

    os.utime(rows[1], (0, 0))  # a row left by an earlier run does not start the clock
    script = (
        "import json, time, sys; time.sleep(1.0); "
        f"[open(p, 'w').write(json.dumps({{'ok': 1}})) for p in {[str(p) for p in rows]!r}]; "
        "time.sleep(600)"
    )
    start = time.monotonic()
    code = tree.call_with_hang_guard(
        [sys.executable, "-c", script],
        rows,
        grace_seconds=0.5,
        poll_seconds=0.1,
        fresh_after=time.time(),
    )
    assert code == 0
    assert 1.0 < time.monotonic() - start < 60


# ------------------------------------------------- the group's session on exit
def _alive(pid: int) -> bool:
    """Whether ``pid`` runs (a zombie does not count)."""
    import os

    try:
        os.kill(pid, 0)
        stat = Path(f"/proc/{pid}/stat").read_text()
    except (ProcessLookupError, OSError):
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


def _wait_for_pid(pid_file: Path, timeout: float = 60.0) -> int:
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        text = pid_file.read_text() if pid_file.is_file() else ""
        if text:
            return int(text)
        time.sleep(0.05)
    raise AssertionError(f"no pid in {pid_file} after {timeout} s")


def _sleeper(pid_file: Path) -> str:
    """A stand-in evaluator that hangs before writing any row."""
    return (
        f"import os, time; open({str(pid_file)!r}, 'w').write(str(os.getpid())); "
        "time.sleep(600)"
    )


_DRIVER = """
import sys
from pathlib import Path

from imitationlab.evaluation import evaluate_checkpoint_tree as tree
tree.call_with_hang_guard(
    [sys.executable, "-c", {child!r}], [Path({row!r})], grace_seconds=120, poll_seconds=0.1
)
"""


@pytest.mark.parametrize("signum", ["SIGTERM", "SIGHUP"])
def test_a_launcher_signal_kills_the_group_session(tmp_path, signum) -> None:
    """A launcher ``timeout`` (SIGTERM) or a closed terminal (SIGHUP) ends the tree
    before any row is written. The group's evaluator runs in its own session, so
    the signal does not reach it; the tree must kill it before it exits, or it
    keeps the GPU after the launcher's lock is free."""
    import os
    import signal
    import subprocess
    import time

    number = getattr(signal, signum)
    pid_file = tmp_path / "child.pid"
    driver = tmp_path / "driver.py"
    driver.write_text(
        _DRIVER.format(child=_sleeper(pid_file), row=str(tmp_path / "row.json"))
    )
    proc = subprocess.Popen([sys.executable, str(driver)])
    pid = None
    try:
        pid = _wait_for_pid(pid_file)
        assert _alive(pid)
        proc.send_signal(number)
        assert proc.wait(timeout=60) == 128 + number
        deadline = time.monotonic() + 10
        while _alive(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _alive(pid), "the group's evaluator outlived the tree"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        if pid is not None and _alive(pid):
            os.killpg(pid, signal.SIGKILL)


def test_an_interrupt_kills_the_group_session(monkeypatch, tmp_path) -> None:
    """Ctrl-C reaches only the tree (the group has its own session); the tree
    kills the group's session and reaps it before the interrupt propagates."""
    import os
    import signal

    pid_file = tmp_path / "child.pid"

    def interrupt_once_started(path, after):
        if pid_file.is_file() and pid_file.read_text():
            raise KeyboardInterrupt
        return False

    monkeypatch.setattr(tree, "_fresh_json", interrupt_once_started)
    try:
        with pytest.raises(KeyboardInterrupt):
            tree.call_with_hang_guard(
                [sys.executable, "-c", _sleeper(pid_file)],
                tmp_path / "row.json",
                grace_seconds=120,
                poll_seconds=0.05,
            )
        assert not _alive(int(pid_file.read_text()))
    finally:
        if (
            pid_file.is_file()
            and pid_file.read_text()
            and _alive(int(pid_file.read_text()))
        ):
            os.killpg(int(pid_file.read_text()), signal.SIGKILL)


def test_the_hang_guard_restores_the_signal_handlers(tmp_path) -> None:
    import signal

    before = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP)}
    assert (
        tree.call_with_hang_guard(
            [sys.executable, "-c", "pass"],
            tmp_path / "none.json",
            grace_seconds=0.5,
            poll_seconds=0.1,
        )
        == 0
    )
    assert {s: signal.getsignal(s) for s in before} == before


def test_a_rerun_child_passes_each_rows_seed_index(monkeypatch, tmp_path) -> None:
    """A rerun plans only the missing rows. Each block's seed index must still be
    the one that names its row (``replica_s2`` is index 2), not the block's order
    among this process's blocks of its checkpoint."""
    run = tmp_path / "run"
    for frames in (100, 200):
        models = run / "tracker" / f"f{frames}" / "models"
        models.mkdir(parents=True)
        (models / f"model_step_{frames}.pt").write_bytes(b"weights")
    for index, folder in (
        (0, "replica"),
        (1, "replica_s1"),
    ):  # f100's repeats 0 and 1 are in
        (tmp_path / folder).mkdir()
        (tmp_path / folder / f"a_seed{index}_replica_f100.json").write_text("{}")
    seen = {}
    monkeypatch.setattr(tree, "_run_evaluator", lambda args: seen.update(args=args))
    monkeypatch.setattr(
        tree.sys,
        "argv",
        [
            "x.py",
            "--tree",
            str(run),
            "--output-root",
            str(tmp_path / "replica{seed_suffix}"),
            "--arm",
            "a",
            "--row",
            "replica",
            "--group",
            "2",
            "--repeats",
            "4",
            "--group-child",
            "--only-frames",
            "100",
            "200",
            "--",
            "--seed",
            "0",
            "--trajectory-ranks",
            "0",
            "1",
            "2",
        ],
    )
    assert tree.main() == 0
    args = seen["args"]
    indices = args[
        args.index("--grouped-seed-indices") + 1 : args.index("--checkpoints")
    ]
    outputs = args[args.index("--output-jsons") + 1 :]
    assert [Path(p).parent.name for p in outputs] == [
        "replica_s2",
        "replica_s3",
        "replica",
        "replica_s1",
        "replica_s2",
        "replica_s3",
    ]
    assert indices == ["2", "3", "0", "1", "2", "3"]
    assert args[args.index("--num-envs") + 1] == "18"  # 3 clips x 6 blocks
