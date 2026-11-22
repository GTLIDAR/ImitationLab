#!/usr/bin/env python3
"""Score every milestone of one checkpoint tree in a single process.

`imitationlab.evaluation.evaluate_checkpoint` starts Isaac Sim at import
time, so scoring N checkpoints as N processes pays N simulation starts. At the
256-clip board that start is most of the per-cell cost. An arm fixes the
command interface, so the only thing that changes between its milestones is the
policy weights: one start can serve the whole budget axis.

This entrypoint plans the cells with
`imitationlab.evaluation.score_tree`, then hands the whole list to the
evaluator through `--checkpoints` / `--output-jsons`. Every argument after `--`
goes to the evaluator verbatim, so the interface overrides stay exactly the
ones the campaign launcher would pass for a single cell.

    python -m imitationlab.evaluation.evaluate_checkpoint_tree \
        --tree /data/pareto_stack/jepa_h1_ee_wide_seed0 \
        --output-root /data/eval/pareto_stack --arm jepa_h1_ee_wide --seed 0 \
        --row milestone -- --task imitation-g1-v1 --algo PPO ...

It exists as a script rather than a `-m` invocation because a cluster stage
runs one script path inside the container.

`--group G --repeats R` (off by default) is NDM Gym's grouped evaluation: up to G
checkpoints, each R times, in one simulator of C x G x R envs, where C is the
board's clip count (`evaluation/grouped_eval.py`). Repeat r is evaluation index
`--seed + r`; its rows land under `--output-root` with `{seed}` or
`{seed_suffix}` filled in (`replica{seed_suffix}` gives `replica`, `replica_s1`,
...), named as one-cell runs name them, and the evaluator records `--seed + r` as
`metadata.grouped.seed_index`. Each group runs in its own process and session,
which writes its rows and exits without Kit's teardown; the parent also kills a
group process that outlives its complete rows by `--hang-grace-seconds` (120 s
unless given), and kills the group's session when the parent itself is stopped
(SIGTERM, SIGHUP or Ctrl-C).

    python -m imitationlab.evaluation.evaluate_checkpoint_tree --tree <run> \
        --output-root '<run>/replica{seed_suffix}' --arm <arm> --seed 0 \
        --row replica --board lafan1_all40_v1 --group 12 --repeats 4 -- \
        --task imitation-g1-v1 --algo WMBPTT ...
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import time

# Stdlib-only, and safe before Isaac Sim or Torch: `score_tree` plans cells from
# the filesystem alone, and `runtime_bootstrap` is the split-runtime helper.
from imitationlab.evaluation.score_tree import (
    group_cells,
    plan_cells,
    plan_repeat_cells,
)
from imitationlab.runners.runtime_bootstrap import (
    configure_cu130_bridge,
    verify_cu130_torch,
)

EVALUATOR = "imitationlab.evaluation.evaluate_checkpoint"
TREE_EVALUATOR = "imitationlab.evaluation.evaluate_checkpoint_tree"
GROUP_HANG_GRACE_SECONDS = 120.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--arm", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--row", type=str, default="milestone")
    # A board name instead of 256 or 4096 literal ranks in a config file. The
    # ranks and `--num-envs` are expanded here, so a campaign declares the
    # board it scores and cannot drift from the board's definition.
    parser.add_argument("--board", type=str, default=None)
    parser.add_argument("--final-only", action="store_true", default=False)
    # Off by default: an interrupted job resumes instead of repeating cells.
    parser.add_argument("--rescore", action="store_true", default=False)
    # One evaluator process per cell. Needed when the evaluator samples its own
    # resets (`--task-reset-selection`): each cell then draws a different
    # population, which the shared-start path refuses, and a shared env would
    # carry the adaptive reset sampler's state from one checkpoint into the next.
    parser.add_argument("--isolate-cells", action="store_true", default=False)
    # Score only these milestones (frame counts); the isolated path passes one.
    parser.add_argument("--only-frames", type=int, nargs="+", default=None)
    # Isolated cells and groups: kill a cell's (group's) process this many
    # seconds after its summary JSON(s) are complete (0 waits forever; unset is 0
    # for isolated cells and 120 for groups). The evaluator can hang after
    # writing it -- one hung 9 h on the workstation, 2026-10-23 -- which would
    # stall every later cell until the job's walltime.
    parser.add_argument("--hang-grace-seconds", type=float, default=None)
    # Score one milestone per bucket of this many frames (0: all). Buckets come
    # from every milestone, scored or not, so repeat runs pick the same ones.
    parser.add_argument("--min-frame-gap", type=int, default=0)
    # Grouped evaluation (off by default): up to this many checkpoints per
    # simulator, each --repeats times, one process per group.
    parser.add_argument("--group", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    # Internal: a group's own process, which scores its one group in-process.
    parser.add_argument("--group-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "evaluator_args",
        nargs=argparse.REMAINDER,
        help="Arguments after `--`, passed to the evaluator verbatim.",
    )
    args = parser.parse_args()

    passthrough = list(args.evaluator_args)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    if not passthrough:
        parser.error("pass the evaluator arguments after `--`.")

    if args.board is not None:
        for flag in ("--trajectory-ranks", "--num-envs"):
            if flag in passthrough:
                parser.error(f"--board already sets {flag}; pass one or the other.")
        from imitationlab.evaluation.protocol import BOARDS

        if args.board not in BOARDS:
            parser.error(f"unknown board: {args.board}")
        ranks = [str(case.trajectory_rank) for case in BOARDS[args.board].cases]
        if not ranks:
            parser.error(f"board {args.board} has no cases.")
        if args.group > 0:
            # A group sets --num-envs to the clips times its blocks.
            passthrough += ["--trajectory-ranks", *ranks]
        else:
            passthrough += ["--num-envs", str(len(ranks)), "--trajectory-ranks", *ranks]
        print(f"[INFO] board {args.board}: {len(ranks)} clips")

    if args.group > 0 or args.repeats != 1 or args.group_child:
        return grouped_main(parser, args, passthrough)
    if args.hang_grace_seconds is None:
        args.hang_grace_seconds = 0.0

    cells = plan_cells(
        args.tree,
        args.output_root,
        arm=args.arm,
        seed=args.seed,
        row=args.row,
        final_only=args.final_only,
        skip_scored=not args.rescore,
    )
    # An isolated child's `--only-frames` was thinned by its parent; thinning again
    # would drop it once a live run has saved a newer checkpoint.
    if args.min_frame_gap > 0 and args.only_frames is None:
        every = plan_cells(
            args.tree,
            args.output_root,
            arm=args.arm,
            seed=args.seed,
            row=args.row,
            final_only=args.final_only,
            skip_scored=False,
        )
        kept = thinned_frames([c.frames for c in every], args.min_frame_gap)
        cells = [c for c in cells if c.frames in kept]
    cells = select_cells(cells, args.only_frames)
    if not cells:
        print(
            f"[INFO] nothing to score: {args.arm} seed{args.seed} {args.row} "
            f"({args.tree})"
        )
        return 0

    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.isolate_cells and (len(cells) > 1 or args.hang_grace_seconds > 0):
        print(
            f"[INFO] {args.arm} seed{args.seed} {args.row}: {len(cells)} cell(s), "
            f"one process each"
        )
        return run_isolated(cells, sys.argv, hang_grace_seconds=args.hang_grace_seconds)
    print(
        f"[INFO] {args.arm} seed{args.seed} {args.row}: {len(cells)} cell(s) "
        f"in one simulation start"
    )
    for cell in cells:
        print(f"[INFO]   f{cell.frames} -> {cell.output_json.name}")

    # In the split container runtime, Kit's Python is the only interpreter with
    # both Kit and Torch -- but Torch lives in the CU130 site-packages and has
    # to be put on the path first. The evaluator imports Torch at module level,
    # so this has to happen before `run_module`. Outside the container the
    # bridge finds nothing and does nothing.
    site_packages = configure_cu130_bridge(
        required=os.environ.get("ISAACLAB_REQUIRE_CU130_RUNTIME") == "1"
    )
    if site_packages is not None:
        # Import the CU130 Torch stack before Kit extensions can load Isaac
        # Sim's bundled NCCL into the process.
        verify_cu130_torch(site_packages)

    sys.argv = (
        [f"{EVALUATOR}.py"]
        + passthrough
        + ["--checkpoints"]
        + [str(cell.checkpoint) for cell in cells]
        + ["--output-jsons"]
        + [str(cell.output_json) for cell in cells]
    )
    runpy.run_module(EVALUATOR, run_name="__main__")
    return 0


def passthrough_ranks(passthrough: list[str]) -> list[str]:
    """The values after ``--trajectory-ranks`` in the evaluator arguments."""
    if "--trajectory-ranks" not in passthrough:
        return []
    ranks = []
    for token in passthrough[passthrough.index("--trajectory-ranks") + 1 :]:
        if token.startswith("--"):
            break
        ranks.append(token)
    return ranks


def grouped_main(
    parser: argparse.ArgumentParser, args: argparse.Namespace, passthrough: list[str]
) -> int:
    """``--group G --repeats R``: plan the cells into groups and score each group."""
    if args.group < 1:
        parser.error("--repeats and the group child need --group >= 1.")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1.")
    if args.isolate_cells:
        parser.error("--group runs one process per group; drop --isolate-cells.")
    if "--num-envs" in passthrough:
        parser.error("--group sets --num-envs (clips x blocks); do not pass it.")
    for flag in (
        "--checkpoint",
        "--checkpoints",
        "--output-json",
        "--output-jsons",
        "--grouped",
    ):
        if flag in passthrough:
            parser.error(f"--group sets {flag}; do not pass it.")
    clips = len(passthrough_ranks(passthrough))
    if clips == 0:
        parser.error(
            "--group needs --board or --trajectory-ranks: the clips of every block."
        )

    cells = plan_repeat_cells(
        args.tree,
        args.output_root,
        arm=args.arm,
        seed=args.seed,
        repeats=args.repeats,
        row=args.row,
        final_only=args.final_only,
        skip_scored=not args.rescore,
    )
    if args.min_frame_gap > 0 and args.only_frames is None:
        every = plan_cells(
            args.tree,
            args.output_root,
            arm=args.arm,
            seed=args.seed,
            row=args.row,
            final_only=args.final_only,
            skip_scored=False,
        )
        kept = thinned_frames([c.frames for c in every], args.min_frame_gap)
        cells = tuple(c for c in cells if c.frames in kept)
    cells = select_cells(list(cells), args.only_frames)
    if not cells:
        print(
            f"[INFO] nothing to score: {args.arm} seed{args.seed}+{args.repeats} "
            f"{args.row} ({args.tree})"
        )
        return 0
    groups = group_cells(cells, args.group)

    if args.group_child:
        if len(groups) != 1:
            raise RuntimeError(f"a group child planned {len(groups)} groups, not 1.")
        (group,) = groups
        # Each block's seed index names its row; a rerun plans only the missing
        # rows, so the evaluator cannot count it from the order of the blocks.
        _run_evaluator(
            passthrough
            + ["--grouped", "--num-envs", str(clips * len(group))]
            + ["--grouped-seed-indices"]
            + [str(cell.seed_index) for cell in group]
            + ["--checkpoints"]
            + [str(cell.checkpoint) for cell in group]
            + ["--output-jsons"]
            + [str(cell.output_json) for cell in group]
        )
        return 0

    for cell in cells:
        Path(cell.output_json).parent.mkdir(parents=True, exist_ok=True)
    grace = (
        GROUP_HANG_GRACE_SECONDS
        if args.hang_grace_seconds is None
        else args.hang_grace_seconds
    )
    print(
        f"[INFO] {args.arm} seed{args.seed}+{args.repeats} {args.row}: {len(cells)} cell(s) "
        f"in {len(groups)} group(s) of up to {args.group} checkpoint(s), one process each"
    )
    return run_groups(groups, sys.argv, clips=clips, hang_grace_seconds=grace)


def _run_evaluator(evaluator_args: list[str]) -> None:
    """Run the evaluator in this process with ``evaluator_args``."""
    site_packages = configure_cu130_bridge(
        required=os.environ.get("ISAACLAB_REQUIRE_CU130_RUNTIME") == "1"
    )
    if site_packages is not None:
        verify_cu130_torch(site_packages)
    sys.argv = [f"{EVALUATOR}.py", *evaluator_args]
    runpy.run_module(EVALUATOR, run_name="__main__")


def group_child_argv(argv: list[str], frames: list[int]) -> list[str]:
    """This script's argv for one group: its frames as ``--only-frames``, and
    ``--group-child``; a parent's own ``--only-frames`` is replaced."""
    split = argv.index("--") if "--" in argv else len(argv)
    own: list[str] = []
    skipping = False
    for arg in argv[1:split]:
        if arg == "--only-frames":
            skipping = True
            continue
        if skipping and not arg.startswith("--"):
            continue
        skipping = False
        if arg.startswith("--only-frames=") or arg == "--group-child":
            continue
        own.append(arg)
    return [
        argv[0],
        *own,
        "--group-child",
        "--only-frames",
        *(str(f) for f in frames),
        *argv[split:],
    ]


def run_groups(
    groups: list[tuple],
    argv: list[str],
    *,
    clips: int,
    hang_grace_seconds: float,
    attempts: int = 2,
) -> int:
    """Score each group in its own process; fail if any cell writes no row.

    A group counts as scored when every one of its rows is a complete JSON
    written after the group started, whatever the exit code (as ``run_isolated``
    counts a cell). A group with a missing row runs again; its child plans only
    the missing rows, unless ``--rescore``.
    """
    failed = []
    for index, group in enumerate(groups):
        frames = sorted({cell.frames for cell in group})
        outputs = [Path(cell.output_json) for cell in group]
        print(
            f"[INFO] group {index + 1}/{len(groups)}: {len(frames)} checkpoint(s), "
            f"{len(group)} block(s), {clips * len(group)} envs",
            flush=True,
        )
        for cell in group:
            print(f"[INFO]   f{cell.frames} -> {cell.output_json}", flush=True)
        cmd = _child_command(group_child_argv(argv, frames))
        started = time.time()
        for attempt in range(1, attempts + 1):
            code = call_with_hang_guard(
                cmd, outputs, hang_grace_seconds, fresh_after=started
            )
            missing = [path for path in outputs if not _fresh_json(path, started)]
            if not missing:
                if code != 0:
                    print(
                        f"[WARN] group {index + 1} exited {code} after writing all "
                        "its rows; scored",
                        flush=True,
                    )
                break
            print(
                f"[ERROR] group {index + 1} attempt {attempt}: {len(missing)} row(s) "
                f"missing (exit {code})",
                flush=True,
            )
        else:
            failed.extend(path.name for path in missing)
    if failed:
        print(f"[ERROR] {len(failed)} row(s) not written: {failed}")
        return 1
    return 0


def thinned_frames(frames: list[int], gap: int) -> set[int]:
    """The earliest milestone of each ``gap``-frame bucket, and the newest one,
    so a finished run's final checkpoint is always scored."""
    first: dict[int, int] = {}
    for f in sorted(frames):
        first.setdefault(f // gap, f)
    return set(first.values()) | ({max(frames)} if frames else set())


def select_cells(cells: list, only_frames: list[int] | None) -> list:
    """Keep the cells whose milestone is in ``only_frames`` (all if None)."""
    if only_frames is None:
        return list(cells)
    wanted = set(only_frames)
    return [cell for cell in cells if cell.frames in wanted]


def single_cell_argv(argv: list[str], frames: int) -> list[str]:
    """This script's argv for one cell: drop `--isolate-cells`, add `--only-frames`."""
    split = argv.index("--") if "--" in argv else len(argv)
    own = [arg for arg in argv[1:split] if arg != "--isolate-cells"]
    if "--only-frames" in own:
        raise ValueError("--isolate-cells already selects one cell per process.")
    return [argv[0], *own, "--only-frames", str(frames), *argv[split:]]


def _child_command(argv: list[str]) -> list[str]:
    """Restart the current implementation in an isolated Python process."""
    return [sys.executable, "-m", TREE_EVALUATOR, *argv[1:]]


def _complete_json(path: Path) -> bool:
    try:
        json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return True


def _fresh_json(path: Path, after: float | None) -> bool:
    """A complete JSON, written at or after ``after`` (any time if None)."""
    try:
        if after is not None and path.stat().st_mtime < after:
            return False
    except OSError:
        return False
    return _complete_json(path)


def call_with_hang_guard(
    cmd: list[str],
    output: Path | list[Path],
    grace_seconds: float,
    poll_seconds: float = 5.0,
    *,
    fresh_after: float | None = None,
) -> int:
    """Run ``cmd``; once every ``output`` parses as JSON, give it ``grace_seconds``
    to exit, then kill its process group and report success (the rows are in).

    ``fresh_after`` (a ``time.time()``) counts only rows written since then, so a
    row left by an earlier run does not start the clock. ``grace_seconds`` <= 0
    waits forever.

    ``cmd`` runs in its own session, so a launcher's ``timeout`` (SIGTERM to this
    process group), a closed terminal (SIGHUP) or Ctrl-C does not reach it. While
    it runs, SIGTERM and SIGHUP raise ``SystemExit`` here, and whatever ends this
    call (those, ``KeyboardInterrupt``, an error) kills the session and reaps it
    first; otherwise the evaluator would keep the GPU after the launcher's lock
    is free. A SIGKILL of this process is not covered.
    """
    outputs = [output] if isinstance(output, Path) else list(output)
    proc = None
    with _exit_on_hangup():
        try:
            proc = subprocess.Popen(cmd, start_new_session=True)
            written_at = None
            while True:
                try:
                    return proc.wait(timeout=poll_seconds)
                except subprocess.TimeoutExpired:
                    pass
                if grace_seconds <= 0:
                    continue
                if written_at is None and all(
                    _fresh_json(path, fresh_after) for path in outputs
                ):
                    written_at = time.monotonic()
                if (
                    written_at is not None
                    and time.monotonic() - written_at > grace_seconds
                ):
                    names = (
                        outputs[0].name if len(outputs) == 1 else f"{len(outputs)} rows"
                    )
                    print(
                        f"[WARN] evaluator still running {grace_seconds:.0f} s after writing "
                        f"{names}; killing it",
                        flush=True,
                    )
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                    return 0
        finally:
            if proc is not None and proc.poll() is None:
                print(
                    f"[WARN] stopping; killing evaluator session {proc.pid}", flush=True
                )
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()


def _raise_system_exit(signum, frame) -> None:
    raise SystemExit(128 + signum)


@contextlib.contextmanager
def _exit_on_hangup():
    """While active, SIGTERM and SIGHUP raise ``SystemExit(128 + signal)`` (the
    shell's exit status for a signal), so ``finally`` blocks run; the previous
    handlers come back on exit. Outside the main thread, where Python cannot set
    handlers, it does nothing."""
    previous = {}
    try:
        for signum in (signal.SIGTERM, signal.SIGHUP):
            previous[signum] = signal.signal(signum, _raise_system_exit)
    except ValueError:  # not the main thread
        pass
    try:
        yield
    finally:
        for signum, handler in previous.items():
            # None: a handler set outside Python; the default is the closest.
            signal.signal(signum, signal.SIG_DFL if handler is None else handler)


def run_isolated(
    cells: list,
    argv: list[str],
    attempts: int = 2,
    *,
    hang_grace_seconds: float = 0.0,
) -> int:
    """Score each cell in its own process; fail if any cell writes no row.

    The evaluator can exit 0 without writing its summary (a crash after Isaac
    Sim teardown, or inside the container wrapper), so a cell counts as scored
    only when its JSON exists and is non-empty. The reverse also happens: the
    summary is complete and ``SimulationApp.close()`` then hangs until Isaac
    Sim's own close timeout exits 1, or Kit crashes at
    teardown (exit 1/139), so a complete JSON written during the cell's own
    run counts as scored whatever the exit code. A cell with no row is retried
    (``attempts`` runs in total), because Kit sometimes crashes at startup.
    ``hang_grace_seconds`` > 0 kills a process that outlives its complete
    summary by that long (:func:`call_with_hang_guard`).
    """
    failed = []
    for cell in cells:
        print(f"[INFO]   f{cell.frames} -> {cell.output_json.name}", flush=True)
        cmd = _child_command(single_cell_argv(argv, cell.frames))
        output = Path(cell.output_json)
        started = time.time()
        for attempt in range(1, attempts + 1):
            if hang_grace_seconds > 0:
                code = call_with_hang_guard(cmd, output, hang_grace_seconds)
            else:
                code = subprocess.call(cmd)
            written = (
                output.is_file()
                and output.stat().st_mtime >= started
                and _complete_json(output)
            )
            if code != 0 and written:
                print(
                    f"[WARN] cell f{cell.frames} exited {code} after writing its "
                    "summary; scored",
                    flush=True,
                )
                break
            if code == 0 and output.is_file() and output.stat().st_size > 0:
                break
            print(
                f"[ERROR] cell f{cell.frames} attempt {attempt} wrote no row "
                f"(exit {code})",
                flush=True,
            )
        else:
            print(f"[ERROR] cell f{cell.frames} failed (exit {code})", flush=True)
            failed.append(cell.frames)
    if failed:
        print(f"[ERROR] {len(failed)}/{len(cells)} cell(s) failed: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
