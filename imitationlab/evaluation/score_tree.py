#!/usr/bin/env python3
"""Plan the cells of a budget-axis evaluation over one checkpoint tree.

A tracker tree holds one checkpoint per milestone, in one of two layouts.

The MIRROR layout, written by a campaign's `mirror.sh`::

    <tree>/tracker/f<frames>/models/model_step_<frames>.pt

The directory name carries the TRUE cumulative frame count, because the
per-segment step counter restarts on every chained resume.

The TRAINER layout, as it stands on the cluster::

    <tree>/tracker/<timestamp>_wandb-<run id>/models/model_step_<steps>.pt

Here the frame count has to come from the file name. Before 2026-08-16 the
step counter restarted on every chained resume, so a second segment's
`model_step_250085376` was not 250M cumulative frames. The learners now name every
checkpoint by cumulative frames and records the same count in it
(`cumulative_env_frames`). A tree spread over several run directories is
therefore read only when every checkpoint's recorded count equals its file
name, no frame count repeats, and no run directory holds a checkpoint past the
next one's first. A resume from an earlier checkpoint breaks the last rule: the
first run's later files are then an abandoned branch. WMBPTT light checkpoints
(`policy_step_<N>.pt`) make that common, because a segment that dies between
full files has light files past its resume point. Otherwise the tree is refused
rather than guessed (a silent guess would put a curve's points at the wrong
budgets, or score a policy from the abandoned branch). Mirror such a tree
first, where the rename is done with the resume record in hand.

Scoring every milestone of one arm gives the metric-against-frames curve;
scoring only the last one gives the row of record.

This module decides WHICH cells to run and where each row lands. It imports
nothing from Isaac Lab, so a launcher can plan the work -- on a login node, in
a test, or inside the evaluation process itself -- without starting a
simulation.

The output name is the convention every campaign launcher already writes::

    <output_root>/<arm>_seed<seed>_<row>_f<frames>.json
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path
import re

__all__ = [
    "AmbiguousTree",
    "Cell",
    "group_cells",
    "milestone_checkpoints",
    "milestone_frames",
    "plan_cells",
    "plan_repeat_cells",
    "repeat_output_root",
]

_FRAME_DIR = re.compile(r"^f(\d+)$")
# ``policy_step_<N>.pt`` is a WMBPTT light checkpoint: the deployed policy and
# its counters, which is all a scoring cell reads.
_STEP_FILE = re.compile(r"^(?:model|policy)_step_(\d+)\.pt$")


class AmbiguousTree(RuntimeError):
    """A multi-run-directory trainer tree whose frame counts are unverified."""


@dataclass(frozen=True)
class Cell:
    """One checkpoint scored into one summary file.

    ``seed_index`` is the evaluation index of a repeat cell (``plan_repeat_cells``),
    the index that names its row; ``plan_cells`` leaves it None.
    """

    frames: int
    checkpoint: Path
    output_json: Path
    seed_index: int | None = None


def milestone_checkpoints(tree: Path) -> dict[int, Path]:
    """Map cumulative frame count -> checkpoint file, for either layout.

    The mirror layout wins when it is present: its directory names are the
    authority on cumulative frames. Otherwise the trainer layout is read, and a
    tree whose checkpoints span more than one run directory raises
    `AmbiguousTree` unless the module's rules verify it: recorded counts equal
    to the names, no count twice, and run directories that do not interleave.
    """
    tracker = Path(tree) / "tracker"
    if not tracker.is_dir():
        return {}

    mirrored: dict[int, Path] = {}
    for entry in sorted(tracker.iterdir()):
        if not entry.is_dir():
            continue
        match = _FRAME_DIR.match(entry.name)
        if match is None:
            continue
        frames = int(match.group(1))
        for name in (f"model_step_{frames}.pt", f"policy_step_{frames}.pt"):
            checkpoint = entry / "models" / name
            if checkpoint.is_file():
                mirrored[frames] = checkpoint
                break
    if mirrored:
        return mirrored

    runs: dict[Path, dict[int, Path]] = {}
    for entry in sorted(tracker.iterdir()):
        models = entry / "models"
        if not entry.is_dir() or not models.is_dir():
            continue
        found: dict[int, Path] = {}
        for checkpoint in sorted(models.iterdir()):
            match = _STEP_FILE.match(checkpoint.name)
            if match is None or not checkpoint.is_file():
                continue
            found[int(match.group(1))] = checkpoint
        if found:
            runs[entry] = found
    if not runs:
        return {}
    if len(runs) == 1:
        return next(iter(runs.values()))
    names = ", ".join(sorted(path.name for path in runs))
    merged: dict[int, Path] = {}
    for found in runs.values():
        for frames, checkpoint in found.items():
            recorded = _recorded_frames(checkpoint)
            if recorded != frames or frames in merged:
                reason = (
                    f"{checkpoint.name} appears in two run directories"
                    if frames in merged
                    else f"{checkpoint} records cumulative_env_frames={recorded}"
                )
                raise AmbiguousTree(
                    f"{tree}: checkpoints span {len(runs)} run directories "
                    f"({names}) and {reason}, so the file names are not "
                    "verified cumulative frame counts. Mirror the tree into "
                    "the f<frames> layout first."
                )
            merged[frames] = checkpoint
    spans = sorted(
        (min(found), max(found), entry.name) for entry, found in runs.items()
    )
    for (_, last, earlier), (first, _, later) in pairwise(spans):
        if last > first:
            raise AmbiguousTree(
                f"{tree}: {earlier} holds a checkpoint at {last} frames, past the "
                f"first one of {later} at {first}. The later run resumed from an "
                f"earlier checkpoint, so {earlier}'s files past that point are an "
                "abandoned branch. Mirror the tree into the f<frames> layout first."
            )
    return merged


def _recorded_frames(checkpoint: Path) -> int | None:
    """The ``cumulative_env_frames`` a checkpoint records, if readable."""
    import torch  # noqa: PLC0415 - keep planning import-light

    try:
        # mmap reads the pickle, not the tensor payloads.
        data = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    except Exception:  # noqa: BLE001 - unreadable means unverified
        return None
    value = data.get("cumulative_env_frames") if isinstance(data, dict) else None
    return None if value is None else int(value)


def milestone_frames(tree: Path) -> tuple[int, ...]:
    """Return the cumulative frame counts in ``tree``, ascending."""
    return tuple(sorted(milestone_checkpoints(tree)))


def plan_cells(
    tree: Path,
    output_root: Path,
    *,
    arm: str,
    seed: int,
    row: str,
    final_only: bool = False,
    skip_scored: bool = True,
) -> tuple[Cell, ...]:
    """Return the ordered cells to score for one arm.

    ``final_only`` keeps just the largest frame count -- the row of record.
    ``skip_scored`` drops a cell whose summary file already exists and is
    non-empty, so an interrupted run resumes instead of repeating work.

    A milestone directory without its checkpoint file is skipped: a tree can
    hold a directory whose transfer never finished.

    Raises `AmbiguousTree` for a trainer-layout tree spread over several run
    directories; see `milestone_checkpoints`.
    """
    tree = Path(tree)
    output_root = Path(output_root)
    checkpoints = milestone_checkpoints(tree)
    if not checkpoints:
        return ()
    frames = tuple(sorted(checkpoints))
    if final_only:
        frames = (frames[-1],)

    cells: list[Cell] = []
    for frame in frames:
        checkpoint = checkpoints[frame]
        output_json = output_root / f"{arm}_seed{seed}_{row}_f{frame}.json"
        if skip_scored and output_json.is_file() and output_json.stat().st_size > 0:
            continue
        cells.append(Cell(frames=frame, checkpoint=checkpoint, output_json=output_json))
    return tuple(cells)


def repeat_output_root(template: Path, seed: int) -> Path:
    """The output root of the evaluation with seed index ``seed``.

    ``{seed}`` in ``template`` becomes the index; ``{seed_suffix}`` becomes
    nothing for index 0 and ``_s<index>`` otherwise, the layout
    ``replica``, ``replica_s1``, ... that ``score_local.sh`` writes
    (``replica${SEED:+_s$SEED}``) and ``time_to_target.py`` reads. A template
    without a placeholder is returned as it is.
    """
    text = str(template)
    suffix = "" if int(seed) == 0 else f"_s{int(seed)}"
    return Path(text.replace("{seed_suffix}", suffix).replace("{seed}", str(int(seed))))


def plan_repeat_cells(
    tree: Path,
    output_root: Path,
    *,
    arm: str,
    seed: int,
    repeats: int,
    row: str,
    final_only: bool = False,
    skip_scored: bool = True,
) -> tuple[Cell, ...]:
    """The cells of ``repeats`` evaluations of every milestone, by frames then repeat.

    Repeat r is evaluation index ``seed + r``: its row is
    ``repeat_output_root(output_root, seed + r) / <arm>_seed<seed + r>_<row>_f<frames>.json``,
    as ``plan_cells`` names it, and its ``seed_index`` is ``seed + r``. More than
    one repeat needs a placeholder in ``output_root``, or every repeat would land
    in one folder, where a reader of one evaluation per checkpoint would take the
    repeats for later checkpoints.
    """
    if int(repeats) < 1:
        raise ValueError("repeats must be at least 1.")
    template = str(output_root)
    if (
        int(repeats) > 1
        and "{seed}" not in template
        and "{seed_suffix}" not in template
    ):
        raise ValueError(
            "more than one repeat needs {seed} or {seed_suffix} in the output root, "
            f"so each repeat has its own folder: {output_root}"
        )
    cells: list[tuple[int, int, Cell]] = []
    for repeat in range(int(repeats)):
        index = int(seed) + repeat
        for cell in plan_cells(
            tree,
            repeat_output_root(output_root, index),
            arm=arm,
            seed=index,
            row=row,
            final_only=final_only,
            skip_scored=skip_scored,
        ):
            cells.append((cell.frames, repeat, replace(cell, seed_index=index)))
    return tuple(cell for _, _, cell in sorted(cells, key=lambda item: item[:2]))


def group_cells(
    cells: tuple[Cell, ...] | list[Cell], group: int
) -> list[tuple[Cell, ...]]:
    """Up to ``group`` checkpoints per group, each with all of its cells.

    ``cells`` come ordered by frames, then repeat (``plan_repeat_cells``), so a
    group's cells are checkpoint-major: the blocks of one checkpoint are
    consecutive, NDM Gym's ``(m * repeats + r)`` order.
    """
    if int(group) < 1:
        raise ValueError("group must be at least 1.")
    by_frames: dict[int, list[Cell]] = {}
    for cell in cells:
        by_frames.setdefault(cell.frames, []).append(cell)
    frames = list(by_frames)
    return [
        tuple(cell for f in frames[i : i + int(group)] for cell in by_frames[f])
        for i in range(0, len(frames), int(group))
    ]
