"""Return and failed clips of a clean imitation-g1-v1 evaluation, in NDM Gym's units.

The readout plays each clip once from frame 0 without noise or randomization and
reports

    return = mean_c [ sum_t r_t / ((L_c - 1) * 0.1) ]

with a clip failed when its episode terminated. The acceptance target of
``wiki/code-opt-ndm-port.md`` is return >= 0.82 with no failed clip. The formula is
NDM's, but the stack is ours, so a value here does not compare with NDM's numbers.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

MAX_STEP_REWARD = 0.1
"""Per-step reward ceiling that normalizes a clip's return."""

TARGET_RETURN = 0.82
TARGET_FAILED_CLIPS = 0


@dataclass(frozen=True)
class ReplicaRow:
    """One checkpoint's readout."""

    replica_return: float
    failed_clips: int
    clips: int
    survived_fraction: float

    @property
    def meets_target(self) -> bool:
        return (
            self.replica_return >= TARGET_RETURN
            and self.failed_clips <= TARGET_FAILED_CLIPS
        )


def clip_lengths(bank_dir: str | Path) -> list[int]:
    """Frames of each clip, in trajectory-rank order, from a bank's manifest."""
    manifest = json.loads(
        (Path(bank_dir) / "reference_arrays_manifest.json").read_text()
    )
    info = manifest["traj_info"]
    return [
        int(end) - int(start)
        for start, end in zip(info["start_index"], info["end_index"], strict=True)
    ]


def replica_row(
    per_environment: Sequence[dict[str, Any]],
    lengths: Sequence[int],
    *,
    max_step_reward: float = MAX_STEP_REWARD,
) -> ReplicaRow:
    """Reduce an evaluation's ``per_environment`` records to one row."""
    if not per_environment:
        raise ValueError("the evaluation has no per-environment records")
    normalized = []
    survived = 0
    frames = 0
    for record in per_environment:
        length = int(lengths[int(record["trajectory_rank"])])
        if length < 2:
            raise ValueError(f"clip of {length} frames has no return to normalize")
        normalized.append(
            float(record["return_sum"]) / ((length - 1) * max_step_reward)
        )
        survived += int(record["survival_steps"])
        frames += length
    return ReplicaRow(
        replica_return=sum(normalized) / len(normalized),
        failed_clips=sum(bool(record["terminated"]) for record in per_environment),
        clips=len(per_environment),
        survived_fraction=survived / frames,
    )


def first_target(
    rows: Iterable[dict[str, Any]], *, hold: int = 1
) -> dict[str, Any] | None:
    """The first row, in the given order, that starts ``hold`` rows meeting the target.

    A row needs ``replica_return`` and ``failed_clips``. Rows in time order give the
    wall-clock time to success when each row carries its checkpoint's clock.
    """
    if hold < 1:
        raise ValueError("hold must be at least 1")
    ordered = list(rows)
    streak = 0
    for index, row in enumerate(ordered):
        good = (
            float(row["replica_return"]) >= TARGET_RETURN
            and float(row["failed_clips"]) <= TARGET_FAILED_CLIPS
        )
        streak = streak + 1 if good else 0
        if streak == hold:
            return ordered[index - hold + 1]
    return None


def mean_of_repeats(
    evaluations: Sequence[Sequence[dict[str, Any]]],
    lengths: Sequence[int],
    *,
    max_step_reward: float = MAX_STEP_REWARD,
) -> dict[str, Any]:
    """One checkpoint over several evaluations, NDM Gym's ``mean_of_repeats``.

    ``evaluations`` holds each evaluation's ``per_environment`` records. The row
    is the mean return and the mean failed count; ``failed_ranks`` lists the
    clips that failed in any evaluation. NDM Gym's target needs the mean failed
    count to be 0, so a single failure in any evaluation misses it.
    """
    if not evaluations:
        raise ValueError("no evaluations to average")
    rows = [
        replica_row(records, lengths, max_step_reward=max_step_reward)
        for records in evaluations
    ]
    failed = sorted(
        {
            int(r["trajectory_rank"])
            for records in evaluations
            for r in records
            if bool(r["terminated"])
        }
    )
    mean_return = sum(row.replica_return for row in rows) / len(rows)
    mean_failed = sum(row.failed_clips for row in rows) / len(rows)
    return {
        "evals": len(rows),
        "replica_return": mean_return,
        "failed_clips": mean_failed,
        "failed_ranks": failed,
        "meets_target": mean_return >= TARGET_RETURN
        and mean_failed <= TARGET_FAILED_CLIPS,
        "repeats": [asdict(row) for row in rows],
    }


def row_from_result(result_path: str | Path, bank_dir: str | Path) -> dict[str, Any]:
    """Read one evaluation JSON and return its row as a plain dictionary."""
    result = json.loads(Path(result_path).read_text())
    return asdict(replica_row(result["per_environment"], clip_lengths(bank_dir)))
