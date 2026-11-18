"""Tracking-versus-training-hours rows from training-env evaluations.

Each row pairs one checkpoint (its ``cumulative_train_seconds`` and
``cumulative_env_frames``) with the ``reset_logs`` of its
``evaluate_checkpoint --record_reset_logs`` JSON, read as a training run would
log them and averaged over the settled half of the rollout:

* ``mpjpe_l_mm`` / ``mpjpe_g_mm``: ``Metrics/reference/mpjpe_{l,g}_mm``;
* ``clip_finished``: ``Episode_Termination/reference_finished``, the share of
  envs whose last episode reached the end of its clip;
* ``timeout``: ``Episode_Termination/time_out`` (10 s without a failure);
* ``success``: their sum, the share whose last episode ended without a
  tracking failure (``Episode_Termination/*`` are population shares over all
  envs, so the terms of a settled rollout sum to one);
* ``return_norm``: ``Metrics/reference/reset_return_norm_undiscounted``.

    python -m imitationlab.evaluation.trainenv_curve \
        --arm ppo <ckpt_dir> <eval_dir> --arm wm <ckpt_dir> <eval_dir> --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

METRIC_KEYS: dict[str, str] = {
    "mpjpe_l_mm": "Metrics/reference/mpjpe_l_mm",
    "mpjpe_g_mm": "Metrics/reference/mpjpe_g_mm",
    "clip_finished": "Episode_Termination/reference_finished",
    "timeout": "Episode_Termination/time_out",
    "return_norm": "Metrics/reference/reset_return_norm_undiscounted",
}
FAILURE_PREFIX = "Episode_Termination/"
SUCCESS_TERMS = ("reference_finished", "time_out")


def row_from_eval(
    eval_json: Mapping[str, Any], *, arm: str, frames: int, seconds: float
) -> dict[str, Any]:
    """One curve row from an evaluation summary and the checkpoint's clocks."""
    # Older summaries store these per-key values under episode_reward.
    keys = (
        (eval_json.get("reset_logs") or {}).get("keys")
        or eval_json.get("episode_reward")
        or {}
    )
    if not keys:
        msg = "evaluation JSON has no reset_logs; rerun with --record_reset_logs."
        raise ValueError(msg)
    row: dict[str, Any] = {
        "arm": arm,
        "frames": int(frames),
        "hours": float(seconds) / 3600.0,
    }
    for name, key in METRIC_KEYS.items():
        row[name] = float(keys[key]["settled_mean"]) if key in keys else float("nan")
    row["success"] = row["clip_finished"] + row["timeout"]
    for key, stats in keys.items():
        term = key[len(FAILURE_PREFIX) :] if key.startswith(FAILURE_PREFIX) else None
        if term and term not in SUCCESS_TERMS:
            row[f"fail_{term}"] = float(stats["settled_mean"])
    return row


def _checkpoint_clocks(path: Path) -> tuple[int, float]:
    import torch

    data = torch.load(path, map_location="cpu", weights_only=False)
    frames = int(data.get("cumulative_env_frames") or path.stem.rsplit("_", 1)[1])
    return frames, float(data["cumulative_train_seconds"])


def eval_json_for(ckpt: Path, eval_dir: Path) -> Path | None:
    """``<eval_dir>/<checkpoint stem>.json`` (the workstation launcher) or the
    ``score_tree`` name ``<arm>_seed<s>_<row>_f<frames>.json`` (cluster eval
    stages; one-segment runs, so the file's step is the cumulative frames).

    A WMBPTT light and full checkpoint at one step hold one policy, so the
    launcher's JSON of either names that step's point.
    """
    step = ckpt.stem.rsplit("_", 1)[1]
    for stem in (ckpt.stem, f"model_step_{step}", f"policy_step_{step}"):
        direct = eval_dir / f"{stem}.json"
        if direct.is_file():
            return direct
    matches = sorted(eval_dir.glob(f"*_f{step}.json"))
    if len(matches) > 1:
        msg = f"several evaluations of {ckpt.name} in {eval_dir}: {[m.name for m in matches]}"
        raise ValueError(msg)
    return matches[0] if matches else None


def curve_rows(arm: str, ckpt_dir: Path, eval_dir: Path) -> list[dict[str, Any]]:
    rows = []
    # A WMBPTT light run writes policy_step_<N>.pt; both names at one N are one point.
    checkpoints = {
        int(ckpt.stem.rsplit("_", 1)[1]): ckpt
        for pattern in ("policy_step_*.pt", "model_step_*.pt")
        for ckpt in ckpt_dir.glob(pattern)
    }
    for _, ckpt in sorted(checkpoints.items()):
        result = eval_json_for(ckpt, eval_dir)
        if result is None:
            continue
        frames, seconds = _checkpoint_clocks(ckpt)
        rows.append(
            row_from_eval(
                json.loads(result.read_text()), arm=arm, frames=frames, seconds=seconds
            )
        )
    return rows


def fixed_row(arm: str, eval_json: Mapping[str, Any]) -> dict[str, Any]:
    """A row for a policy with no training clock here (e.g. a released SONIC)."""
    row = row_from_eval(eval_json, arm=arm, frames=0, seconds=0.0)
    row["hours"] = float("nan")
    return row


def write_csv(rows: Iterable[Mapping[str, Any]], path: Path) -> None:
    rows = list(rows)
    cols: list[str] = []
    for r in rows:
        cols.extend(k for k in r if k not in cols)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--arm",
        nargs=3,
        action="append",
        metavar=("NAME", "CKPT_DIR", "EVAL_DIR"),
        required=True,
    )
    parser.add_argument(
        "--fixed", nargs=2, action="append", default=[], metavar=("NAME", "EVAL_JSON")
    )
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args(argv)
    rows = [
        fixed_row(name, json.loads(Path(ev).read_text())) for name, ev in args.fixed
    ]
    rows += [
        r for name, ck, ev in args.arm for r in curve_rows(name, Path(ck), Path(ev))
    ]
    for r in rows:
        print(
            f"{r['arm']:6s} {r['hours']:5.2f} h {r['frames'] / 1e6:8.0f}M  L={r['mpjpe_l_mm']:6.1f}  "
            f"G={r['mpjpe_g_mm']:6.1f}  finished={r['clip_finished']:.3f}  success={r['success']:.3f}  "
            f"return={r['return_norm']:.3f}"
        )
    if args.csv:
        write_csv(rows, args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
