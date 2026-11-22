#!/usr/bin/env python3
"""Convert the full SONIC-filtered BONES-SEED G1 set to LAFAN1-style NPZ at scale.

The batched ``batch_csv_to_npz.py`` pads every motion in a batch to the
batch's longest trajectory, so throwing all ~130k motions (durations 69 -> 21617
source frames) into one Isaac session would need >100 GB of host RAM. This
orchestrator uses the same converter in bounded batches:

  1. extract  -- pull the selected CSVs out of ``g1.tar.gz`` in one streaming pass,
  2. normalize -- convert BONES euler/deg/cm CSV layout to this repo's converter
                  CSV layout (pos_m, quat_xyzw, joint_rad), in parallel,
  3. convert   -- length-bucket the motions under an ``envs x max_frames`` memory
                  budget and run ``batch_csv_to_npz.py`` once per bucket (PhysX),
  4. anchor    -- re-anchor each NPZ so frame-0 root XY == (0, 0), in parallel,
  5. audit     -- verify the 15-key set and joint/body name ordering, and
  6. manifest  -- write the motion manifest and language sidecar.

Every stage is resumable: re-running skips CSVs / NPZs that already exist.

Run from the repo root in the Isaac Lab env for the convert stage:

    pixi run -e isaaclab python scripts/data/convert_bones_seed_full.py \
        --selection data/bones_seed/selection/g1_bones_seed_sonic_selection.json \
        --archive data/bones_seed/raw/g1.tar.gz \
        --work-root data/bones_seed/converted \
        --device cuda:0

Stages 1-2 and 4-5 need no Isaac Sim; pass ``--stages extract,normalize`` in the
default env, then ``--stages convert`` in ``-e isaaclab``, then the rest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import shutil
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from filelock import FileLock

from imitationlab.data_tools.common.anchor_npz import anchor_one
from imitationlab.data_tools.common.manifest_io import sanitize_motion_name, write_json
from imitationlab.data_tools.bones_seed.csv import normalize_bones_csv
from imitationlab.data_tools.common.isaac_process import run_isaac

REPO_ROOT = Path(__file__).resolve().parents[3]
ALL_STAGES = ("extract", "normalize", "convert", "anchor", "audit", "manifest")
EXPECTED_NPZ_KEYS = {
    "fps",
    "qpos",
    "qvel",
    "root_pos",
    "root_quat",
    "root_lin_vel",
    "root_ang_vel",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "joint_names",
    "body_names",
}


# ------------------------------ selection ---------------------------------


def _load_selection(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    motions = payload.get("motions") if isinstance(payload, dict) else None
    if not isinstance(motions, list) or not motions:
        raise ValueError(f"Selection needs a nonempty 'motions' list: {path}")
    names = set()
    members = set()
    for motion in motions:
        filename = motion["filename"]
        member = motion["g1_path"].removeprefix("./")
        if not filename or Path(filename).name != filename or filename in {".", ".."}:
            raise ValueError(f"Invalid motion filename: {filename!r}")
        name = sanitize_motion_name(filename)
        if name in names or member in members or not member:
            raise ValueError(
                f"Duplicate motion name or invalid archive member: {filename!r}"
            )
        names.add(name)
        members.add(member)
        motion["g1_path"] = member
    return payload


def _staged_csv_path(csv_dir: Path, filename: str) -> Path:
    return csv_dir / f"{filename}.csv"


def _npz_path(npz_dir: Path, filename: str) -> Path:
    return npz_dir / f"{filename}.npz"


# ------------------------------ stage: extract ----------------------------


def stage_extract(
    selection: dict[str, Any], archive: Path, raw_dir: Path
) -> dict[str, int]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    wanted: dict[str, str] = {m["g1_path"]: m["filename"] for m in selection["motions"]}
    # Skip already-extracted CSVs.
    remaining = {
        g1_path: filename
        for g1_path, filename in wanted.items()
        if not (raw_dir / f"{filename}.csv").is_file()
    }
    print(
        f"[extract] wanted={len(wanted)} already_present={len(wanted) - len(remaining)} "
        f"to_extract={len(remaining)}"
    )
    if not remaining:
        return {"extracted": 0, "present": len(wanted)}
    if not archive.is_file():
        raise FileNotFoundError(f"BONES-SEED archive not found: {archive}")

    extracted = 0
    t0 = time.time()
    with tarfile.open(archive, mode="r|*") as tar:
        for member in tar:
            if not member.isfile():
                continue
            filename = remaining.pop(member.name.removeprefix("./"), None)
            if filename is None:
                continue
            target = raw_dir / f"{filename}.csv"
            source = tar.extractfile(member)
            if source is None:
                raise RuntimeError(f"Could not extract archive member: {member.name}")
            staged = target.with_suffix(".csv.tmp")
            with source, staged.open("wb") as handle:
                shutil.copyfileobj(source, handle)
            staged.replace(target)
            extracted += 1
            if extracted % 5000 == 0:
                rate = extracted / max(time.time() - t0, 1e-6)
                print(f"[extract] {extracted} extracted ({rate:.0f}/s)")
            if not remaining:
                break
    if remaining:
        raise RuntimeError(
            f"{len(remaining)} selected CSV(s) were missing from the archive, "
            f"e.g. {list(remaining.values())[:5]}"
        )
    print(f"[extract] done: extracted={extracted} in {time.time() - t0:.1f}s")
    return {"extracted": extracted, "present": len(wanted)}


# ------------------------------ stage: normalize --------------------------


def _normalize_one(args: tuple[str, str]) -> tuple[str, str | None]:
    source, target = map(Path, args)
    try:
        normalize_bones_csv(source, target)
    except (ValueError, OSError) as exc:
        return source.name, str(exc)
    return source.name, None


def stage_normalize(
    selection: dict[str, Any],
    raw_dir: Path,
    csv_dir: Path,
    workers: int,
) -> dict[str, int]:
    csv_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[str, str]] = []
    for m in selection["motions"]:
        filename = m["filename"]
        raw_path = raw_dir / f"{filename}.csv"
        staged_path = _staged_csv_path(csv_dir, filename)
        if staged_path.is_file():
            continue
        if not raw_path.is_file():
            raise FileNotFoundError(f"Missing extracted CSV: {raw_path}")
        jobs.append((str(raw_path), str(staged_path)))

    print(
        f"[normalize] total={len(selection['motions'])} to_normalize={len(jobs)} "
        f"workers={workers}"
    )
    if not jobs:
        return {"normalized": 0, "failed": 0}

    normalized = 0
    failed = 0
    failures: list[str] = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (name, error) in enumerate(
            pool.map(_normalize_one, jobs, chunksize=16), start=1
        ):
            if error is None:
                normalized += 1
            else:
                failed += 1
                failures.append(f"{name}: {error}")
            if i % 5000 == 0:
                rate = i / max(time.time() - t0, 1e-6)
                print(f"[normalize] {i}/{len(jobs)} ({rate:.0f}/s) failed={failed}")
    if failures:
        print(f"[normalize] {failed} failure(s), first 10:")
        for line in failures[:10]:
            print(f"  - {line}")
        raise RuntimeError(f"Normalization failed for {failed} motion(s).")
    print(
        f"[normalize] done: normalized={normalized} failed={failed} "
        f"in {time.time() - t0:.1f}s"
    )
    return {"normalized": normalized, "failed": failed}


# ------------------------------ stage: convert ----------------------------


def _count_frames(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def _plan_buckets(
    items: list[tuple[str, int]],
    max_envs: int,
    cell_budget: int,
) -> list[list[str]]:
    """Length-bucket motions so each batch obeys len(batch)*max_frames <= budget.

    ``items`` is (filename, output_frames), sorted ascending by output_frames.
    """
    buckets: list[list[str]] = []
    current: list[str] = []
    current_max = 0
    for filename, frames in items:
        if frames > cell_budget:
            raise ValueError(
                f"{filename} alone needs {frames} cells (budget {cell_budget})."
            )
        prospective_max = max(current_max, frames)
        if current and (
            len(current) + 1 > max_envs
            or (len(current) + 1) * prospective_max > cell_budget
        ):
            buckets.append(current)
            current = [filename]
            current_max = frames
        else:
            current.append(filename)
            current_max = prospective_max
    if current:
        buckets.append(current)
    return buckets


def stage_convert(
    selection: dict[str, Any],
    csv_dir: Path,
    npz_dir: Path,
    batch_dir: Path,
    input_fps: float,
    output_fps: float,
    max_envs: int,
    cell_budget: int,
    device: str,
    python_exe: str,
    workers: int,
) -> dict[str, Any]:
    npz_dir.mkdir(parents=True, exist_ok=True)
    batch_dir.mkdir(parents=True, exist_ok=True)
    batch_converter = Path(__file__).resolve().parents[1] / "common" / "isaac_batch.py"

    # One scandir per directory instead of 2*130k per-file stat calls (much faster
    # on a large-directory spinning disk, and makes resume near-instant).
    existing_npz = {
        entry.name[:-4] for entry in os.scandir(npz_dir) if entry.name.endswith(".npz")
    }
    existing_staged = {
        entry.name[:-4] for entry in os.scandir(csv_dir) if entry.name.endswith(".csv")
    }
    pending_names = []
    done = 0
    for m in selection["motions"]:
        filename = m["filename"]
        if filename in existing_npz:
            done += 1
            continue
        if filename not in existing_staged:
            raise FileNotFoundError(
                f"Missing staged CSV (run normalize first): "
                f"{_staged_csv_path(csv_dir, filename)}"
            )
        pending_names.append(filename)

    pending = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        paths = (_staged_csv_path(csv_dir, name) for name in pending_names)
        for name, frames in zip(
            pending_names, pool.map(_count_frames, paths, chunksize=64), strict=True
        ):
            if frames < 2:
                raise ValueError(f"{name} has fewer than two normalized frames.")
            pending.append(
                (name, max(2, int(np.ceil(frames * output_fps / input_fps)) + 1))
            )

    pending.sort(key=lambda x: x[1])
    buckets = _plan_buckets(pending, max_envs=max_envs, cell_budget=cell_budget)
    # Build the name -> frames map ONCE; rebuilding it inside the bucket loops is
    # O(n^2) over ~130k motions.
    frames_by_name = dict(pending)
    total_cells = (
        sum(len(b) * max(frames_by_name[f] for f in b) for b in buckets)
        if buckets
        else 0
    )
    print(
        f"[convert] total={len(selection['motions'])} already_done={done} "
        f"pending={len(pending)} buckets={len(buckets)} "
        f"est_cell_frames={total_cells:,}"
    )
    if not buckets:
        return {"buckets": 0, "converted": 0, "already_done": done}

    converted = 0
    t0 = time.time()
    for bi, bucket in enumerate(buckets):
        bmax = max(frames_by_name[f] for f in bucket)
        with TemporaryDirectory(prefix=".conversion-", dir=npz_dir) as staged:
            staged_dir = Path(staged)
            jobs = [
                {
                    "input_file": str(_staged_csv_path(csv_dir, f)),
                    "output_name": str(_npz_path(staged_dir, f)),
                }
                for f in bucket
            ]
            jobs_json = batch_dir / f"bucket_{bi:04d}.json"
            write_json(jobs_json, jobs)
            cmd = [
                python_exe,
                str(batch_converter),
                "--jobs-json",
                str(jobs_json),
                "--input-fps",
                str(input_fps),
                "--output-fps",
                str(output_fps),
                "--device",
                device,
            ]
            print(
                f"[convert] bucket {bi + 1}/{len(buckets)}: envs={len(bucket)} "
                f"max_frames={bmax} cells={len(bucket) * bmax:,}"
            )
            print(f"[CMD] {' '.join(shlex.quote(c) for c in cmd)}")
            run_isaac(cmd)
            missing = [f for f in bucket if not _npz_path(staged_dir, f).is_file()]
            if missing:
                raise RuntimeError(
                    f"bucket {bi} produced {len(missing)} missing NPZ(s), e.g. {missing[:5]}"
                )
            for name in bucket:
                _npz_path(staged_dir, name).replace(_npz_path(npz_dir, name))
        converted += len(bucket)
        rate = converted / max(time.time() - t0, 1e-6)
        print(
            f"[convert] bucket {bi + 1} ok | converted={converted}/{len(pending)} "
            f"({rate:.1f} motions/s)"
        )
    print(f"[convert] done: converted={converted} in {time.time() - t0:.1f}s")
    return {"buckets": len(buckets), "converted": converted, "already_done": done}


# ------------------------------ stage: anchor -----------------------------


def _anchor_job(args: tuple[Path, float]) -> dict[str, float]:
    path, tol = args
    return anchor_one(path, tol=tol)


def stage_anchor(
    selection: dict[str, Any], npz_dir: Path, tol: float, workers: int
) -> None:
    jobs = (
        (_npz_path(npz_dir, motion["filename"]), tol) for motion in selection["motions"]
    )
    with ProcessPoolExecutor(max_workers=workers) as pool:
        max_before = max(
            stats["before"] for stats in pool.map(_anchor_job, jobs, chunksize=16)
        )
    print(f"[anchor] max pre-anchor root XY = {max_before:.4f} m")


# ------------------------------ stage: audit ------------------------------


G1_ENV_CFG = (
    REPO_ROOT
    / "imitationlab"
    / "envs"
    / "isaaclab"
    / "tasks"
    / "manager_based"
    / "imitation"
    / "config"
    / "g1"
    / "common"
    / "constants.py"
)


def _canonical_dataset_body_names() -> list[str] | None:
    """Parse ``G1_29DOF_DATASET_BODY_NAMES`` from the env cfg without importing Isaac.

    This list is the authoritative reference body set the training env binds to
    by name (30 bodies; the fixed rubber-hand links are intentionally excluded).
    """
    import ast

    try:
        tree = ast.parse(G1_ENV_CFG.read_text(encoding="utf-8"))
    except OSError:
        return None
    for node in tree.body:
        # The constant is an annotated assignment: ``NAME: list[str] = [...]``.
        if isinstance(node, ast.AnnAssign):
            target = node.target
            value = node.value
        elif isinstance(node, ast.Assign):
            target = node.targets[0] if len(node.targets) == 1 else None
            value = node.value
        else:
            continue
        if (
            isinstance(target, ast.Name)
            and target.id == "G1_29DOF_DATASET_BODY_NAMES"
            and value is not None
        ):
            return [str(x) for x in ast.literal_eval(value)]
    return None


def stage_audit(
    selection: dict[str, Any],
    npz_dir: Path,
    sample: int,
) -> dict[str, Any]:
    canonical_bodies = _canonical_dataset_body_names()
    if canonical_bodies is None:
        raise RuntimeError("Could not read G1_29DOF_DATASET_BODY_NAMES.")

    names = [m["filename"] for m in selection["motions"]]
    if sample > 0 and sample < len(names):
        step = max(1, len(names) // sample)
        names = names[::step][:sample]
    print(f"[audit] auditing {len(names)} NPZ(s)")

    # Joint names have no separate literal; enforce self-consistency across the
    # tree (all NPZ must agree) plus a 29-count check.
    ref_joint_names: list[str] | None = None
    ref_body_names: list[str] | None = None
    problems: list[str] = []
    checked = 0
    for filename in names:
        path = _npz_path(npz_dir, filename)
        if not path.is_file():
            problems.append(f"{filename}: NPZ missing")
            continue
        with np.load(path, allow_pickle=False) as data:
            keys = set(data.files)
            if not EXPECTED_NPZ_KEYS.issubset(keys):
                problems.append(
                    f"{filename}: missing keys {sorted(EXPECTED_NPZ_KEYS - keys)}"
                )
                continue
            joint_names = [str(x) for x in data["joint_names"].tolist()]
            body_names = [str(x) for x in data["body_names"].tolist()]
            root_pos = data["root_pos"]
            if ref_joint_names is None:
                ref_joint_names = joint_names
            if ref_body_names is None:
                ref_body_names = body_names
            if len(joint_names) != 29:
                problems.append(f"{filename}: {len(joint_names)} joints (expected 29)")
            if joint_names != ref_joint_names:
                problems.append(f"{filename}: joint_names differ across tree")
            if body_names != ref_body_names:
                problems.append(f"{filename}: body_names differ across tree")
            if body_names != canonical_bodies:
                problems.append(
                    f"{filename}: body_names != canonical "
                    f"G1_29DOF_DATASET_BODY_NAMES (got {len(body_names)} bodies)"
                )
            else:
                pelvis = data["body_pos_w"][:, body_names.index("pelvis")]
                if not np.allclose(root_pos, pelvis, atol=2e-5, rtol=0):
                    problems.append(f"{filename}: root and pelvis translations differ")
            xy0 = np.abs(root_pos[0, :2])
            if not np.isfinite(xy0).all() or float(xy0.max()) > 1e-4:
                problems.append(
                    f"{filename}: root0 XY not anchored (max |xy|={float(xy0.max()):.2e})"
                )
        checked += 1
    print(f"[audit] checked={checked} problems={len(problems)}")
    for line in problems[:20]:
        print(f"  - {line}")
    return {
        "checked": checked,
        "problems": len(problems),
        "problem_examples": problems[:20],
        "canonical_body_match": canonical_bodies is not None
        and ref_body_names == canonical_bodies,
        "joint_names": ref_joint_names or [],
        "body_names": ref_body_names or [],
    }


# ------------------------------ stage: manifest ---------------------------


def stage_manifest(
    selection: dict[str, Any],
    npz_dir: Path,
    manifest_path: Path,
    language_path: Path,
    input_fps: float,
    output_fps: float,
) -> dict[str, int]:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    language_path.parent.mkdir(parents=True, exist_ok=True)

    trajectories: list[dict[str, Any]] = []
    language_motions: list[dict[str, Any]] = []
    missing = 0
    for m in selection["motions"]:
        filename = m["filename"]
        npz_file = _npz_path(npz_dir, filename)
        if not npz_file.is_file():
            missing += 1
            continue
        motion_name = sanitize_motion_name(filename)
        trajectories.append(
            {
                "name": motion_name,
                "path": os.path.relpath(npz_file, manifest_path.parent),
                "input_fps": float(output_fps),
            }
        )
        language_motions.append(
            {
                "name": motion_name,
                "bones_seed_filename": filename,
                "category": m.get("category"),
                "is_mirror": m.get("is_mirror"),
                "language_goal": m.get("language_goal") or "",
                "move_duration_frames": m.get("move_duration_frames"),
            }
        )

    if missing:
        raise FileNotFoundError(f"{missing} selected motions have no NPZ export.")
    dataset_name = selection.get("dataset_name", "bones_seed_sonic_full")
    manifest = {
        "dataset_name": dataset_name,
        "dataset": {"trajectories": {"lafan1_csv": trajectories}},
        "metadata": {
            "source_dataset": "bones-studio/seed",
            "selection": selection.get("selection"),
            "selection_sha256": selection.get("selection_sha256"),
            "num_motions": len(trajectories),
            "num_missing_npz": missing,
            "input_fps": float(input_fps),
            "output_fps": float(output_fps),
            "language_annotations_path": os.path.relpath(
                language_path, manifest_path.parent
            ),
            "paths_are_relative_to_manifest": True,
        },
    }

    language = {
        "dataset_name": dataset_name,
        "source": "bones-studio/seed",
        "manifest": os.path.relpath(manifest_path, language_path.parent),
        "input_fps": float(input_fps),
        "output_fps": float(output_fps),
        "motions": language_motions,
    }
    write_json(language_path, language)
    write_json(manifest_path, manifest)

    print(
        f"[manifest] wrote {len(trajectories)} trajectories "
        f"(missing_npz={missing}) -> {manifest_path}"
    )
    return {"trajectories": len(trajectories), "missing_npz": missing}


# ------------------------------ main --------------------------------------


def add_arguments(
    parser: argparse.ArgumentParser, *, selection_required: bool = True
) -> None:
    if selection_required:
        parser.add_argument("--selection", type=Path, required=True)
    else:
        source = parser.add_mutually_exclusive_group(required=True)
        source.add_argument(
            "--selection", type=Path, help="An existing SONIC selection JSON."
        )
        source.add_argument(
            "--metadata-csv",
            type=Path,
            help="Authorized local BONES-SEED metadata CSV.",
        )
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument(
        "--archive",
        type=Path,
        default=REPO_ROOT / "data" / "bones_seed" / "raw" / "g1.tar.gz",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="all",
        help=f"Comma list of stages to run, or 'all'. Choices: {','.join(ALL_STAGES)}",
    )
    parser.add_argument("--input-fps", type=float, default=120.0)
    parser.add_argument("--output-fps", type=float, default=50.0)
    parser.add_argument("--max-envs", type=int, default=512)
    parser.add_argument(
        "--cell-budget",
        type=int,
        default=500_000,
        help="Max envs*max_frames per Isaac batch (bounds host/GPU memory).",
    )
    parser.add_argument(
        "--normalize-workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Deprecated no-op. Isaac Lab develop runs headless by default.",
    )
    parser.add_argument("--anchor-tol", type=float, default=1e-5)
    parser.add_argument("--anchor-workers", type=int, default=4)
    parser.add_argument(
        "--audit-sample",
        type=int,
        default=2000,
        help="Audit this many NPZs (evenly sampled). 0 audits all.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Debug: only process the first N selected motions (0 = all).",
    )
    parser.add_argument("--python", type=str, default=sys.executable)


def run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if (
        min(
            args.max_envs, args.cell_budget, args.normalize_workers, args.anchor_workers
        )
        <= 0
    ):
        parser.error("worker counts and batch limits must be positive")
    if any(
        not math.isfinite(value) or value <= 0
        for value in (args.input_fps, args.output_fps, args.anchor_tol)
    ):
        parser.error("frame rates and anchor tolerance must be positive and finite")
    if min(args.limit, args.audit_sample) < 0:
        parser.error("motion and audit limits must be nonnegative")

    stages = (
        ALL_STAGES
        if args.stages == "all"
        else tuple(s.strip() for s in args.stages.split(",") if s.strip())
    )
    if not stages:
        parser.error("--stages must name at least one stage")
    for s in stages:
        if s not in ALL_STAGES:
            parser.error(f"unknown stage: {s}")

    work_root = args.work_root.expanduser().resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    with FileLock(work_root / ".conversion.lock", timeout=0):
        metadata_csv = getattr(args, "metadata_csv", None)
        if metadata_csv is not None:
            from imitationlab.data_tools.bones_seed.select import write_selection

            args.selection = work_root / "selection.json"
            write_selection(metadata_csv, args.selection)
        return _run_stages(args, stages, work_root)


def _run_stages(
    args: argparse.Namespace, stages: tuple[str, ...], work_root: Path
) -> int:
    raw_dir = work_root / "raw_csv"
    csv_dir = work_root / "staged_csv"
    npz_dir = work_root / "npz" / "g1"
    batch_dir = work_root / "batch_jobs"
    record_dir = work_root / "records"
    record_dir.mkdir(parents=True, exist_ok=True)

    selection = _load_selection(args.selection.expanduser().resolve())
    if args.limit > 0:
        selection = {**selection, "motions": selection["motions"][: args.limit]}
        print(f"[main] LIMIT active: processing {len(selection['motions'])} motions")

    identity_path = record_dir / "conversion_inputs.json"
    identity = {
        "selection": hashlib.sha256(
            json.dumps(selection["motions"], sort_keys=True).encode()
        ).hexdigest(),
        "input_fps": args.input_fps,
        "output_fps": args.output_fps,
    }
    if identity_path.is_file():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("Selection or frame rates changed. Use a new --work-root.")
    else:
        write_json(identity_path, identity)

    print(
        f"[main] motions={len(selection['motions'])} stages={stages} work_root={work_root}"
    )
    record_path = record_dir / "convert_record.json"
    record: dict[str, Any] = {
        **(
            json.loads(record_path.read_text(encoding="utf-8"))
            if record_path.is_file()
            else {}
        ),
        "selection": str(args.selection),
        "selection_sha256": selection.get("selection_sha256"),
        "num_motions": len(selection["motions"]),
        "stages": list(stages),
        "input_fps": args.input_fps,
        "output_fps": args.output_fps,
        "max_envs": args.max_envs,
        "cell_budget": args.cell_budget,
    }

    if "extract" in stages:
        record["extract"] = stage_extract(
            selection, args.archive.expanduser().resolve(), raw_dir
        )
    if "normalize" in stages:
        record["normalize"] = stage_normalize(
            selection, raw_dir, csv_dir, args.normalize_workers
        )
    if "convert" in stages:
        record["convert"] = stage_convert(
            selection,
            csv_dir,
            npz_dir,
            batch_dir,
            args.input_fps,
            args.output_fps,
            args.max_envs,
            args.cell_budget,
            args.device,
            args.python,
            args.normalize_workers,
        )
    if "anchor" in stages:
        stage_anchor(selection, npz_dir, args.anchor_tol, args.anchor_workers)
        record["anchor"] = {"tol": args.anchor_tol}
    if "audit" in stages:
        record["audit"] = stage_audit(selection, npz_dir, args.audit_sample)
        if record["audit"]["problems"]:
            write_json(record_path, record)
            return 1
    if "manifest" in stages:
        manifest_path = (
            work_root / "manifests" / "g1_bones_seed_sonic_full_manifest.json"
        )
        language_path = (
            work_root / "language" / "g1_bones_seed_sonic_full_language.json"
        )
        record["manifest"] = stage_manifest(
            selection,
            npz_dir,
            manifest_path,
            language_path,
            args.input_fps,
            args.output_fps,
        )

    write_json(record_path, record)
    print(f"[main] wrote record: {record_path}")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return run(parser.parse_args(argv), parser)


if __name__ == "__main__":
    sys.exit(main())
