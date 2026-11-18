"""Inspect training manifests and packed banks without loading a simulator."""

from __future__ import annotations

import csv
import math
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from imitationlab.data_tools.reference_arrays import build


_BODY_WIDTHS = {
    "body_pos_w": 3,
    "body_quat_w": 4,
    "body_lin_vel_w": 3,
    "body_ang_vel_w": 3,
}


def _npz_frames(path: Path, *, for_pack: bool) -> int:
    with zipfile.ZipFile(path) as archive:
        arrays = {}
        for key in ("qpos", "qvel", "joint_pos", "joint_vel", *_BODY_WIDTHS):
            try:
                with archive.open(f"{key}.npy") as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        header = np.lib.format.read_array_header_1_0
                    elif version == (2, 0):
                        header = np.lib.format.read_array_header_2_0
                    else:
                        raise ValueError(
                            f"{key} has an unsupported NumPy header {version}"
                        )
                    shape, _, dtype = header(stream)
            except KeyError:
                continue
            if dtype.kind not in "fiu" or len(shape) < 2 or min(shape) <= 0:
                raise ValueError(
                    f"{key} has invalid numeric shape or dtype: {shape}, {dtype}"
                )
            arrays[key] = shape

    if "qpos" not in arrays and (for_pack or "joint_pos" not in arrays):
        raise ValueError(
            "NPZ must contain qpos" if for_pack else "NPZ lacks qpos and joint_pos"
        )
    frames = arrays["qpos" if "qpos" in arrays else "joint_pos"][0]
    for key, shape in arrays.items():
        if shape[0] != frames:
            raise ValueError(f"{key} has {shape[0]} frames, expected {frames}")
        if key in _BODY_WIDTHS:
            if len(shape) != 3 or shape[-1] != _BODY_WIDTHS[key]:
                raise ValueError(f"{key} has invalid body dimensions: {shape}")
        elif len(shape) != 2:
            raise ValueError(f"{key} must be a frame-by-channel matrix")
    bodies = {arrays[key][1] for key in _BODY_WIDTHS if key in arrays}
    if len(bodies) > 1:
        raise ValueError("body arrays disagree on body count")
    if "qpos" in arrays:
        joints = arrays["qpos"][1] - 7
        if joints < 0 or ("qvel" in arrays and arrays["qvel"][1] != joints + 6):
            raise ValueError("qpos and qvel have inconsistent joint dimensions")
        if "joint_pos" in arrays and arrays["joint_pos"][1] != joints:
            raise ValueError("qpos and joint_pos have inconsistent joint dimensions")
    if "joint_pos" in arrays and "joint_vel" in arrays:
        if arrays["joint_pos"] != arrays["joint_vel"]:
            raise ValueError("joint position and velocity arrays disagree on shape")
    return frames


def _csv_frames(path: Path) -> int:
    width = None
    frames = 0
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.reader(stream):
            if not row:
                continue
            if width is None:
                width = len(row)
            if width < 8 or len(row) != width:
                raise ValueError(
                    "CSV frames must have consistent root and joint columns"
                )
            if any(not math.isfinite(float(value)) for value in row):
                raise ValueError("CSV contains a non-finite motion value")
            frames += 1
    return frames


def _inspect_clip(
    item: tuple[str, tuple[Path, dict[str, Any]]], *, for_pack: bool
) -> int:
    name, (path, entry) = item
    try:
        fps = float(entry.get("input_fps", 0.0))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("input_fps must be positive and finite")
        if path.suffix.lower() == ".npz":
            frames = _npz_frames(path, for_pack=for_pack)
        elif path.suffix.lower() == ".csv" and not for_pack:
            frames = _csv_frames(path)
        else:
            raise ValueError("expected NPZ" if for_pack else "expected NPZ or CSV")
        span = entry.get("frame_range")
        if span is not None:
            if not (
                isinstance(span, list)
                and len(span) == 2
                and all(type(value) is int for value in span)
                and 1 <= span[0] < span[1] <= frames
            ):
                raise ValueError(
                    f"invalid inclusive frame_range {span!r} for {frames} frames"
                )
            if for_pack and span != [1, frames]:
                raise ValueError("bake frame_range trims into NPZ files before packing")
            frames = span[1] - span[0] + 1
        if frames < 2:
            raise ValueError("at least two motion frames are required")
        return frames - 1
    except (OSError, ValueError, TypeError, zipfile.BadZipFile) as error:
        raise ValueError(f"Motion {name!r} at {path}: {error}") from error


def validate_manifest(
    path: Path,
    *,
    npz_root: Path | None = None,
    for_pack: bool = False,
    workers: int = 4,
) -> tuple[int, int]:
    """Resolve all entries and validate their frame rate, shape, and usable span."""
    if workers < 1:
        raise ValueError("--workers must be positive")
    clips = build.resolve_manifest_clips(path, npz_root=npz_root)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = sum(
            pool.map(lambda item: _inspect_clip(item, for_pack=for_pack), clips.items())
        )
    return len(clips), rows


def validate_bank(
    path: Path,
    *,
    manifest: Path | None = None,
    npz_root: Path | None = None,
    persist_id: str | None = None,
    body_names: list[str] | None = None,
    anchor_body: str | None = None,
    expected_motions: int | None = None,
    expected_transitions: int | None = None,
    verify_source: bool = False,
    scan: bool = False,
    workers: int = 4,
) -> tuple[str, int, int]:
    errors = build.validation_errors(
        path,
        persist_id=persist_id,
        body_names=body_names,
        anchor_body=anchor_body,
        expected_motions=expected_motions,
        expected_transitions=expected_transitions,
    )
    if errors:
        raise ValueError("\n".join(errors))
    arrays, sidecar = build.open_reference_arrays(path)
    if manifest is not None:
        if build.sha256_file(manifest) != sidecar["key"]["manifest_sha256"]:
            raise ValueError(
                f"{manifest} SHA-256 does not match the bank's source manifest"
            )
        validate_manifest(manifest, npz_root=npz_root, for_pack=True, workers=workers)
    if verify_source:
        if manifest is None:
            raise ValueError("--verify-source requires --manifest")
        build.verify_against_source(path, manifest=manifest, npz_root=npz_root)
    for key, array in arrays.items():
        if scan:
            rows = (
                array[start : start + 16384] for start in range(0, len(array), 16384)
            )
        else:
            rows = (array[[0, len(array) // 2, len(array) - 1]],)
        if any(not np.isfinite(chunk).all() for chunk in rows):
            raise ValueError(f"{key} contains non-finite values")
    info = sidecar["traj_info"]
    return (
        sidecar["key"]["source"]["persist_id"],
        len(info["ordered_traj_list"]),
        info["written"],
    )


# Sidecar fields that describe where and under which name a bank was built,
# not what it contains. The manifest SHA-256 depends on the absolute clip
# paths the manifest lists, so it differs for the same motions on another host.
_LOCATION_KEYS = ("manifest", "manifest_sha256", "source")


def content_fingerprint(path: Path, *, chunk_rows: int = 16384) -> str:
    """SHA-256 of a bank's layout and array bytes, independent of its location.

    Two banks with the same trajectories, names, layout and float values share
    a fingerprint even when they were packed from different directories or
    under different persist IDs.
    """
    import hashlib
    import json

    arrays, sidecar = build.open_reference_arrays(path)
    key = {k: v for k, v in sidecar["key"].items() if k not in _LOCATION_KEYS}
    info = sidecar["traj_info"]
    layout = {
        "key": key,
        "traj_info": {
            name: info[name]
            for name in ("written", "start_index", "end_index", "ordered_traj_list")
        },
    }
    digest = hashlib.sha256(json.dumps(layout, sort_keys=True).encode())
    for name in sorted(arrays):
        array = arrays[name]
        digest.update(name.encode())
        for start in range(0, len(array), chunk_rows):
            digest.update(np.ascontiguousarray(array[start : start + chunk_rows]).data)
    return digest.hexdigest()


def run(args) -> int:
    target = args.target.expanduser().resolve()
    if target.name == build.SIDECAR_NAME and target.is_file():
        target = target.parent
    if target.is_dir():
        identity, motions, rows = validate_bank(
            target,
            manifest=args.manifest,
            npz_root=args.npz_root,
            persist_id=args.persist_id,
            body_names=args.body_names,
            anchor_body=args.anchor_body,
            expected_motions=args.expected_motions,
            expected_transitions=args.expected_transitions,
            verify_source=args.verify_source,
            scan=args.scan,
            workers=args.workers,
        )
        print(f"Valid bank {identity}: {motions:,} trajectories, {rows:,} transitions")
        if args.fingerprint or args.expected_fingerprint:
            fingerprint = content_fingerprint(target)
            print(f"Content fingerprint: {fingerprint}")
            if args.expected_fingerprint and fingerprint != args.expected_fingerprint:
                raise ValueError(
                    "Bank content differs from the expected fingerprint "
                    f"{args.expected_fingerprint}"
                )
    else:
        if (
            args.manifest
            or args.persist_id
            or args.body_names
            or args.anchor_body
            or args.verify_source
            or args.scan
            or args.fingerprint
            or args.expected_fingerprint
        ):
            raise ValueError("Bank-specific options require a packed bank directory")
        motions, rows = validate_manifest(
            target, npz_root=args.npz_root, for_pack=args.for_pack, workers=args.workers
        )
        if args.expected_motions is not None and motions != args.expected_motions:
            raise ValueError(
                f"Manifest has {motions} motions, expected {args.expected_motions}"
            )
        if args.expected_transitions is not None and rows != args.expected_transitions:
            raise ValueError(
                f"Manifest has {rows} source transitions, expected {args.expected_transitions}"
            )
        print(
            f"Valid manifest {target}: {motions:,} motions, {rows:,} source transitions"
        )
    return 0
