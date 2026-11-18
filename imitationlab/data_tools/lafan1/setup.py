#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Download the Hugging Face LAFAN1 retargeting dataset into ./data.

By default this script downloads the Unitree G1 subset from:
https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset

It can also optionally convert the downloaded CSV files into NPZ motions and
generate a manifest JSON by delegating to ``scripts/data/prepare_lafan1_from_csv.py``.

Examples:
    pixi run python scripts/data/setup_lafan1_dataset.py

    pixi run -e isaaclab python scripts/data/setup_lafan1_dataset.py \
        --prepare-npz --headless

    pixi run -e isaaclab python scripts/data/setup_lafan1_dataset.py \
        --prepare-npz --headless --auto-trim-mode g1_shoulder_roll
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from imitationlab.data_tools.common.cli_args import hide_legacy_aliases


DATASET_REPO_ID = "lvhaidong/LAFAN1_Retargeting_Dataset"
DATASET_REPO_URL = f"https://huggingface.co/datasets/{DATASET_REPO_ID}"
ROOT_ALLOW_PATTERNS = ["README.md", "LICENSE", "dataset_infos.json", "meta_data/*"]
ROBOT_CHOICES = ("g1", "h1", "h1_2", "all")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_data_root() -> Path:
    return _repo_root() / "data" / "lafan1"


def _resolve_batch_converter_script(explicit_path: str | None) -> Path | None:
    if explicit_path is not None:
        converter_script = Path(explicit_path).expanduser().resolve()
        if not converter_script.is_file():
            raise FileNotFoundError(
                f"batch_converter_script not found: {converter_script}"
            )
        return converter_script

    candidate = Path(__file__).resolve().parents[1] / "common" / "isaac_batch.py"
    if candidate.is_file():
        return candidate.resolve()
    return None


def _robot_subdirs(robot_type: str) -> list[str]:
    if robot_type == "all":
        return ["g1", "h1", "h1_2"]
    return [robot_type]


def _download_patterns(
    robot_type: str, include_robot_description: bool, include_visualizer: bool
) -> list[str]:
    patterns = list(ROOT_ALLOW_PATTERNS)
    patterns.extend(f"{robot_subdir}/*" for robot_subdir in _robot_subdirs(robot_type))
    if include_robot_description:
        patterns.append("robot_description/*")
    if include_visualizer:
        patterns.append("rerun_visualize.py")
    return patterns


def _csv_dir_for_robot(raw_root: Path, robot_type: str) -> Path:
    if robot_type == "all":
        return raw_root
    return raw_root / robot_type


def _npz_dir_for_robot(data_root: Path, robot_type: str) -> Path:
    return data_root / "npz" / robot_type


def _manifest_path_for_robot(data_root: Path, robot_type: str) -> Path:
    return data_root / "manifests" / f"{robot_type}_lafan1_manifest.json"


def _count_csv_files(csv_dir: Path, recursive: bool) -> int:
    pattern = "**/*.csv" if recursive else "*.csv"
    return sum(1 for path in csv_dir.glob(pattern) if path.is_file())


def _format_command(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _run_prepare_pipeline(
    *,
    args: argparse.Namespace,
    csv_dir: Path,
    npz_dir: Path,
    manifest_path: Path,
    batch_converter_script: Path,
) -> None:
    from imitationlab.data_tools.lafan1.prepare import main as prepare

    prepare_args = [
        "--csv-dir",
        str(csv_dir),
        "--npz-dir",
        str(npz_dir),
        "--manifest-path",
        str(manifest_path),
        "--recursive",
        "--batch-converter-script",
        str(batch_converter_script),
        "--python",
        args.python,
        "--input-fps",
        str(args.input_fps),
        "--output-fps",
        str(args.output_fps),
        "--dataset-name",
        args.dataset_name,
        "--batch-size",
        str(args.batch_size),
    ]

    if args.motion_name_prefix:
        prepare_args.extend(["--motion-name-prefix", args.motion_name_prefix])
    if args.frame_range is not None:
        prepare_args.extend(
            ["--frame-range", str(args.frame_range[0]), str(args.frame_range[1])]
        )
    if args.auto_trim_mode != "none":
        prepare_args.extend(["--auto-trim-mode", args.auto_trim_mode])
        prepare_args.extend(
            ["--auto-trim-search-frames", str(args.auto_trim_search_frames)]
        )
        prepare_args.extend(
            [
                "--auto-trim-shoulder-roll-threshold",
                str(args.auto_trim_shoulder_roll_threshold),
            ]
        )
        prepare_args.extend(
            ["--auto-trim-hold-frames", str(args.auto_trim_hold_frames)]
        )
        prepare_args.extend(["--auto-trim-pre-roll", str(args.auto_trim_pre_roll)])
    if args.device is not None:
        prepare_args.extend(["--device", args.device])
    if args.overwrite_npz:
        prepare_args.append("--overwrite")
    else:
        prepare_args.append("--skip-existing")

    print(f"[INFO] Preparing NPZ motions and manifest from: {csv_dir}")
    command = [
        sys.executable,
        "-m",
        "imitationlab.data_tools.lafan1.prepare",
        *prepare_args,
    ]
    print(f"[CMD]  {_format_command(command)}")
    prepare(prepare_args)


def add_arguments(
    parser: argparse.ArgumentParser, *, prepare_default: bool = False
) -> None:
    parser.add_argument(
        "--data-root",
        "--data_root",
        type=str,
        default=str(_default_data_root()),
        help="Root folder for raw CSV files, converted NPZ files, and manifest output.",
    )
    parser.add_argument(
        "--robot-type",
        "--robot_type",
        type=str,
        choices=ROBOT_CHOICES,
        default="g1",
        help="Robot subset to download from the Hugging Face dataset repo.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Dataset repo revision to download from Hugging Face.",
    )
    parser.add_argument(
        "--prepare-npz",
        action="store_true",
        default=prepare_default,
        help="Convert CSV files to NPZ and write a manifest JSON.",
    )
    parser.add_argument(
        "--csv-dir", type=Path, help="Prepare an existing local CSV directory."
    )
    parser.add_argument(
        "--npz-dir", type=Path, help="Override the output NPZ directory."
    )
    parser.add_argument(
        "--manifest-path", type=Path, help="Override the output manifest path."
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use CSVs already in the data root.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64 if prepare_default else 0,
        help="Motions per parallel Isaac batch. Zero uses one batch for all motions.",
    )
    parser.add_argument(
        "--batch-converter-script",
        "--batch_converter_script",
        type=str,
        default=None,
        help="Optional path to the batched CSV-to-NPZ converter script.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used for the downstream CSV-to-NPZ conversion step.",
    )
    parser.add_argument(
        "--dataset-name",
        "--dataset_name",
        type=str,
        default="lafan1",
        help="Dataset name written into the generated manifest when --prepare-npz is used.",
    )
    parser.add_argument(
        "--motion-name-prefix",
        "--motion_name_prefix",
        type=str,
        default="",
        help="Optional prefix for generated motion names in the manifest.",
    )
    parser.add_argument(
        "--input-fps",
        "--input_fps",
        type=float,
        default=30.0,
        help="Input CSV FPS for the retargeted Hugging Face dataset during conversion.",
    )
    parser.add_argument(
        "--output-fps",
        "--output_fps",
        type=float,
        default=50.0,
        help="Output NPZ FPS for the conversion step.",
    )
    parser.add_argument(
        "--frame-range",
        "--frame_range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=None,
        help="Optional frame range passed through to prepare_lafan1_from_csv.py.",
    )
    parser.add_argument(
        "--auto-trim-mode",
        "--auto_trim_mode",
        type=str,
        choices=("none", "g1_shoulder_roll"),
        default="none",
        help=(
            "Optional per-motion trim heuristic forwarded to "
            "prepare_lafan1_from_csv.py."
        ),
    )
    parser.add_argument(
        "--auto-trim-search-frames",
        "--auto_trim_search_frames",
        type=int,
        default=800,
        help="Maximum number of CSV frames inspected by the auto-trim heuristic.",
    )
    parser.add_argument(
        "--auto-trim-shoulder-roll-threshold",
        "--auto_trim_shoulder_roll_threshold",
        type=float,
        default=1.0,
        help="Absolute shoulder-roll threshold for --auto-trim-mode g1_shoulder_roll.",
    )
    parser.add_argument(
        "--auto-trim-hold-frames",
        "--auto_trim_hold_frames",
        type=int,
        default=5,
        help="Require the auto-trim condition to hold for this many frames.",
    )
    parser.add_argument(
        "--auto-trim-pre-roll",
        "--auto_trim_pre_roll",
        type=int,
        default=5,
        help="Keep this many frames before the detected auto-trim start.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Deprecated no-op. Isaac Lab develop runs headless by default.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional sim device for conversion, for example cuda:0.",
    )
    parser.add_argument(
        "--include-robot-description",
        action="store_true",
        default=False,
        help="Also download robot_description assets from the Hugging Face repo.",
    )
    parser.add_argument(
        "--include-visualizer",
        action="store_true",
        default=False,
        help="Also download the dataset rerun_visualize.py helper from the Hugging Face repo.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        default=False,
        help="Force Hugging Face to re-download files even if the local metadata cache is up to date.",
    )
    parser.add_argument(
        "--overwrite-npz",
        action="store_true",
        default=False,
        help="Overwrite existing NPZ files during the optional conversion step.",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    hide_legacy_aliases(parser)
    run(parser.parse_args(argv))


def run(args: argparse.Namespace) -> None:
    if args.batch_size < 0:
        raise ValueError("--batch-size must be nonnegative.")
    data_root = Path(args.data_root).expanduser().resolve()
    raw_root = data_root / "raw"
    csv_dir = (
        args.csv_dir.expanduser().resolve()
        if args.csv_dir is not None
        else _csv_dir_for_robot(raw_root=raw_root, robot_type=args.robot_type)
    )
    npz_dir = (
        args.npz_dir.expanduser().resolve()
        if args.npz_dir is not None
        else _npz_dir_for_robot(data_root=data_root, robot_type=args.robot_type)
    )
    manifest_path = (
        args.manifest_path.expanduser().resolve()
        if args.manifest_path is not None
        else _manifest_path_for_robot(data_root=data_root, robot_type=args.robot_type)
    )

    if args.csv_dir is None and not args.skip_download:
        _download_csv(args, raw_root)
    if not csv_dir.is_dir():
        raise FileNotFoundError(f"Expected CSV directory does not exist: {csv_dir}")
    csv_count = _count_csv_files(csv_dir, recursive=True)
    if csv_count == 0:
        raise RuntimeError(f"No CSV files found under: {csv_dir}")
    print(f"[INFO] CSV files ready under {csv_dir}: {csv_count}")

    if not args.prepare_npz:
        print(
            "[INFO] Pass --prepare-npz to convert the CSV files and write a manifest."
        )
        return

    batch_converter_script = _resolve_batch_converter_script(
        args.batch_converter_script
    )
    if batch_converter_script is None:
        raise FileNotFoundError("Could not locate the batched CSV-to-NPZ converter.")
    _run_prepare_pipeline(
        args=args,
        csv_dir=csv_dir,
        npz_dir=npz_dir,
        manifest_path=manifest_path,
        batch_converter_script=batch_converter_script,
    )
    print(f"[INFO] Manifest: {manifest_path}")


def _download_csv(args: argparse.Namespace, raw_root: Path) -> None:
    from huggingface_hub import snapshot_download

    raw_root.mkdir(parents=True, exist_ok=True)
    allow_patterns = _download_patterns(
        robot_type=args.robot_type,
        include_robot_description=bool(args.include_robot_description),
        include_visualizer=bool(args.include_visualizer),
    )

    print(f"[INFO] Downloading LAFAN1 dataset from: {DATASET_REPO_URL}")
    print(f"[INFO] Revision: {args.revision}")
    print(f"[INFO] Local raw dataset root: {raw_root}")
    print(f"[INFO] Allow patterns: {allow_patterns}")

    snapshot_download(
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(raw_root),
        allow_patterns=allow_patterns,
        force_download=bool(args.force_download),
    )


if __name__ == "__main__":
    main()
