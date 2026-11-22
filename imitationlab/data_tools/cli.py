"""Commands for preparing, packing, and validating motion data."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def _pack(args: argparse.Namespace) -> int:
    from imitationlab.data_tools.reference_arrays import build
    from imitationlab.data_tools.validate import validate_manifest

    if not args.validate_only:
        validate_manifest(
            args.manifest, npz_root=args.npz_root, for_pack=True, workers=args.workers
        )
    return build.run(args)


def _build_parser() -> argparse.ArgumentParser:
    from imitationlab.data_tools import validate
    from imitationlab.data_tools.bones_seed import prepare as bones
    from imitationlab.data_tools.lafan1 import setup as lafan
    from imitationlab.data_tools.reference_arrays import build

    parser = argparse.ArgumentParser(
        prog="imitationlab data",
        description="Prepare motion data, pack reference arrays, or validate an existing dataset.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Run a resumable dataset conversion.")
    datasets = prepare.add_subparsers(dest="dataset", required=True)
    lafan_parser = datasets.add_parser(
        "lafan1",
        help="Download or convert local LAFAN1 G1 CSVs in bounded Isaac batches.",
    )
    lafan.add_arguments(lafan_parser, prepare_default=True)
    lafan_parser.set_defaults(handler=lafan.run)
    bones_parser = datasets.add_parser(
        "bones_seed",
        help="Convert authorized local BONES-SEED data. Uses the existing staged pipeline.",
    )
    bones.add_arguments(bones_parser, selection_required=False)
    bones_parser.set_defaults(handler=lambda args: bones.run(args, bones_parser))

    pack = commands.add_parser(
        "pack", help="Pack an NPZ manifest into a reference bank."
    )
    build.add_arguments(pack, manifest_positional=True)
    pack.set_defaults(handler=_pack)

    validator = commands.add_parser(
        "validate", help="Validate a reference bank directory or a clip manifest."
    )
    validator.add_argument("target", type=Path)
    validator.add_argument(
        "--manifest", type=Path, help="Check a bank against this source manifest."
    )
    validator.add_argument("--persist-id", help="Expected packed-bank identity.")
    validator.add_argument(
        "--npz-root", type=Path, help="Override the manifest's clip directories."
    )
    validator.add_argument(
        "--body-names", nargs="+", help="Expected body order in the bank."
    )
    validator.add_argument(
        "--anchor-body", help="Expected baked anchor body in the bank."
    )
    validator.add_argument("--expected-motions", type=int)
    validator.add_argument("--expected-transitions", type=int)
    validator.add_argument(
        "--workers", type=int, default=4, help="Workers for checking manifest clips."
    )
    validator.add_argument(
        "--for-pack",
        action="store_true",
        help="Require a manifest of packable NPZ files.",
    )
    validator.add_argument(
        "--verify-source",
        action="store_true",
        help="Compare sampled bank rows against source NPZs.",
    )
    validator.add_argument(
        "--scan", action="store_true", help="Scan all bank rows for non-finite values."
    )
    validator.add_argument(
        "--fingerprint",
        action="store_true",
        help="Print a content fingerprint that ignores the bank's paths and name.",
    )
    validator.add_argument(
        "--expected-fingerprint",
        help="Require this content fingerprint (implies --fingerprint).",
    )
    validator.set_defaults(handler=validate.run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = args.handler(args)
        return 0 if result is None else result
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"imitationlab data: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
