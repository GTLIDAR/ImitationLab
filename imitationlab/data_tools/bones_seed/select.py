#!/usr/bin/env python3
"""Select the full SONIC-filtered BONES-SEED G1 motion set from the metadata CSV.

Apply the public SONIC motion-name filter and write a deterministic selection
for the batch converter. Motions keep their mirror and neutral annotations.

Run from the repo root:

    pixi run python scripts/data/select_bones_seed_sonic.py \
        --metadata-csv data/bones_seed/raw/metadata/seed_metadata_v004.csv \
        --output data/bones_seed/selection/g1_bones_seed_sonic_selection.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from imitationlab.data_tools.bones_seed.sonic import (
    SONIC_RELEASE_FILTER_KEYWORDS,
    matched_keywords,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_METADATA_CSV = (
    REPO_ROOT / "data" / "bones_seed" / "raw" / "metadata" / "seed_metadata_v004.csv"
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--report-only",
        action="store_true",
        default=False,
        help="Print counts and the selection hash without writing the JSON.",
    )
    args = parser.parse_args(argv)
    if not args.report_only and args.output is None:
        parser.error("--output is required unless --report-only is set.")

    write_selection(args.metadata_csv, args.output if not args.report_only else None)


def write_selection(metadata_csv: Path, output: Path | None) -> None:
    metadata_csv = metadata_csv.expanduser().resolve()
    if not metadata_csv.is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")

    keywords = SONIC_RELEASE_FILTER_KEYWORDS
    with metadata_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    motions: list[dict[str, Any]] = []
    excluded = 0
    missing_path = 0
    for row in rows:
        filename = (row.get("filename") or "").strip()
        if not filename:
            continue
        if matched_keywords(filename, keywords):
            excluded += 1
            continue
        g1_path = (row.get("move_g1_path") or "").strip()
        if not g1_path:
            missing_path += 1
            continue
        duration = row.get("move_duration_frames")
        motions.append(
            {
                "filename": filename,
                "g1_path": g1_path,
                "category": (row.get("category") or "").strip() or None,
                "is_mirror": (row.get("is_mirror") or "").strip().lower() == "true",
                "is_neutral": (row.get("is_neutral") or "").strip() in ("1", "1.0"),
                "move_duration_frames": int(duration)
                if duration and str(duration).strip().isdigit()
                else None,
                "language_goal": (row.get("content_short_description") or "").strip()
                or None,
            }
        )

    # Deterministic order: by g1_path so the selection is stable across runs.
    motions.sort(key=lambda m: m["g1_path"])

    payload = {
        "dataset_name": "bones_seed_sonic_full",
        "source": "bones-studio/seed",
        "selection": "sonic_keyword_exclusion",
        "sonic_filter_source": (
            "NVlabs/GR00T-WholeBodyControl gear_sonic/data_process/"
            "filter_and_copy_bones_data.py default --filter-keywords"
        ),
        "sonic_filter_keywords": keywords,
        "metadata_csv": str(metadata_csv),
        "num_total_rows": len(rows),
        "num_excluded": excluded,
        "num_missing_g1_path": missing_path,
        "num_selected": len(motions),
        "motions": motions,
    }

    selection_bytes = json.dumps(payload["motions"], sort_keys=True).encode("utf-8")
    selection_hash = hashlib.sha256(selection_bytes).hexdigest()
    payload["selection_sha256"] = selection_hash

    print(f"[INFO] Metadata rows:     {len(rows)}")
    print(f"[INFO] SONIC-excluded:    {excluded}")
    print(f"[INFO] Missing g1_path:   {missing_path}")
    print(f"[INFO] Selected motions:  {len(motions)}")
    print(f"[INFO] Selection SHA-256: {selection_hash}")

    if output is None:
        return

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(f"[INFO] Wrote selection: {output}")


if __name__ == "__main__":
    sys.exit(main())
