"""CPU-only checks for the native preparation, packing, and validation commands."""

from __future__ import annotations

import json

import pytest
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np

from imitationlab.data_tools.bones_seed.csv import BONES_G1_EXPECTED_COLUMNS
from imitationlab.data_tools.cli import main


def _make_manifest(tmp_path: Path) -> Path:
    for name in ("walk", "turn"):
        np.savez(
            tmp_path / f"{name}.npz",
            qpos=np.ones((4, 8), dtype=np.float32),
            qvel=np.zeros((4, 7), dtype=np.float32),
            joint_names=np.asarray(["joint"]),
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {"name": name, "path": f"{name}.npz", "input_fps": 50}
                for name in ("walk", "turn")
            ]
        )
    )
    return manifest


def test_help_loads_without_simulator():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from imitationlab.data_tools.cli import _build_parser;"
            "_build_parser().print_help(); "
            "assert not any(m.startswith(('isaaclab', 'isaacsim')) for m in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert all(command in result.stdout for command in ("prepare", "pack", "validate"))


def test_pack_validate_and_reject_an_inconsistent_bank(tmp_path, capsys):
    manifest = _make_manifest(tmp_path)
    bank = tmp_path / "bank"
    assert main(["validate", str(manifest), "--for-pack"]) == 0
    assert (
        main(
            [
                "pack",
                str(manifest),
                "--output-dir",
                str(bank),
                "--persist-id",
                "test@v1",
                "--workers",
                "2",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "validate",
                str(bank),
                "--manifest",
                str(manifest),
                "--verify-source",
                "--persist-id",
                "test@v1",
                "--expected-transitions",
                "6",
            ]
        )
        == 0
    )
    assert main(["validate", str(bank), "--persist-id", "wrong"]) == 1
    assert "expected persist_id" in capsys.readouterr().err

    sidecar_path = bank / "reference_arrays_manifest.json"
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["traj_info"]["start_index"][1] = 1
    sidecar_path.write_text(json.dumps(sidecar))
    assert main(["validate", str(bank)]) == 1
    assert "invalid trajectory span" in capsys.readouterr().err


def test_manifest_validation_checks_layout_and_frame_range(tmp_path, capsys):
    manifest = _make_manifest(tmp_path)
    entries = json.loads(manifest.read_text())
    entries[0]["frame_range"] = [1, 2]
    manifest.write_text(json.dumps(entries))
    assert main(["validate", str(manifest)]) == 0
    assert main(["validate", str(manifest), "--for-pack"]) == 1
    assert "bake frame_range" in capsys.readouterr().err

    np.savez(
        tmp_path / "turn.npz",
        qpos=np.zeros((4, 8)),
        qvel=np.zeros((3, 7)),
    )
    assert main(["validate", str(manifest)]) == 1
    assert "qvel has 3 frames" in capsys.readouterr().err


def test_prepare_lafan1_resumes_completed_batches(tmp_path, capsys):
    csv_dir, npz_dir = tmp_path / "csv", tmp_path / "npz"
    manifest, log = tmp_path / "manifest.json", tmp_path / "converter.log"
    csv_dir.mkdir()
    for name in ("first", "second"):
        (csv_dir / f"{name}.csv").write_text("0,0,1,0,0,0,1,0\n" * 4)
    failure_flag = tmp_path / "fail-second"
    failure_flag.touch()
    converter = tmp_path / "export.py"
    converter.write_text(
        f"""
import argparse
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--jobs-json", type=Path)
args, _ = parser.parse_known_args()
for job in json.loads(args.jobs_json.read_text()):
    motion = Path(job["input_file"]).stem
    with Path({str(log)!r}).open("a") as stream:
        stream.write(motion + "\\n")
    if motion == "second" and Path({str(failure_flag)!r}).exists():
        raise SystemExit(5)
    np.savez(job["output_name"], qpos=np.zeros((2, 8)), fps=np.asarray([50]))
"""
    )
    cmd = [
        "prepare",
        "lafan1",
        "--csv-dir",
        str(csv_dir),
        "--npz-dir",
        str(npz_dir),
        "--manifest-path",
        str(manifest),
        "--batch-size",
        "1",
        "--batch-converter-script",
        str(converter),
        "--frame-range",
        "2",
        "3",
    ]
    assert main(cmd) == 1
    assert (npz_dir / "first.npz").is_file()
    assert not manifest.exists()
    failure_flag.unlink()
    assert main(cmd) == 0
    assert log.read_text().splitlines() == ["first", "second", "second"]
    assert main(["validate", str(manifest), "--for-pack"]) == 0
    assert main([*cmd, "--output-fps", "60"]) == 1
    assert "conversion settings changed" in capsys.readouterr().err


def test_prepare_bones_from_local_metadata(tmp_path):
    source = tmp_path / "clip.csv"
    np.savetxt(
        source,
        np.zeros((2, len(BONES_G1_EXPECTED_COLUMNS))),
        delimiter=",",
        comments="",
        header=",".join(BONES_G1_EXPECTED_COLUMNS),
    )
    archive = tmp_path / "g1.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(source, arcname="g1/clip.csv")
    metadata = tmp_path / "metadata.csv"
    metadata.write_text("filename,move_g1_path\nclip,g1/clip.csv\n")
    work = tmp_path / "work"
    cmd = [
        "prepare",
        "bones_seed",
        "--metadata-csv",
        str(metadata),
        "--archive",
        str(archive),
        "--work-root",
        str(work),
        "--stages",
        "extract,normalize",
        "--normalize-workers",
        "2",
    ]
    assert main(cmd) == 0
    assert (work / "staged_csv" / "clip.csv").is_file()
    assert main(cmd) == 0
    record = json.loads((work / "records" / "convert_record.json").read_text())
    assert record["extract"]["extracted"] == 0
    assert record["normalize"]["normalized"] == 0


def test_options_are_kebab_case_only():
    from imitationlab.data_tools.cli import _build_parser

    parser = _build_parser()
    argv = ["prepare", "bones_seed", "--metadata-csv", "m.csv", "--work-root", "w"]
    args = parser.parse_args([*argv, "--input-fps", "60", "--anchor-tol", "0.1"])
    assert (args.metadata_csv, args.work_root) == (Path("m.csv"), Path("w"))
    assert (args.input_fps, args.anchor_tol) == (60.0, 0.1)

    bones = parser._subparsers._group_actions[0].choices["prepare"]
    help_text = bones._subparsers._group_actions[0].choices["bones_seed"].format_help()
    assert "--input-fps" in help_text and "--metadata-csv" in help_text
    assert "--input_fps" not in help_text and "--work_root" not in help_text
    with pytest.raises(SystemExit):
        parser.parse_args([*argv, "--input_fps", "60"])


def test_fingerprint_ignores_where_and_under_which_name_a_bank_was_packed(
    tmp_path, capsys
):
    banks = []
    for host in ("host_a", "host_b"):
        root = tmp_path / host
        root.mkdir()
        manifest = _make_manifest(root)
        # Absolute clip paths make the manifest bytes, and so its SHA-256,
        # depend on the directory, as real prepared manifests do.
        entries = json.loads(manifest.read_text())
        for entry in entries:
            entry["path"] = str(root / entry["path"])
        manifest.write_text(json.dumps(entries))
        bank = root / "bank"
        args = ["pack", str(manifest), "--output-dir", str(bank)]
        assert main([*args, "--persist-id", f"{host}@v1", "--workers", "1"]) == 0
        banks.append(bank)
    capsys.readouterr()

    fingerprints = []
    for bank in banks:
        assert main(["validate", str(bank), "--fingerprint"]) == 0
        line = capsys.readouterr().out.strip().splitlines()[-1]
        assert line.startswith("Content fingerprint: ")
        fingerprints.append(line.split()[-1])
    assert fingerprints[0] == fingerprints[1]
    expected = ["--expected-fingerprint", fingerprints[0]]
    assert main(["validate", str(banks[1]), *expected]) == 0

    qpos = np.memmap(banks[1] / "qpos.memmap", dtype=np.float32, mode="r+")
    qpos[0] += 1
    qpos.flush()
    del qpos
    assert main(["validate", str(banks[1]), *expected]) == 1
    assert "differs from the expected fingerprint" in capsys.readouterr().err
