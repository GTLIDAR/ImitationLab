"""CPU checks for manifest merging and the shared manifest helpers."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from imitationlab.data_tools.common.manifest_io import (
    motion_manifest_schema,
    sanitize_motion_name,
    write_json,
)
from imitationlab.data_tools.merge_manifests import build_parser, main


def _manifest(root: Path, name: str, motions: list[str], fps: float = 50.0) -> Path:
    npz_dir = root / name / "npz"
    npz_dir.mkdir(parents=True)
    entries = []
    for motion in motions:
        (npz_dir / f"{motion}.npz").write_bytes(b"")
        entries.append({"name": motion, "path": f"npz/{motion}.npz", "input_fps": fps})
    path = root / name / "manifest.json"
    write_json(
        path,
        {"dataset_name": name, "dataset": {"trajectories": {"lafan1_csv": entries}}},
    )
    return path


def _sources(tmp_path: Path) -> list[str]:
    lafan = _manifest(tmp_path, "lafan1", ["walk1_subject1", "dance-2"])
    bones = _manifest(tmp_path, "bones", ["pick_up_001"])
    language = tmp_path / "bones" / "language.json"
    write_json(
        language,
        {"motions": [{"name": "pick_up_001", "language_goal": "pick up a box"}]},
    )
    return [
        "--lafan1-manifest", str(lafan),
        "--lafan1-language", "",
        "--bones-manifest", str(bones),
        "--bones-language", str(language),
        "--exclude-dance102",
        "--output-manifest", str(tmp_path / "out" / "manifest.json"),
        "--output-language", str(tmp_path / "out" / "language.json"),
    ]  # fmt: skip


def test_schema_loads_without_isaac_lab() -> None:
    code = (
        "import sys\n"
        "from imitationlab.data_tools.common.manifest_io import motion_manifest_schema\n"
        "schema = motion_manifest_schema()\n"
        "assert schema.CLIP_LOADER_KEY == 'lafan1_csv'\n"
        "assert schema is motion_manifest_schema()\n"
        "assert not any(m.startswith(('isaaclab', 'isaacsim')) for m in sys.modules)\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_sanitize_motion_name_keeps_existing_names() -> None:
    assert (
        sanitize_motion_name(Path("dance/dance1-subject2")) == "dance_dance1_subject2"
    )
    assert sanitize_motion_name("Walk Fast!.v2") == "Walk_Fast_v2"
    assert sanitize_motion_name("--") == "motion"


def test_write_json_creates_parents_and_leaves_no_staging_file(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b.json"
    write_json(target, {"x": 1})
    assert json.loads(target.read_text(encoding="utf-8")) == {"x": 1}
    assert sorted(p.name for p in target.parent.iterdir()) == ["b.json"]


def test_merge_prefixes_names_and_writes_language(tmp_path: Path) -> None:
    main(_sources(tmp_path))

    out = tmp_path / "out"
    manifest = motion_manifest_schema().read_manifest(out / "manifest.json")
    entries = manifest["dataset"]["trajectories"]["lafan1_csv"]
    assert [entry["name"] for entry in entries] == [
        "lafan1__walk1_subject1",
        "lafan1__dance_2",
        "bones_seed__pick_up_001",
    ]
    assert entries[0]["path"] == "../lafan1/npz/walk1_subject1.npz"
    assert manifest["metadata"]["family"] == "unified"
    assert manifest["metadata"]["control_freq"] == 50.0

    language = json.loads((out / "language.json").read_text(encoding="utf-8"))
    goals = {motion["name"]: motion["language_goal"] for motion in language["motions"]}
    assert goals["bones_seed__pick_up_001"] == "pick up a box"
    # Without a sidecar entry the goal falls back to the humanized name.
    assert goals["lafan1__walk1_subject1"] == "walk"


def test_merge_refuses_name_collisions_and_missing_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = _sources(tmp_path)
    lafan = str(tmp_path / "lafan1" / "manifest.json")
    with pytest.raises(SystemExit):
        main([*argv, "--bones-manifest", lafan, "--bones-prefix", "lafan1__"])
    assert (
        "Duplicate merged motion name: lafan1__walk1_subject1"
        in capsys.readouterr().err
    )
    assert not (tmp_path / "out").exists()

    (tmp_path / "lafan1" / "npz" / "dance-2.npz").unlink()
    with pytest.raises(SystemExit):
        main(argv)
    assert "dance-2.npz" in capsys.readouterr().err
    main([*argv, "--allow-missing-motion-files"])
    assert (tmp_path / "out" / "manifest.json").is_file()


def test_merge_cli_accepts_and_hides_legacy_spellings() -> None:
    parser = build_parser()
    args = parser.parse_args(["--output_manifest", "m.json", "--no_prefix_names"])
    assert args.output_manifest == "m.json" and args.no_prefix_names
    help_text = parser.format_help()
    assert "--output-manifest" in help_text
    assert "--output_manifest" not in help_text
