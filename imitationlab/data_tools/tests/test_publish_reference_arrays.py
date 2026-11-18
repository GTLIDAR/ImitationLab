"""Tests for publishing and retrieving reference arrays.

The gate is the whole value here: 49.4 GB should not move over a network, in
either direction, unless the directory would also satisfy the environment at
load time. No test touches the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from imitationlab.data_tools.reference_arrays.build import (
    SIDECAR_NAME,
    build_reference_arrays,
)
from imitationlab.data_tools.reference_arrays.publish import (
    dataset_card,
    fetch,
    main,
    push,
)


BODY_NAMES = ["pelvis", "torso_link"]
ANCHOR = "pelvis"
PERSIST_ID = "synthetic@cafe"
LENGTHS = {"walk": 8, "wave": 5}


@pytest.fixture
def built(tmp_path: Path) -> Path:
    return _build(tmp_path)


def _build(tmp_path: Path, *, dataset_name: str = "lafan1") -> Path:
    npz_dir = tmp_path / "npz"
    npz_dir.mkdir()
    entries = []
    for index, (name, frames) in enumerate(LENGTHS.items()):
        rng = np.random.default_rng(index)
        np.savez(
            npz_dir / f"{name}.npz",
            qpos=rng.standard_normal((frames, 10), dtype=np.float32),
            qvel=rng.standard_normal((frames, 9), dtype=np.float32),
            body_pos_w=rng.standard_normal((frames, 2, 3), dtype=np.float32),
            body_quat_w=rng.standard_normal((frames, 2, 4), dtype=np.float32),
            joint_names=np.asarray([f"j{i}" for i in range(3)]),
            body_names=np.asarray(BODY_NAMES),
        )
        entries.append({"name": name, "path": f"npz/{name}.npz", "input_fps": 50.0})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"dataset": {"trajectories": {"lafan1_csv": entries}}}),
        encoding="utf-8",
    )
    output_dir = tmp_path / "arrays"
    build_reference_arrays(
        manifest=manifest,
        output_dir=output_dir,
        persist_id=PERSIST_ID,
        body_names=list(BODY_NAMES),
        anchor_body=ANCHOR,
        workers=1,
        dataset_name=dataset_name,
    )
    return output_dir


def test_dry_run_reports_the_payload_without_uploading(built: Path) -> None:
    result = push(
        source_dir=built,
        repo_id="org/whatever",
        persist_id=PERSIST_ID,
        body_names=list(BODY_NAMES),
        anchor_body=ANCHOR,
        expected_motions=len(LENGTHS),
        dry_run=True,
    )

    assert set(result["files"]) == {
        "qpos.memmap",
        "qvel.memmap",
        "body_pos_w.memmap",
        "body_quat_w.memmap",
        "anchor_pos_w.memmap",
        "anchor_quat_w.memmap",
        SIDECAR_NAME,
    }
    assert result["bytes"] > 0
    # A dry run must not leave the card behind either.
    assert not (built / "README.md").exists()


def test_push_refuses_a_directory_that_would_not_load(built: Path) -> None:
    # Sizes still match the sidecar, so only the identity check sees this.
    with pytest.raises(RuntimeError, match="Refusing to publish"):
        push(
            source_dir=built,
            repo_id="org/whatever",
            persist_id="a-different-id@0000",
            dry_run=True,
        )

    with pytest.raises(RuntimeError, match="Refusing to publish"):
        push(
            source_dir=built,
            repo_id="org/whatever",
            persist_id=PERSIST_ID,
            expected_motions=len(LENGTHS) + 1,
            dry_run=True,
        )


def test_push_refuses_a_truncated_array(built: Path) -> None:
    path = built / "body_quat_w.memmap"
    with path.open("r+b") as handle:
        handle.truncate(path.stat().st_size - 4)

    with pytest.raises(RuntimeError, match="expected"):
        push(
            source_dir=built,
            repo_id="org/whatever",
            persist_id=PERSIST_ID,
            dry_run=True,
        )


def test_push_refuses_an_interrupted_build(built: Path) -> None:
    (built / SIDECAR_NAME).unlink()
    with pytest.raises(RuntimeError, match="missing"):
        push(
            source_dir=built,
            repo_id="org/whatever",
            persist_id=PERSIST_ID,
            dry_run=True,
        )


def test_fetch_validates_what_it_downloaded(
    built: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A download that lands short must fail before training reads it."""
    dest = tmp_path / "dest"

    def fake_download(repo_id, *, repo_type, local_dir, max_workers):
        target = Path(local_dir)
        target.mkdir(parents=True, exist_ok=True)
        for item in built.iterdir():
            (target / item.name).write_bytes(item.read_bytes())
        return str(target)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
    out = fetch(
        repo_id="org/whatever",
        dest_dir=dest,
        persist_id=PERSIST_ID,
        body_names=list(BODY_NAMES),
        anchor_body=ANCHOR,
        expected_motions=len(LENGTHS),
    )
    assert out == dest

    # Now make the *remote* side short, so the next fetch lands a truncated
    # array. Truncating the destination would just be repaired by the download.
    truncated = built / "qpos.memmap"
    with truncated.open("r+b") as handle:
        handle.truncate(truncated.stat().st_size - 4)
    with pytest.raises(RuntimeError, match="Refusing to use downloaded"):
        fetch(
            repo_id="org/whatever",
            dest_dir=tmp_path / "dest2",
            persist_id=PERSIST_ID,
            expected_motions=len(LENGTHS),
        )


def test_verify_remote_catches_an_upload_that_did_not_land(
    built: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failure mode that mattered: the client reports success anyway.

    A resumed upload cache pointed at a recreated repo skips every large file
    and still exits 0, so only comparing against the Hub listing catches it.
    """
    from imitationlab.data_tools.reference_arrays import publish as mod

    class FakeSibling:
        def __init__(self, name, size):
            self.rfilename, self.size = name, size

    class FakeInfo:
        def __init__(self, siblings):
            self.siblings = siblings

    class FakeApi:
        def __init__(self, siblings):
            self._siblings = siblings

        def repo_info(self, repo_id, *, repo_type, files_metadata):
            return FakeInfo(self._siblings)

    local = {
        p.name: p.stat().st_size
        for p in built.iterdir()
        if p.is_file() and not p.name.startswith(".")
    }
    import huggingface_hub

    # Everything present at the right size: passes.
    monkeypatch.setattr(
        huggingface_hub,
        "HfApi",
        lambda: FakeApi([FakeSibling(n, s) for n, s in local.items()]),
    )
    mod.verify_remote("org/whatever", built)

    # Nine of ten committed to a repo that no longer exists.
    partial = [FakeSibling("README.md", 10), FakeSibling(".gitattributes", 5)]
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: FakeApi(partial))
    with pytest.raises(RuntimeError, match="did not land"):
        mod.verify_remote("org/whatever", built)

    # Present but truncated.
    short = [FakeSibling(n, max(s - 4, 0)) for n, s in local.items()]
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: FakeApi(short))
    with pytest.raises(RuntimeError, match="bytes on the Hub"):
        mod.verify_remote("org/whatever", built)


def test_card_records_identity_and_conventions(built: Path) -> None:
    sidecar = json.loads((built / SIDECAR_NAME).read_text(encoding="utf-8"))
    card = dataset_card(sidecar, source_repo="org/source")

    assert PERSIST_ID in card
    assert "org/source" in card
    assert "`anchor_quat_w`" in card and "xyzw" in card
    assert "`body_quat_w`" in card and "wxyz" in card
    assert f"Baked anchor body: `{ANCHOR}`" in card
    # The trajectory table is what makes the arrays loadable at all.
    assert SIDECAR_NAME in card
    assert "derived at load time" in card


class _FakeRepoInfo:
    def __init__(self, private, siblings=()):
        self.private = private
        self.siblings = list(siblings)


class _FakeSibling:
    def __init__(self, name, size):
        self.rfilename, self.size = name, size


class _FakeHub:
    """Records Hub calls; ``visibility`` is None when the repo does not exist."""

    def __init__(self, visibility: bool | None):
        self.visibility = visibility
        self.created: list[dict] = []
        self.uploaded: list[str] = []
        self.folder: Path | None = None

    def api(self):
        hub = self

        class Api:
            def repo_exists(self, repo_id, *, repo_type):
                return hub.visibility is not None

            def create_repo(self, repo_id, *, repo_type, private):
                hub.created.append({"repo_id": repo_id, "private": private})
                hub.visibility = private

            def repo_info(self, repo_id, *, repo_type, files_metadata=False):
                siblings = []
                if hub.folder is not None:
                    siblings = [
                        _FakeSibling(path.name, path.stat().st_size)
                        for path in hub.folder.iterdir()
                        if path.is_file()
                    ]
                return _FakeRepoInfo(hub.visibility, siblings)

        return Api()

    def upload_large_folder(self, *, folder_path, repo_id, repo_type):
        self.uploaded.append(repo_id)
        self.folder = Path(folder_path)


@pytest.fixture
def fake_hub(monkeypatch: pytest.MonkeyPatch):
    import huggingface_hub

    def install(visibility: bool | None) -> _FakeHub:
        hub = _FakeHub(visibility)
        monkeypatch.setattr(huggingface_hub, "HfApi", hub.api)
        monkeypatch.setattr(
            huggingface_hub, "upload_large_folder", hub.upload_large_folder
        )
        return hub

    return install


def _push(source_dir: Path, *, private: bool = True, dry_run: bool = False):
    return push(
        source_dir=source_dir,
        repo_id="org/bank",
        persist_id=PERSIST_ID,
        private=private,
        dry_run=dry_run,
    )


def test_push_creates_a_missing_repo_with_the_requested_visibility(
    built: Path, fake_hub
) -> None:
    hub = fake_hub(None)
    _push(built)
    assert hub.created == [{"repo_id": "org/bank", "private": True}]
    assert hub.uploaded == ["org/bank"]


def test_push_uploads_into_an_existing_repo_of_matching_visibility(
    built: Path, fake_hub
) -> None:
    hub = fake_hub(True)
    _push(built)
    assert hub.created == []
    assert hub.uploaded == ["org/bank"]


@pytest.mark.parametrize("private", [True, False])
def test_push_refuses_an_existing_repo_of_other_visibility(
    built: Path, fake_hub, private: bool
) -> None:
    """``create_repo(exist_ok=True)`` would keep the old visibility silently."""
    hub = fake_hub(not private)
    with pytest.raises(RuntimeError, match="Refusing to push to org/bank"):
        _push(built, private=private)
    assert hub.created == [] and hub.uploaded == []
    assert not (built / "README.md").exists()


def test_public_push_refuses_a_bones_seed_bank(tmp_path: Path, fake_hub) -> None:
    source = _build(tmp_path, dataset_name="bones_seed")
    hub = fake_hub(None)
    with pytest.raises(RuntimeError, match="BONES-SEED"):
        _push(source, private=False, dry_run=True)
    with pytest.raises(RuntimeError, match="BONES-SEED"):
        _push(source, private=False)
    assert hub.created == [] and hub.uploaded == []

    # A private push of the same bank is allowed.
    _push(source, private=True)
    assert hub.uploaded == ["org/bank"]


def test_cli_reports_a_refused_public_push(
    tmp_path: Path, fake_hub, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _build(tmp_path, dataset_name="bones_seed")
    fake_hub(None)
    argv = ["push", "--source-dir", str(source), "--repo-id", "org/bank"]
    assert main([*argv, "--persist-id", PERSIST_ID, "--public"]) == 1
    assert "BONES-SEED" in capsys.readouterr().err
    # Legacy snake-case spellings remain accepted.
    legacy = ["push", "--source_dir", str(source), "--repo_id", "org/bank"]
    assert main([*legacy, "--persist_id", PERSIST_ID, "--dry_run"]) == 0
