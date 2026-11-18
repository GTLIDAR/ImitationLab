import os
import stat

import pytest
import torch

from imitationlab.encoders import _checkpoint_io as io


def test_counts_nested_tensor_bytes():
    state = {"a": torch.zeros(4), "b": [torch.zeros(2, dtype=torch.float64), 3]}
    assert io.tensor_nbytes(state) == 4 * 4 + 2 * 8


def test_stages_in_tempdir_when_it_has_room(tmp_path, monkeypatch):
    monkeypatch.setattr(io.tempfile, "gettempdir", lambda: str(tmp_path / "tmp"))
    monkeypatch.setattr(io, "_free_bytes", lambda path: 10 << 30)
    assert io.staging_dir(tmp_path / "out" / "x.pt", 1 << 30) == tmp_path / "tmp"


def test_stages_beside_target_when_tempdir_is_too_small(tmp_path, monkeypatch):
    small = tmp_path / "tmp"
    monkeypatch.setattr(io.tempfile, "gettempdir", lambda: str(small))
    monkeypatch.setattr(
        io, "_free_bytes", lambda path: (1 << 20) if path == small else (10 << 30)
    )
    target = tmp_path / "out" / "x.pt"
    assert io.staging_dir(target, 1 << 30) == target.parent


def test_save_is_complete_readable_and_leaves_no_staging_file(tmp_path, monkeypatch):
    monkeypatch.setattr(io.tempfile, "gettempdir", lambda: str(tmp_path / "missing"))
    target = tmp_path / "out" / "best.pt"
    io.save_staged({"w": torch.arange(3.0)}, target)
    assert torch.equal(torch.load(target)["w"], torch.arange(3.0))
    assert sorted(p.name for p in target.parent.iterdir()) == ["best.pt"]
    umask = os.umask(0)
    os.umask(umask)
    assert stat.S_IMODE(target.stat().st_mode) == 0o666 & ~umask


def test_failed_write_names_the_directory_and_removes_the_partial_file(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(io.tempfile, "gettempdir", lambda: str(tmp_path / "missing"))

    def broken_save(state, path):
        with open(path, "wb") as handle:
            handle.write(b"partial")
        raise RuntimeError("basic_ios::clear: iostream error")

    monkeypatch.setattr(io.torch, "save", broken_save)
    target = tmp_path / "out" / "best.pt"
    with pytest.raises(RuntimeError, match="TMPDIR"):
        io.save_staged({"w": torch.zeros(1)}, target)
    assert list(target.parent.iterdir()) == []
