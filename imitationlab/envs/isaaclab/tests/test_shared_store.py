"""Shared-memory packed store: one builder, every process maps the same pages."""

from __future__ import annotations

import multiprocessing as mp
from datetime import timedelta
from pathlib import Path

import pytest
import torch

from imitationlab.envs.isaaclab.envs import shared_store
from imitationlab.envs.isaaclab.envs.shared_store import (
    job_scope,
    open_node_shared_packed,
    open_shared_packed,
    shared_store_path,
    store_key,
    unlink_shared,
)


def _fill(tensor: torch.Tensor) -> None:
    tensor.copy_(torch.arange(tensor.numel(), dtype=tensor.dtype).view_as(tensor))


def _child(directory: str, queue) -> None:
    calls = []

    def fill(tensor: torch.Tensor) -> None:
        calls.append(1)
        _fill(tensor)

    tensor, _built = open_shared_packed(
        "unit", (64, 8), fill, directory=Path(directory), poll_s=0.01
    )
    queue.put((len(calls), float(tensor.sum())))


def test_one_builder_and_identical_contents(tmp_path: Path) -> None:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [ctx.Process(target=_child, args=(str(tmp_path), queue)) for _ in range(3)]
    for proc in procs:
        proc.start()
    results = [queue.get(timeout=120) for _ in procs]
    for proc in procs:
        proc.join(timeout=60)
    expected = float(torch.arange(64 * 8, dtype=torch.float32).sum())
    assert sum(calls for calls, _ in results) == 1
    assert all(total == expected for _, total in results)
    unlink_shared("unit", directory=tmp_path)
    assert not shared_store_path("unit", directory=tmp_path).exists()


def test_size_mismatch_is_refused(tmp_path: Path) -> None:
    open_shared_packed("size", (4, 4), _fill, directory=tmp_path)
    # A second open with another shape finds the ready store and refuses it.
    with pytest.raises(RuntimeError, match="expected"):
        open_shared_packed("size", (4, 5), _fill, directory=tmp_path)


def test_a_failed_build_fails_the_waiters_at_once(tmp_path: Path) -> None:
    def broken(_tensor: torch.Tensor) -> None:
        raise OSError("source read failed")

    with pytest.raises(OSError, match="source read failed"):
        open_shared_packed("broken", (4, 4), broken, directory=tmp_path)
    assert not shared_store_path("broken", directory=tmp_path).exists()
    # A waiter (the lock is taken) sees the failure marker, not a timeout.
    with pytest.raises(RuntimeError, match="failed"):
        open_shared_packed(
            "broken", (4, 4), _fill, directory=tmp_path, timeout_s=30.0, poll_s=0.01
        )


def test_store_key_separates_content_and_jobs() -> None:
    base = dict(
        body_names=["pelvis", "torso_link"],
        field_names=["qpos", "qvel"],
        shape=(10, 4),
        dtype=torch.float32,
        scope="job1",
    )
    key = store_key("bones@e714bbff", **base)
    assert key == store_key("bones@e714bbff", **base)
    assert key.startswith("bones@e714bbff-")
    assert key != store_key("bones@e714bbff", **{**base, "scope": "job2"})
    assert key != store_key(
        "bones@e714bbff", **{**base, "body_names": ["pelvis", "left_knee_link"]}
    )
    assert key != store_key("bones@e714bbff", **{**base, "shape": (10, 5)})
    assert key != store_key("other@0", **base)


def test_job_scope_is_shared_by_one_torchrun_launch(monkeypatch) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "run-abc")
    assert job_scope() == "run-abc"
    monkeypatch.setenv("SLURM_JOB_ID", "6510480")
    assert job_scope() == "6510480-run-abc"
    monkeypatch.delenv("TORCHELASTIC_RUN_ID")
    assert job_scope() == f"6510480-pid{__import__('os').getpid()}"


def test_single_process_maps_then_unlinks(tmp_path: Path) -> None:
    tensor = open_node_shared_packed("solo", (8, 3), _fill, directory=tmp_path)
    assert tensor is not None
    assert torch.equal(tensor, torch.arange(24, dtype=torch.float32).view(8, 3))
    # The names are gone; the mapping still holds the pages.
    assert list(tmp_path.iterdir()) == []
    tensor.add_(1.0)
    assert float(tensor[0, 0]) == 1.0


def test_no_room_falls_back_without_creating_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(shared_store, "shm_free_bytes", lambda _directory: 1024)
    calls = []
    assert (
        open_node_shared_packed(
            "full", (1024, 8), lambda t: calls.append(t), directory=tmp_path
        )
        is None
    )
    assert calls == []
    assert list(tmp_path.iterdir()) == []


def _rank(rank: int, directory: str, rendezvous: str, queue) -> None:
    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        calls = []

        def fill(tensor: torch.Tensor) -> None:
            calls.append(1)
            _fill(tensor)

        tensor = open_node_shared_packed(
            "ranks", (32, 4), fill, directory=Path(directory), poll_s=0.01
        )
        assert tensor is not None
        # After the call every name is gone on every rank.
        names = sorted(p.name for p in Path(directory).iterdir() if p.name != "group")
        dist.barrier()
        # One rank writes; the other sees the write: one set of pages.
        if rank == 0:
            tensor[3, 2] = -7.0
        dist.barrier()
        queue.put((rank, len(calls), float(tensor[3, 2]), names))
    finally:
        dist.destroy_process_group()


def test_two_ranks_share_one_copy_and_remove_its_names(tmp_path: Path) -> None:
    store_dir = tmp_path / "shm"
    store_dir.mkdir()
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [
        ctx.Process(
            target=_rank,
            args=(rank, str(store_dir), str(tmp_path / "group"), queue),
        )
        for rank in range(2)
    ]
    for proc in procs:
        proc.start()
    results = sorted(queue.get(timeout=120) for _ in procs)
    for proc in procs:
        proc.join(timeout=60)
        assert proc.exitcode == 0
    assert sum(calls for _, calls, _, _ in results) == 1
    assert [value for _, _, value, _ in results] == [-7.0, -7.0]
    assert all(names == [] for _, _, _, names in results)
