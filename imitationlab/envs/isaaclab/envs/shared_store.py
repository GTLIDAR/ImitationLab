"""One host copy of a packed reference store shared by the processes of a node.

Each training process normally reads the reference arrays into its own
anonymous host memory (44.8 GiB for the BONES-SEED store), so N ranks on one
node need N copies. :func:`open_shared_packed` keeps one copy in a POSIX
shared-memory file (``/dev/shm``): the first process to take the build lock
fills it, the others wait for its ready marker, and every process maps the
same pages. A ``MAP_SHARED`` tmpfs mapping can be page-locked with
``cudaHostRegister`` in each process, so the zero-copy device gather
(:class:`~imitationlab.envs.isaaclab.envs.host_mapped.HostMappedTensor`) works on it
as on private memory (checked 2026-10-24: two mappings of one file register
and gather identically).

The file outlives its processes unless unlinked. :func:`open_node_shared_packed`
is the entry point for the processes of one job: every process agrees on
whether the tmpfs has room (a collective, before anyone creates the file),
maps the store, waits at a barrier until every process has mapped it, and then
removes its names. From then on the pages live exactly as long as the last
mapping, so a crash after the barrier leaks nothing. A crash between the build
and the barrier leaves the file behind; ``atexit`` removes it on any Python
exit, but not after a SIGKILL.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import os
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

SHM_DIR = Path("/dev/shm")

# Free tmpfs space kept beyond the store itself before sharing is attempted.
SHM_RESERVE_BYTES = 2 * 1024**3


def shared_store_path(key: str, *, directory: Path = SHM_DIR) -> Path:
    """The tmpfs path for a store identified by ``key`` (e.g. a persist id)."""
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)
    return directory / f"isaaclab_imitation_{safe}.bin"


def job_scope() -> str:
    """A token shared by the processes of one job and by no other job.

    torchrun gives every worker of one launch the same ``TORCHELASTIC_RUN_ID``
    (a fresh UUID under ``--standalone``); a single process uses its own pid.
    Scoping the store name to the job keeps two jobs on one node from racing
    on one file's build and unlink, and keeps a stale file from a crashed job
    from being mistaken for a ready store.
    """
    run_id = os.environ.get("TORCHELASTIC_RUN_ID")
    job_id = os.environ.get("SLURM_JOB_ID")
    if run_id:
        return f"{job_id}-{run_id}" if job_id else run_id
    return f"{job_id}-pid{os.getpid()}" if job_id else f"pid{os.getpid()}"


def store_key(
    persist_id: str,
    *,
    body_names: Sequence[str],
    field_names: Sequence[str],
    shape: tuple[int, int],
    dtype: torch.dtype,
    scope: str,
) -> str:
    """The shared-store name: content identity (persist id, body set, layout) + job."""
    layout = json.dumps(
        {
            "bodies": [str(name) for name in body_names],
            "fields": [str(name) for name in field_names],
            "shape": [int(shape[0]), int(shape[1])],
            "dtype": str(dtype),
        },
        sort_keys=True,
    )
    digest = hashlib.sha1(layout.encode()).hexdigest()[:16]
    return f"{persist_id}-{digest}-{scope}"


def shm_free_bytes(directory: Path = SHM_DIR) -> int | None:
    """Free bytes on the filesystem holding ``directory`` (``None`` if absent)."""
    try:
        stat = os.statvfs(directory)
    except OSError:
        return None
    return int(stat.f_bavail) * int(stat.f_frsize)


def open_shared_packed(
    key: str,
    shape: tuple[int, int],
    fill: Callable[[torch.Tensor], None],
    *,
    dtype: torch.dtype = torch.float32,
    directory: Path = SHM_DIR,
    timeout_s: float = 3600.0,
    poll_s: float = 1.0,
) -> tuple[torch.Tensor, bool]:
    """Map (building it if this process is first) a shared ``[rows, width]`` store.

    ``fill`` receives the mapped tensor and writes every row; it runs in
    exactly one process per ``key``. Other processes block until the builder
    writes the ready marker (``timeout_s``), then map the same file. A builder
    whose ``fill`` raises writes a failure marker, so the others fail at once
    instead of waiting out the timeout. Returns ``(tensor, built_here)``.
    """
    path = shared_store_path(key, directory=directory)
    ready = path.with_suffix(".ready")
    failed = path.with_suffix(".failed")
    lock = path.with_suffix(".lock")
    numel = int(shape[0]) * int(shape[1])
    nbytes = numel * torch.empty((), dtype=dtype).element_size()
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        builder = False
    else:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        builder = True

    if builder:
        start = time.perf_counter()
        try:
            tensor = torch.from_file(str(path), shared=True, size=numel, dtype=dtype)
            tensor = tensor.view(int(shape[0]), int(shape[1]))
            fill(tensor)
        except BaseException:
            failed.write_text(str(os.getpid()))
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            raise
        ready.write_text(str(nbytes))
        logger.warning(
            "Built the shared reference store %s (%.1f GiB) in %.1f s.",
            path,
            nbytes / 1024**3,
            time.perf_counter() - start,
        )
        return tensor, True

    deadline = time.monotonic() + float(timeout_s)
    while not ready.exists():
        if failed.exists():
            raise RuntimeError(f"The builder of {path} failed; see its traceback.")
        if time.monotonic() > deadline:
            raise TimeoutError(f"Timed out waiting for {ready}.")
        time.sleep(poll_s)
    if path.stat().st_size != nbytes:
        raise RuntimeError(
            f"{path} holds {path.stat().st_size} bytes, expected {nbytes}."
        )
    tensor = torch.from_file(str(path), shared=True, size=numel, dtype=dtype)
    logger.warning("Mapped the shared reference store %s.", path)
    return tensor.view(int(shape[0]), int(shape[1])), False


def unlink_shared(key: str, *, directory: Path = SHM_DIR) -> None:
    """Remove the store's names; mapped pages live until their last mapping."""
    path = shared_store_path(key, directory=directory)
    for name in (
        path,
        path.with_suffix(".ready"),
        path.with_suffix(".lock"),
        path.with_suffix(".failed"),
    ):
        try:
            name.unlink()
        except FileNotFoundError:
            pass


def _process_group():
    """The initialized default process group's backend, or ``None``."""
    try:
        import torch.distributed as dist
    except ImportError:  # pragma: no cover - torch without distributed
        return None
    if not (dist.is_available() and dist.is_initialized()):
        return None
    return dist


def _all_agree(value: bool) -> bool:
    """Logical AND of ``value`` over every process of the job (collective)."""
    dist = _process_group()
    if dist is None:
        return bool(value)
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if dist.get_backend() == "nccl"
        else torch.device("cpu")
    )
    flag = torch.tensor([int(bool(value))], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def open_node_shared_packed(
    key: str,
    shape: tuple[int, int],
    fill: Callable[[torch.Tensor], None],
    *,
    dtype: torch.dtype = torch.float32,
    directory: Path = SHM_DIR,
    reserve_bytes: int = SHM_RESERVE_BYTES,
    timeout_s: float = 3600.0,
    poll_s: float = 1.0,
) -> torch.Tensor | None:
    """Every process of the job maps one store; ``None`` means "use private memory".

    Call it from EVERY process of the job (it contains collectives when a
    ``torch.distributed`` group is initialized): first all processes agree that
    ``directory`` has room for the store plus ``reserve_bytes``, then each maps
    it (one builds), then a barrier, then the names are removed. Without a
    process group the caller is the only process, so it unlinks right away.
    """
    numel = int(shape[0]) * int(shape[1])
    nbytes = numel * torch.empty((), dtype=dtype).element_size()
    free = shm_free_bytes(directory)
    room = free is not None and free >= nbytes + int(reserve_bytes)
    if not _all_agree(room):
        logger.warning(
            "Shared reference store disabled: %s has %s free, the store needs "
            "%.1f GiB plus a %.1f GiB reserve (or another process lacks room). "
            "Each process reads its own private copy.",
            directory,
            "no filesystem" if free is None else f"{free / 1024**3:.1f} GiB",
            nbytes / 1024**3,
            reserve_bytes / 1024**3,
        )
        return None

    # Any Python exit before the barrier below removes the names (idempotent).
    def cleanup() -> None:
        unlink_shared(key, directory=directory)

    atexit.register(cleanup)
    tensor, _built = open_shared_packed(
        key,
        shape,
        fill,
        dtype=dtype,
        directory=directory,
        timeout_s=timeout_s,
        poll_s=poll_s,
    )
    dist = _process_group()
    if dist is not None:
        # Every process has mapped the file; the names can go.
        dist.barrier()
    unlink_shared(key, directory=directory)
    atexit.unregister(cleanup)
    return tensor


__all__ = [
    "job_scope",
    "open_node_shared_packed",
    "open_shared_packed",
    "shared_store_path",
    "shm_free_bytes",
    "store_key",
    "unlink_shared",
]
