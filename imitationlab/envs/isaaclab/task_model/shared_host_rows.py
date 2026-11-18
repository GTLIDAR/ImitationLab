"""Run-local, RAM-backed reference rows shared by trainer and probe processes.

Linux memfd pages can be CUDA host-registered in each process and are charged
once to the job's memory cgroup. They need neither a second bank allocation nor
a large /dev/shm mount. A descriptor is valid only while its owner is alive.
"""

from __future__ import annotations

import ctypes
import os
import weakref
from pathlib import Path

import torch


class SharedHostRows:
    """A float32 matrix in anonymous shared memory, addressed through /proc."""

    def __init__(self, rows: int, columns: int, *, path: str = "") -> None:
        if rows < 1 or columns < 1:
            raise ValueError("shared rows need positive dimensions.")
        size = rows * columns * 4
        self.owner = not path
        if self.owner:
            # Some Pixi Python builds omit os.memfd_create despite glibc support.
            libc = ctypes.CDLL(None, use_errno=True)
            create = libc.memfd_create
            create.argtypes, create.restype = (
                [ctypes.c_char_p, ctypes.c_uint],
                ctypes.c_int,
            )
            fd = create(b"g1-reference", 1)  # MFD_CLOEXEC; child opens /proc path
            if fd < 0:
                raise OSError(ctypes.get_errno(), "memfd_create")
            self._close = weakref.finalize(self, os.close, fd)
            os.fchmod(fd, 0o600)
            os.ftruncate(fd, size)
            path = f"/proc/{os.getpid()}/fd/{fd}"
        elif Path(path).stat().st_size != size:
            raise ValueError("the shared reference library has a different row count.")
        self.path = path
        self.host = torch.from_file(
            path, shared=True, size=rows * columns, dtype=torch.float32
        ).reshape(rows, columns)
        if self.owner:
            self.host.zero_()
