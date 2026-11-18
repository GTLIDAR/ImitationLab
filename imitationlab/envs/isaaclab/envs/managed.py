"""CUDA managed (unified) memory as a torch tensor.

``cudaMallocManaged`` gives one address that both the CPU and the GPU read; the
driver moves pages to whichever side touches them. :class:`ManagedTensor` wraps
such an allocation as a CUDA tensor, so a read-only table larger than the device
can live in host memory while the pages the GPU actually reads migrate to it on
first use and stay there (``read_mostly``: the GPU keeps a duplicate of each page
it read, dropped again under memory pressure). Against :class:`HostMappedTensor`,
whose every read crosses PCIe, the hot rows are served from device memory after
their first touch; the cold rows still cost a page fault and a migration.

The advice calls use the CUDA 13 runtime signatures (``cudaMemLocation``). The
tensor is valid until :meth:`close`; keep the object alive with it.
"""

from __future__ import annotations

import ctypes
import glob
import logging
import os
from time import perf_counter

import torch

logger = logging.getLogger(__name__)

_CUDA_MEM_ATTACH_GLOBAL = 1
_ADVISE_SET_READ_MOSTLY = 1
_ADVISE_UNSET_READ_MOSTLY = 2
_ADVISE_SET_PREFERRED_LOCATION = 3
_ADVISE_UNSET_PREFERRED_LOCATION = 4
_ADVISE_SET_ACCESSED_BY = 5
_ADVISE_UNSET_ACCESSED_BY = 6
_LOCATION_DEVICE = 1
_LOCATION_HOST = 2

_TYPESTR = {
    torch.float32: "<f4",
    torch.float64: "<f8",
    torch.float16: "<f2",
    torch.int64: "<i8",
    torch.int32: "<i4",
    torch.uint8: "|u1",
    torch.bool: "|b1",
}


class _MemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _CudaArrayInterface:
    def __init__(self, ptr: int, shape: tuple[int, ...], typestr: str) -> None:
        self.__cuda_array_interface__ = {
            "shape": shape,
            "typestr": typestr,
            "data": (ptr, False),
            "version": 3,
            "strides": None,
        }


_cudart: ctypes.CDLL | None = None


def cudart() -> ctypes.CDLL:
    """The CUDA runtime torch was built against (``libcudart.so.<major>``)."""
    global _cudart
    if _cudart is None:
        major = str(torch.version.cuda or "").split(".")[0]
        site = os.path.dirname(os.path.dirname(torch.__file__))
        patterns = [
            os.path.join(site, "nvidia", "cu*", "lib", f"libcudart.so.{major}"),
            os.path.join(
                site, "nvidia", "cuda_runtime", "lib", f"libcudart.so.{major}"
            ),
            os.path.join(os.path.dirname(torch.__file__), "lib", "libcudart*.so*"),
        ]
        found = [p for pat in patterns for p in glob.glob(pat)]
        if not found:
            raise RuntimeError("libcudart was not found beside torch.")
        lib = ctypes.CDLL(found[0])
        lib.cudaMallocManaged.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
            ctypes.c_uint,
        ]
        lib.cudaMallocManaged.restype = ctypes.c_int
        lib.cudaFree.argtypes = [ctypes.c_void_p]
        lib.cudaFree.restype = ctypes.c_int
        lib.cudaMemAdvise.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            _MemLocation,
        ]
        lib.cudaMemAdvise.restype = ctypes.c_int
        lib.cudaGetErrorString.argtypes = [ctypes.c_int]
        lib.cudaGetErrorString.restype = ctypes.c_char_p
        _cudart = lib
    return _cudart


def _check(lib: ctypes.CDLL, err: int, what: str) -> None:
    if int(err) != 0:
        raise RuntimeError(
            f"{what} failed: {lib.cudaGetErrorString(err).decode()} ({int(err)})"
        )


class ManagedTensor:
    """A zero-filled CUDA tensor on ``cudaMallocManaged`` memory."""

    def __init__(
        self, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device | str
    ) -> None:
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("ManagedTensor is read through a CUDA device.")
        typestr = _TYPESTR.get(dtype)
        if typestr is None:
            raise TypeError(f"Unsupported dtype {dtype} for managed memory.")
        self.device = device
        self.index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        numel = 1
        for s in shape:
            numel *= int(s)
        self.nbytes = numel * torch.empty((), dtype=dtype).element_size()
        lib = cudart()
        ptr = ctypes.c_void_p()
        start = perf_counter()
        with torch.cuda.device(device):
            _check(
                lib,
                lib.cudaMallocManaged(
                    ctypes.byref(ptr),
                    ctypes.c_size_t(self.nbytes),
                    _CUDA_MEM_ATTACH_GLOBAL,
                ),
                "cudaMallocManaged",
            )
            self.ptr = int(ptr.value or 0)
            self.tensor = torch.as_tensor(
                _CudaArrayInterface(self.ptr, tuple(int(s) for s in shape), typestr),
                device=device,
            )
        if self.tensor.data_ptr() != self.ptr:
            self.close()
            raise RuntimeError("The CUDA tensor does not alias the managed allocation.")
        # Zero from the host side: the pages start resident in host memory.
        ctypes.memset(self.ptr, 0, self.nbytes)
        logger.warning(
            "Allocated %.1f GiB of CUDA managed memory for %s (%.1f s).",
            self.nbytes / 1024**3,
            device,
            perf_counter() - start,
        )

    def _advise(self, advice: int, location: _MemLocation) -> None:
        lib = cudart()
        _check(
            lib,
            lib.cudaMemAdvise(
                ctypes.c_void_p(self.ptr),
                ctypes.c_size_t(self.nbytes),
                advice,
                location,
            ),
            "cudaMemAdvise",
        )

    def prefer_host(self, on: bool = True) -> None:
        """Pin the pages' home to the host (writes from the GPU then cross PCIe
        instead of pulling pages in); off lets them migrate to the reader."""
        host = _MemLocation(_LOCATION_HOST, 0)
        self._advise(
            _ADVISE_SET_PREFERRED_LOCATION if on else _ADVISE_UNSET_PREFERRED_LOCATION,
            host,
        )
        dev = _MemLocation(_LOCATION_DEVICE, self.index)
        self._advise(_ADVISE_SET_ACCESSED_BY if on else _ADVISE_UNSET_ACCESSED_BY, dev)

    def read_mostly(self, on: bool = True) -> None:
        """Let the GPU keep a read-only duplicate of every page it reads."""
        self._advise(
            _ADVISE_SET_READ_MOSTLY if on else _ADVISE_UNSET_READ_MOSTLY,
            _MemLocation(_LOCATION_DEVICE, self.index),
        )

    def close(self) -> None:
        if not getattr(self, "ptr", 0):
            return
        self.tensor = None  # type: ignore[assignment]
        with torch.cuda.device(self.device):
            torch.cuda.synchronize(self.device)
            cudart().cudaFree(ctypes.c_void_p(self.ptr))
        self.ptr = 0

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # pragma: no cover - interpreter teardown
            pass


__all__ = ["ManagedTensor", "cudart"]
