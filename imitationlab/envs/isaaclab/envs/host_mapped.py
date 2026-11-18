"""Device-addressable views of host memory (CUDA zero-copy).

A CPU tensor registered with ``cudaHostRegister(..., cudaHostRegisterMapped)``
is page-locked in place and, under unified virtual addressing, readable by
CUDA kernels at its host address. :class:`HostMappedTensor` registers a
contiguous CPU tensor and exposes a CUDA tensor aliasing the same bytes, so
ordinary torch ops (``index_select`` with device indices) read host rows over
PCIe inside one kernel: no host synchronization, no staging copy, and
capturable in a CUDA graph. Values are the stored bytes, so a gather through
the alias equals the same gather on the CPU tensor bit for bit.

The alias is only valid while the registration holds: keep the
:class:`HostMappedTensor` alive as long as the CPU tensor and call
:meth:`HostMappedTensor.close` before freeing it.
"""

from __future__ import annotations

import logging
from time import perf_counter

import torch

logger = logging.getLogger(__name__)

# cudaHostRegisterMapped: map the range into the device address space.
_CUDA_HOST_REGISTER_MAPPED = 2

_TYPESTR = {
    torch.float32: "<f4",
    torch.float64: "<f8",
    torch.float16: "<f2",
    torch.int64: "<i8",
    torch.int32: "<i4",
    torch.int16: "<i2",
    torch.int8: "|i1",
    torch.uint8: "|u1",
    torch.bool: "|b1",
}


class _CudaArrayInterface:
    """Minimal ``__cuda_array_interface__`` holder for ``torch.as_tensor``."""

    def __init__(self, ptr: int, shape: tuple[int, ...], typestr: str) -> None:
        self.__cuda_array_interface__ = {
            "shape": shape,
            "typestr": typestr,
            "data": (ptr, False),
            "version": 3,
            "strides": None,
        }


class HostMappedTensor:
    """Register a contiguous CPU tensor and expose a CUDA alias of it."""

    def __init__(self, host: torch.Tensor, device: torch.device | str) -> None:
        device = torch.device(device)
        if host.device.type != "cpu":
            raise ValueError("HostMappedTensor needs a CPU tensor.")
        if not host.is_contiguous():
            raise ValueError("HostMappedTensor needs a contiguous tensor.")
        if device.type != "cuda":
            raise ValueError("HostMappedTensor maps into a CUDA device.")
        typestr = _TYPESTR.get(host.dtype)
        if typestr is None:
            raise TypeError(f"Unsupported dtype {host.dtype} for a host mapping.")
        self.host = host
        self.device = device
        self._registered = False
        nbytes = host.numel() * host.element_size()
        start = perf_counter()
        with torch.cuda.device(device):
            err = torch.cuda.cudart().cudaHostRegister(
                host.data_ptr(), nbytes, _CUDA_HOST_REGISTER_MAPPED
            )
            if int(err) != 0:
                raise RuntimeError(f"cudaHostRegister failed: {err}")
            self._registered = True
            self.device_view = torch.as_tensor(
                _CudaArrayInterface(host.data_ptr(), tuple(host.shape), typestr),
                device=device,
            )
        if self.device_view.data_ptr() != host.data_ptr():
            self.close()
            raise RuntimeError("The CUDA alias does not share the host address.")
        logger.warning(
            "Mapped %.1f GiB of host memory into %s for zero-copy gathers (%.1f s).",
            nbytes / 1024**3,
            device,
            perf_counter() - start,
        )

    def close(self) -> None:
        """Drop the alias and unregister the host range (idempotent)."""
        if not self._registered:
            return
        self.device_view = None  # type: ignore[assignment]
        with torch.cuda.device(self.device):
            torch.cuda.synchronize(self.device)
            torch.cuda.cudart().cudaHostUnregister(self.host.data_ptr())
        self._registered = False

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # pragma: no cover - interpreter teardown
            pass


__all__ = ["HostMappedTensor"]
