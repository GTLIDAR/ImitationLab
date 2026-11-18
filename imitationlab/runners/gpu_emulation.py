"""Emulate a smaller GPU on a larger one (``IMITATION_EMULATE_GPU_GIB``).

A recipe meant for 16 GiB cards is tuned on cluster GPUs with 48-80 GiB. Setting
``IMITATION_EMULATE_GPU_GIB=<GiB>`` before training makes the process allocate a ballast
of ``total - GiB`` bytes on its device at start, before the simulator or any other
allocator runs, and hold it to exit. Every later allocation, torch's and the physics
engine's alike, then sees at most ``GiB`` free, so a recipe that would not fit the small
card fails here the same way. The ballast is one live tensor, never reused by torch's
caching allocator. Speed is not emulated: compute units and bandwidth stay the large GPU's.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

ENV_VAR = "IMITATION_EMULATE_GPU_GIB"
_ballast = None


def ballast_bytes(total: int, free: int, target_gib: float) -> int:
    """Bytes to hold so that at most ``target_gib`` of ``total`` stays usable; what other
    processes already use counts toward the ballast (it is not usable either)."""
    target = int(target_gib * 2**30)
    if target <= 0:
        raise ValueError(f"{ENV_VAR} must be positive, got {target_gib}.")
    used_elsewhere = total - free
    return max(0, total - target - used_elsewhere)


def apply_from_env() -> int:
    """Allocate the ballast if ``IMITATION_EMULATE_GPU_GIB`` is set; returns its bytes."""
    global _ballast
    value = os.environ.get(ENV_VAR, "").strip()
    if not value or _ballast is not None:
        return 0
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(f"{ENV_VAR}={value} needs a CUDA device.")
    free, total = torch.cuda.mem_get_info()
    nbytes = ballast_bytes(total, free, float(value))
    if nbytes > 0:
        _ballast = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    free_after, _ = torch.cuda.mem_get_info()
    logger.warning(
        "%s=%s: holding %.2f GiB of %.2f GiB; %.2f GiB free for training.",
        ENV_VAR,
        value,
        nbytes / 2**30,
        total / 2**30,
        free_after / 2**30,
    )
    return nbytes
