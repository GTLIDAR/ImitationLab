"""Counter-based random numbers keyed per environment.

Reset-time draws in the eager env come from one shared generator stream and
are shaped by how many environments reset this step (``sample_uniform((k,
6))``), so every draw depends on which other environments happened to reset.
The masked step instead derives each environment's numbers from a hash of
``(seed, stream, env index, per-env counter, column)``: a pure function, so

- an environment's reset noise depends only on its own episode count;
- computing all ``N`` rows and selecting the resetting ones gives exactly the
  numbers a subset computation gives;
- the kernels are elementwise integer ops: no generator state, no host sync,
  and they replay in a CUDA graph with the counters read from device memory.

The mixer is splitmix64 (Steele, Lea, Flood 2014) applied in a chain over the
keys. Torch has no unsigned 64-bit arithmetic, so the hash runs on ``int64``
with wrapping multiplication and masked right shifts (logical shifts).
"""

from __future__ import annotations

import torch

_MASK64 = (1 << 64) - 1


def _signed(value: int) -> int:
    """An unsigned 64-bit constant as the int64 with the same bits."""
    value &= _MASK64
    return value - (1 << 64) if value >= (1 << 63) else value


_GOLDEN = _signed(0x9E3779B97F4A7C15)
_MIX1 = _signed(0xBF58476D1CE4E5B9)
_MIX2 = _signed(0x94D049BB133111EB)


def _shr(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Logical right shift of int64 bit patterns."""
    return (x >> bits) & ((1 << (64 - bits)) - 1)


def splitmix64(x: torch.Tensor) -> torch.Tensor:
    """The splitmix64 finalizer over int64 bit patterns (elementwise)."""
    z = x + _GOLDEN
    z = (z ^ _shr(z, 30)) * _MIX1
    z = (z ^ _shr(z, 27)) * _MIX2
    return z ^ _shr(z, 31)


def _splitmix64_int(x: int) -> int:
    """splitmix64 on a Python int (unsigned 64-bit), for host-side constants."""
    z = (x + 0x9E3779B97F4A7C15) & _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return z ^ (z >> 31)


def hash_keys(seed: int, stream: int, *keys: torch.Tensor) -> torch.Tensor:
    """Chain-hash integer key tensors (broadcast together) into int64 bits.

    The ``(seed, stream)`` prefix is hashed on the host and enters the first
    kernel as a scalar, so nothing is copied to the device (capture-safe).
    """
    if not keys:
        raise ValueError("hash_keys needs at least one key tensor.")
    prefix = _signed(
        _splitmix64_int(
            (int(seed) & _MASK64) ^ ((int(stream) * 0x632BE59BD9B4E019) & _MASK64)
        )
    )
    state = splitmix64(keys[0].to(torch.int64) ^ prefix)
    for key in keys[1:]:
        state = splitmix64(state ^ key.to(torch.int64))
    return state


def uniform(
    seed: int,
    stream: int,
    *keys: torch.Tensor,
    width: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Uniform [0, 1) numbers of shape ``broadcast(keys).shape + (width,)``.

    float32 output uses the top 24 bits, float64 the top 53, so every value is
    exactly representable and strictly below 1.
    """
    state = hash_keys(seed, stream, *keys).unsqueeze(-1)
    columns = torch.arange(width, dtype=torch.int64, device=state.device)
    bits = splitmix64(state ^ (columns * _GOLDEN))
    if dtype == torch.float64:
        return _shr(bits, 11).to(torch.float64) * (2.0**-53)
    if dtype == torch.float32:
        return _shr(bits, 40).to(torch.float32) * (2.0**-24)
    raise TypeError(f"uniform supports float32 and float64, not {dtype}.")


def uniform_range(
    lower: torch.Tensor | float,
    upper: torch.Tensor | float,
    seed: int,
    stream: int,
    *keys: torch.Tensor,
    width: int,
) -> torch.Tensor:
    """``lower + (upper - lower) * u`` with ``u`` from :func:`uniform` (float32)."""
    u = uniform(seed, stream, *keys, width=width, dtype=torch.float32)
    return lower + (upper - lower) * u


__all__ = ["hash_keys", "splitmix64", "uniform", "uniform_range"]
