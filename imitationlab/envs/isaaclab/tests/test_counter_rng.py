"""Counter-based RNG: reference values, subset invariance, distribution."""

from __future__ import annotations

import pytest
import torch

from imitationlab.envs.isaaclab.envs.graph_step import counter_rng


def _splitmix64_reference(x: int) -> int:
    mask = (1 << 64) - 1
    z = (x + 0x9E3779B97F4A7C15) & mask
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
    return z ^ (z >> 31)


def _devices() -> list[str]:
    return ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


@pytest.mark.parametrize("device", _devices())
def test_splitmix64_matches_unsigned_reference(device: str) -> None:
    values = [0, 1, 2, 12345, (1 << 63) - 1, 1 << 63, (1 << 64) - 1]
    signed = [v - (1 << 64) if v >= (1 << 63) else v for v in values]
    out = counter_rng.splitmix64(torch.tensor(signed, dtype=torch.int64, device=device))
    got = [int(v) & ((1 << 64) - 1) for v in out.cpu()]
    assert got == [_splitmix64_reference(v) for v in values]


@pytest.mark.parametrize("device", _devices())
def test_rows_depend_only_on_their_own_keys(device: str) -> None:
    n = 1000
    env = torch.arange(n, device=device)
    counter = torch.randint(0, 50, (n,), device=device)
    full = counter_rng.uniform(7, 3, env, counter, width=6)
    subset = torch.tensor([3, 17, 999, 500], device=device)
    part = counter_rng.uniform(7, 3, env[subset], counter[subset], width=6)
    assert torch.equal(full[subset], part)
    # Another stream or counter gives other numbers.
    assert not torch.equal(full, counter_rng.uniform(7, 4, env, counter, width=6))
    assert not torch.equal(full, counter_rng.uniform(7, 3, env, counter + 1, width=6))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_uniform_is_in_unit_interval_and_flat(dtype: torch.dtype) -> None:
    env = torch.arange(200_000)
    u = counter_rng.uniform(0, 1, env, torch.zeros_like(env), width=4, dtype=dtype)
    assert u.dtype == dtype
    assert float(u.min()) >= 0.0 and float(u.max()) < 1.0
    assert abs(float(u.mean()) - 0.5) < 3e-3
    assert abs(float(u.var()) - 1.0 / 12.0) < 2e-3
    hist = torch.histc(u.float(), bins=20, min=0.0, max=1.0) / u.numel()
    assert float((hist - 0.05).abs().max()) < 3e-3
    # Columns are not correlated with each other.
    corr = torch.corrcoef(u.T.float())
    off_diag = corr - torch.eye(4)
    assert float(off_diag.abs().max()) < 1e-2
