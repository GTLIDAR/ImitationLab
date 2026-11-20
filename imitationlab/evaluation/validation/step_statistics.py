"""Simulator-independent comparison helpers for ImitationLab."""

from __future__ import annotations

import math
from typing import Any

START_BINS = 20
LENGTH_BIN_EDGES = (0, 50, 100, 200, 400, 800, 1600, 3200, 1 << 40)


def _block_means(values: list[float], block: int, skip: int) -> list[float]:
    values = values[skip:]
    count = len(values) // block
    return [sum(values[i * block : (i + 1) * block]) / block for i in range(count)]


def _mean_se(blocks: list[float]) -> tuple[float, float]:
    n = len(blocks)
    mean = sum(blocks) / n
    if n < 2:
        return mean, float("inf")
    var = sum((b - mean) ** 2 for b in blocks) / (n - 1)
    return mean, math.sqrt(var / n)


def _chi_square_p(a: list[float], b: list[float]) -> tuple[float, float]:
    """Total-variation distance and the chi-square homogeneity p-value."""
    na, nb = sum(a), sum(b)
    if na <= 0 or nb <= 0:
        return float("nan"), float("nan")
    tv = 0.5 * sum(abs(x / na - y / nb) for x, y in zip(a, b))
    stat, dof = 0.0, -1
    for x, y in zip(a, b):
        total = x + y
        if total <= 0:
            continue
        dof += 1
        ea, eb = total * na / (na + nb), total * nb / (na + nb)
        stat += (x - ea) ** 2 / ea + (y - eb) ** 2 / eb
    if dof <= 0:
        return tv, 1.0
    # Wilson-Hilferty normal approximation to the chi-square upper tail.
    z = ((stat / dof) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * dof))) / math.sqrt(
        2.0 / (9.0 * dof)
    )
    return tv, 0.5 * math.erfc(z / math.sqrt(2.0))


def compare_statistics(
    a: dict[str, Any],
    b: dict[str, Any],
    *,
    block: int = 24,
    skip_blocks: int = 1,
) -> dict[str, Any]:
    """Per-scalar means and z-scores, per-histogram TV and chi-square p."""
    rows = []
    skip = block * skip_blocks
    for name in a["series"]:
        if name.startswith("_") or name not in b["series"]:
            continue
        ma, sa = _mean_se(_block_means(a["series"][name], block, skip))
        mb, sb = _mean_se(_block_means(b["series"][name], block, skip))
        se = math.sqrt(sa**2 + sb**2)
        z = (mb - ma) / se if se > 0 else (0.0 if mb == ma else float("inf"))
        rows.append({"name": name, "a": ma, "b": mb, "se": se, "z": z})
    lengths = {}
    for label, rec in (("a", a), ("b", b)):
        sums = _block_means(rec["series"]["_len_sum"], block, skip)
        counts = _block_means(rec["series"]["_len_count"], block, skip)
        lengths[label] = [s / c for s, c in zip(sums, counts) if c > 0]
    ma, sa = _mean_se(lengths["a"])
    mb, sb = _mean_se(lengths["b"])
    se = math.sqrt(sa**2 + sb**2)
    rows.append(
        {
            "name": "episode_length_at_reset",
            "a": ma,
            "b": mb,
            "se": se,
            "z": (mb - ma) / se if se > 0 else 0.0,
        }
    )
    hists = []
    for name, counts_a in a["hist"].items():
        tv, p = _chi_square_p(counts_a, b["hist"][name])
        hists.append({"name": name, "tv": tv, "p": p})
    return {"scalars": rows, "hists": hists}
