"""Reset-time episode logs, recorded the way a training run reports them.

On every step with a reset, Isaac Lab rebuilds ``extras["log"]``:
``Episode_Reward/<term>`` is the mean over the environments reset at that step
of the term's weighted episode sum divided by ``max_episode_length_s``, and
command terms add ``Metrics/<term>/<name>`` the same way. Training logs one such
snapshot per training iteration. :class:`ResetLogRecorder` keeps every
snapshot of a fixed-length rollout so an evaluation can report the same
quantities as a training curve, averaged once the reset sampler has settled.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

RESET_LOG_PREFIXES = ("Episode_Reward/", "Episode_Termination/", "Metrics/")


class ResetLogRecorder:
    """Collect ``extras["log"]`` snapshots on reset steps; summarize per key."""

    def __init__(self, prefixes: tuple[str, ...] = RESET_LOG_PREFIXES) -> None:
        self.prefixes = tuple(prefixes)
        self.steps: list[int] = []
        self.resets: list[int] = []
        self.values: dict[str, list[tuple[int, float]]] = {}

    def record(self, step: int, log: Mapping[str, Any], reset_count: int) -> None:
        """Store one snapshot; call only on steps where ``reset_count > 0``."""
        keys = [key for key in log if key.startswith(self.prefixes)]
        if not keys:
            return
        values = torch.stack(
            [torch.as_tensor(log[key]).float().reshape(()).cpu() for key in keys]
        ).tolist()
        for key, value in zip(keys, values):
            self.values.setdefault(key, []).append((int(step), float(value)))
        self.steps.append(int(step))
        self.resets.append(int(reset_count))

    def summary(self, total_steps: int, *, bins: int = 20) -> dict[str, Any]:
        """Per key: mean over the second half of the rollout and binned means."""
        settled_from = int(total_steps) // 2
        width = max(1, -(-int(total_steps) // int(bins)))
        out: dict[str, Any] = {}
        for key, pairs in self.values.items():
            settled = [v for s, v in pairs if s >= settled_from]
            sums = [0.0] * bins
            counts = [0] * bins
            for s, v in pairs:
                index = min(s // width, bins - 1)
                sums[index] += v
                counts[index] += 1
            out[key] = {
                "settled_mean": (
                    sum(settled) / len(settled) if settled else float("nan")
                ),
                "settled_snapshots": len(settled),
                "bin_means": [
                    sums[i] / counts[i] if counts[i] else float("nan")
                    for i in range(bins)
                ],
            }
        return {
            "settled_from_step": settled_from,
            "snapshots": len(self.steps),
            "keys": out,
        }


__all__ = ["RESET_LOG_PREFIXES", "ResetLogRecorder"]
