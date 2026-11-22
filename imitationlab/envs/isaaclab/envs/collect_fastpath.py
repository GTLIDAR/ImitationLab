"""Switches for the rollout-collection fast paths (opt-in).

Each fast path replaces host-side work on the control-step path (Python
dispatch, host synchronizations, redundant copies) with code that produces the
same tensors bit for bit: same values, dtypes, shapes, devices, and random-number
consumption. ``IMITATIONLAB_ENV_COLLECT_FASTPATH`` selects them:

- unset, empty, ``0`` or ``none``: the original code paths (the default);
- ``1`` or ``all``: every fast path;
- a comma-separated list of the names in :data:`FEATURES`: only those, for
  bisecting.

The collection equivalence test runs ``0`` and ``1`` against one recorded
physics trace and compares every rollout tensor. The value is read once, at
import, so set it before the process starts. :func:`describe` is the resolved
state that entrypoints log and record.
"""

from __future__ import annotations

import logging
import os

ENV_VAR = "IMITATIONLAB_ENV_COLLECT_FASTPATH"

FEATURES: tuple[str, ...] = (
    # tracking_reward_points: cached offsets tensor instead of a per-step
    # pageable host-to-device copy (a full host synchronization).
    "reward_offsets",
    # Terminal-observation reader: zero buffers without TensorDict.from_dict.
    "spec_zero",
    # Reset reference rows: one fused gather of the packed resident backing.
    "reset_gather",
    # Index-based, out-of-place terminal-observation swap (no mask gathers,
    # no defensive observation copies). ImitationRLEnv only.
    "terminal_swap",
    # Terminal observations: one gather instead of gather + clone.
    "terminal_gather",
    # Per-step memo of single-frame reference command components.
    "component_memo",
    # No host sync for the one-frame frame-skip loop's break flag.
    "read_done",
    # Reference command buffer refreshed on read (its property), not on every
    # command compute.
    "lazy_command",
    # Reference rows (next-frame prefetch and reset rows) gathered by a CUDA
    # kernel straight from the host-mapped packed backing, with device
    # indices: no host index copy, no CPU worker, no host wait.
    "device_gather",
    # Command terms whose resample timer is effectively never (1e9 s) skip
    # the per-step `(time_left <= 0).nonzero()` check.
    "never_resample",
    # Reset-sampler return credit by a device mask over the reset rows, not by selecting
    # the completed rows on the host.
    "masked_credit",
)


def _parse(value: str) -> frozenset[str]:
    text = value.strip().lower()
    if text in {"", "0", "false", "no", "off", "none"}:
        return frozenset()
    if text in {"1", "true", "yes", "on", "all"}:
        return frozenset(FEATURES)
    names = {name.strip() for name in text.split(",") if name.strip()}
    unknown = names - set(FEATURES)
    if unknown:
        raise ValueError(
            f"{ENV_VAR} names unknown fast paths {sorted(unknown)}; "
            f"known: {', '.join(FEATURES)}."
        )
    return frozenset(names)


ENABLED: frozenset[str] = _parse(os.environ.get(ENV_VAR, ""))
"""Names of the active fast paths."""


def fastpath_enabled(feature: str) -> bool:
    """Whether the named collection fast path is active."""
    if feature not in FEATURES:
        raise KeyError(f"Unknown collection fast path {feature!r}.")
    return feature in ENABLED


def describe() -> dict[str, object]:
    """Resolved switch state, for logs and recorded run parameters."""
    return {
        "env_var": ENV_VAR,
        "value": os.environ.get(ENV_VAR, ""),
        "enabled": [name for name in FEATURES if name in ENABLED],
    }


if ENABLED:
    # Warning level so it reaches the job log without logging configuration.
    logging.getLogger(__name__).warning(
        "Environment collection fast paths ON (%s=%r): %s",
        ENV_VAR,
        os.environ.get(ENV_VAR, ""),
        ", ".join(describe()["enabled"]),
    )


__all__ = ["ENABLED", "ENV_VAR", "FEATURES", "describe", "fastpath_enabled"]
