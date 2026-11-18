"""Scalar validators shared by the encoder and planner configurations.

Each validator coerces ``value`` to the target type and raises ``ValueError``
naming the field when the value is out of range. The float validators also
reject non-finite values.
"""

from __future__ import annotations

import math

__all__ = [
    "require_fraction",
    "require_non_negative_float",
    "require_non_negative_int",
    "require_positive_float",
    "require_positive_int",
]


def require_positive_int(name: str, value: int) -> int:
    """Return ``int(value)``; raise when it is not > 0."""
    normalized = int(value)
    if normalized <= 0:
        msg = f"{name} must be > 0, got {value!r}."
        raise ValueError(msg)
    return normalized


def require_non_negative_int(name: str, value: int) -> int:
    """Return ``int(value)``; raise when it is < 0."""
    normalized = int(value)
    if normalized < 0:
        msg = f"{name} must be >= 0, got {value!r}."
        raise ValueError(msg)
    return normalized


def require_positive_float(name: str, value: float) -> float:
    """Return ``float(value)``; raise when it is not finite and > 0."""
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        msg = f"{name} must be > 0, got {value!r}."
        raise ValueError(msg)
    return normalized


def require_non_negative_float(name: str, value: float) -> float:
    """Return ``float(value)``; raise when it is not finite and >= 0."""
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        msg = f"{name} must be >= 0, got {value!r}."
        raise ValueError(msg)
    return normalized


def require_fraction(name: str, value: float) -> float:
    """Return ``float(value)``; raise when it is not in the open interval (0, 1)."""
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 < normalized < 1.0:
        msg = f"{name} must be in (0, 1), got {value!r}."
        raise ValueError(msg)
    return normalized
