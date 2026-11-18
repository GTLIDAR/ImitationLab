"""Switch for the rollout-collection fast paths (opt-in).

A fast path removes host-side or dead work from the collection step and
leaves every rollout tensor bit-identical (values, dtypes, shapes, random
numbers). ``IMITATIONLAB_COLLECT_FASTPATH=1`` (or ``all``) enables them;
unset, empty or ``0`` keeps the original code paths, which is the default and
what collection equivalence tests compare against. The historical name
``RLOPT_COLLECT_FASTPATH`` is still read when the new name is unset.

The value is read once, at import, so set it before the process starts.
:func:`describe` is the resolved state, recorded in the W&B run config.
"""

from __future__ import annotations

import logging
import os

ENV_VAR = "IMITATIONLAB_COLLECT_FASTPATH"
LEGACY_ENV_VAR = "RLOPT_COLLECT_FASTPATH"


def _source_env_var() -> str:
    """The variable that sets the switch: the new name, else the legacy one."""
    if ENV_VAR in os.environ or LEGACY_ENV_VAR not in os.environ:
        return ENV_VAR
    return LEGACY_ENV_VAR


def _read_switch() -> bool:
    name = _source_env_var()
    value = os.environ.get(name, "").strip().lower()
    if value in {"", "0", "false", "no", "off", "none"}:
        return False
    if value in {"1", "true", "yes", "on", "all"}:
        return True
    msg = f"{name} must be 0/1 (or off/on, none/all); got {value!r}."
    raise ValueError(msg)


COLLECT_FASTPATH: bool = _read_switch()
"""Whether the exactness-preserving collection fast paths are active."""


def describe() -> dict[str, object]:
    """Resolved switch state, for logs and recorded run parameters."""
    name = _source_env_var()
    return {
        "env_var": name,
        "value": os.environ.get(name, ""),
        "enabled": COLLECT_FASTPATH,
    }


if COLLECT_FASTPATH:
    # Warning level so it reaches the job log without logging configuration.
    logging.getLogger(__name__).warning(
        "Collection fast paths ON (%s=%r): hl-skill anchor-encoder skip",
        _source_env_var(),
        os.environ.get(_source_env_var(), ""),
    )


__all__ = ["COLLECT_FASTPATH", "ENV_VAR", "LEGACY_ENV_VAR", "describe"]
