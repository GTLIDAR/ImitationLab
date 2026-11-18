"""Find the Isaac Lab imitation env under TorchRL and Gym wrappers."""

from __future__ import annotations

from typing import Any


def imitation_env_classes() -> tuple[type, ...]:
    """The imitation env base classes (``ImitationRLEnv`` backs ``imitation-g1-v1``).

    The import is deferred because these modules need a running simulator.
    """
    from imitationlab.envs.isaaclab.envs import ImitationRLEnv

    return (ImitationRLEnv,)


def unwrap_imitation_env(env: object) -> Any:
    """Return the imitation env inside ``env`` or raise ``TypeError``."""
    classes = imitation_env_classes()
    current = env
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, classes):
            return current
        unwrapped = getattr(current, "unwrapped", None)
        if isinstance(unwrapped, classes):
            return unwrapped
        current = (
            getattr(current, "base_env", None)
            or getattr(current, "env", None)
            or getattr(current, "_env", None)
        )
    raise TypeError(
        "Could not unwrap an imitation env; expected one of "
        f"{[cls.__name__ for cls in classes]}."
    )
