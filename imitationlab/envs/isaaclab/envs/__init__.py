"""Simulation environment classes and low-level runtime helpers."""

from importlib import import_module

__all__ = ["ImitationRLEnv"]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.imitation_rl_env"), name)
    globals()[name] = value
    return value
