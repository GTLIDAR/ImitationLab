"""Policy learners for ImitationLab."""

from importlib import import_module

_EXPORTS = {
    "PPO": ".ppo",
    "PPOTrainerConfig": ".ppo",
    "SuperMimic": ".supermimic",
    "SVG": ".svg",
    "DiffSR": ".diffsr",
    "WMBPTT": ".common.world_model.learner",
    "WMBPTTTrainerConfig": ".common.world_model.config",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
