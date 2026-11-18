"""Imagined BPTT policy learning."""

from importlib import import_module

_EXPORTS = {
    "SuperMimic": ".learner",
    "WMBPTT": "imitationlab.algorithms.common.world_model.learner",
    "WMBPTTConfig": "imitationlab.algorithms.common.world_model.config",
    "WMBPTTTrainerConfig": "imitationlab.algorithms.common.world_model.config",
    "WorldModelTask": "imitationlab.algorithms.common.world_model.task",
    "load_task": "imitationlab.algorithms.common.world_model.task",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
