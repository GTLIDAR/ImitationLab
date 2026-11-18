from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch import Tensor


def as_float(value: Any) -> float | None:
    """Convert scalar metric values into floats for status-line logging."""
    if isinstance(value, int | float | np.floating):
        return float(value)
    if isinstance(value, Tensor) and value.ndim == 0:
        return float(value.detach().item())
    return None


def log_info(log_info_dict: dict, metrics_to_log: dict):
    # log all the keys
    for key, value in log_info_dict.items():
        if "/" in key:
            metrics_to_log.update(
                {key: (value.item() if isinstance(value, Tensor) else value)}
            )
        else:
            metrics_to_log.update(
                {
                    "Episode/" + key: (
                        value.item() if isinstance(value, Tensor) else value
                    )
                }
            )


def get_activation_class(activation_name: str) -> type[torch.nn.Module]:
    """Get activation class from activation name across agents."""
    activation_map = {
        "relu": torch.nn.ReLU,
        "elu": torch.nn.ELU,
        "tanh": torch.nn.Tanh,
        "gelu": torch.nn.GELU,
        "silu": torch.nn.SiLU,
        "swish": torch.nn.SiLU,
    }
    return activation_map.get(activation_name, torch.nn.ELU)


def discover_env_method(env: object, method_name: str) -> Callable | None:
    """Discover a callable by walking common wrapper attributes."""
    stack: list[object] = [env]
    visited: set[int] = set()

    while len(stack) > 0:
        current = stack.pop()
        obj_id = id(current)
        if obj_id in visited:
            continue
        visited.add(obj_id)

        method = getattr(current, method_name, None)
        if callable(method):
            return method

        for attr_name in ("base_env", "env", "_env", "unwrapped"):
            try:
                next_obj = getattr(current, attr_name, None)
            except Exception:
                continue
            if next_obj is None:
                continue
            if isinstance(next_obj, (list, tuple)):
                stack.extend(next_obj)
            else:
                stack.append(next_obj)
    return None
