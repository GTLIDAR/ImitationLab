"""TensorDict observation keys and feature shapes."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias, cast

import numpy as np
import torch
from tensordict import TensorDict
from torch import Tensor

ObsKey: TypeAlias = str | tuple[str, ...]
BatchKey: TypeAlias = str | tuple[str, ...]


def dedupe_keys(keys: list[BatchKey]) -> list[BatchKey]:
    """Return keys in insertion order, removing duplicates."""
    normalized = [normalize_batch_key(key) for key in keys]
    return list(dict.fromkeys(normalized))


def normalize_batch_key(key: Any) -> BatchKey:
    """Normalize key-like values to ``str`` or ``tuple[str, ...]``."""
    if isinstance(key, str):
        return key
    if isinstance(key, tuple):
        normalized_tuple = tuple(str(part) for part in key)
        if len(normalized_tuple) == 1:
            return normalized_tuple[0]
        return normalized_tuple
    if isinstance(key, Sequence):
        normalized_tuple = tuple(str(part) for part in key)
        if len(normalized_tuple) == 0:
            raise ValueError("Batch key sequence must not be empty.")
        if len(normalized_tuple) == 1:
            return normalized_tuple[0]
        return normalized_tuple
    raise TypeError(f"Unsupported key type: {type(key)!r}")


def next_obs_key(key: ObsKey) -> tuple[str, ...]:
    """Prefix an observation key with ``"next"`` preserving nested paths."""
    key = cast(ObsKey, normalize_batch_key(key))
    if isinstance(key, tuple):
        return ("next", *key)
    return ("next", key)


def strip_next_prefix(key: BatchKey) -> ObsKey:
    """Remove a leading ``"next"`` from a batch key if present."""
    key = normalize_batch_key(key)
    if isinstance(key, tuple) and len(key) > 0 and key[0] == "next":
        rest = key[1:]
        if len(rest) == 1:
            return cast(ObsKey, rest[0])
        return cast(ObsKey, rest)
    return cast(ObsKey, key)


def mapping_get_obs_value(obs: Mapping[Any, Any], key: ObsKey) -> Tensor | np.ndarray:
    """Read an observation value from a flat-or-nested mapping."""
    key = cast(ObsKey, normalize_batch_key(key))
    if key in obs:
        return cast(Tensor | np.ndarray, obs[key])
    if not isinstance(key, tuple):
        return cast(Tensor | np.ndarray, obs[key])
    cur: Any = obs
    for part in key:
        cur = cur[part]
    return cast(Tensor | np.ndarray, cur)


def flatten_feature_tensor(tensor: Tensor, feature_ndim: int) -> Tensor:
    """Flatten feature dims while keeping leading batch dims unchanged."""
    if feature_ndim <= 0:
        return tensor.unsqueeze(-1)
    if feature_ndim == 1:
        return tensor
    return tensor.flatten(start_dim=tensor.ndim - feature_ndim)


def obs_key_feature_shape(env: Any, key: ObsKey) -> tuple[int, ...]:
    """Return the unbatched feature shape registered for one observation key."""
    shape = tuple(int(dim) for dim in env.observation_spec[key].shape)
    batch_prefix = tuple(int(dim) for dim in getattr(env, "batch_size", ()))
    if batch_prefix and shape[: len(batch_prefix)] == batch_prefix:
        return shape[len(batch_prefix) :]
    return shape


def obs_key_feature_ndim(env: Any, key: ObsKey) -> int:
    """Return how many trailing dims belong to one observation sample."""
    return len(obs_key_feature_shape(env, key))


def obs_key_feature_dim(env: Any, key: ObsKey) -> int:
    """Return the flattened feature size for one observation sample."""
    shape = obs_key_feature_shape(env, key)
    return int(math.prod(shape)) if shape else 1


def flatten_obs_features_from_td(
    td: TensorDict,
    keys: Sequence[ObsKey],
    key_feature_ndims: Mapping[ObsKey, int],
    *,
    next_obs: bool,
    detach: bool,
) -> Tensor:
    """Materialize flattened observation features for one or more keys."""
    if len(keys) == 0:
        raise ValueError("Observation feature extraction requires at least one key.")

    parts: list[Tensor] = []
    for key in keys:
        obs_key = next_obs_key(key) if next_obs else key
        obs = cast(Tensor, td.get(cast(BatchKey, obs_key)))
        obs = flatten_feature_tensor(obs, int(key_feature_ndims[key]))
        parts.append(obs.detach() if detach else obs)
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)


def flatten_action_features_from_td(
    td: TensorDict,
    action_feature_ndim: int,
    *,
    detach: bool,
) -> Tensor:
    """Materialize flattened action features from a batch."""
    action = cast(Tensor, td.get("action"))
    action = flatten_feature_tensor(action, action_feature_ndim)
    return action.detach() if detach else action


def infer_batch_shape(tensor: Tensor, feature_ndim: int) -> tuple[int, ...]:
    """Infer batch shape for a tensor with known feature rank."""
    if feature_ndim <= 0:
        return tuple(int(dim) for dim in tensor.shape)
    return tuple(int(dim) for dim in tensor.shape[:-feature_ndim])


def infer_batch_shape_from_mapping(
    values: Mapping[BatchKey, Tensor],
    key_feature_ndims: Mapping[ObsKey, int],
) -> tuple[int, ...]:
    """Infer the common batch shape for a dict of observation tensors."""

    batch_shape: tuple[int, ...] | None = None
    for key, value in values.items():
        feature_ndim = int(key_feature_ndims[cast(ObsKey, key)])
        current_batch_shape = tuple(
            value.shape[:-feature_ndim] if feature_ndim > 0 else value.shape
        )
        if batch_shape is None:
            batch_shape = current_batch_shape
            continue
        if current_batch_shape != batch_shape:
            msg = (
                "Observation batch shape mismatch: "
                f"{batch_shape} vs {current_batch_shape}."
            )
            raise ValueError(msg)
    return batch_shape or (1,)


def flatten_obs_group(td: TensorDict) -> TensorDict:
    """Flatten nested ``observation`` groups to match on-policy batch layout."""
    if "observation" in td:
        obs_group = cast(TensorDict, td.pop("observation"))
        for key in list(obs_group.keys()):
            td.set(key, obs_group.get(key))
    if "next" in td:
        next_td = cast(TensorDict, td.get("next"))
        if "observation" in next_td:
            next_obs_group = cast(TensorDict, next_td.pop("observation"))
            for key in list(next_obs_group.keys()):
                next_td.set(key, next_obs_group.get(key))
    return td
