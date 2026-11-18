"""Cache keys for body and joint index selections.

Isaac Lab finalizes a ``SceneEntityCfg`` so that a term call receives device
``torch.long`` tensors for ``joint_ids`` and ``body_ids``. The MDP fast paths
cache their results per step under the content of the selection, so two terms
with equal selections share one result.

A content key built with ``tuple(int(i) for i in ids)`` reads the device once
per element. This module reads each tensor once with a single ``tolist`` and
remembers the key for that tensor object. Later calls with the same tensor
cost one dictionary lookup and no device sync.
"""

from __future__ import annotations

import weakref
from collections.abc import Sequence

import torch

# Bound the table in case a caller builds a new index tensor on every call.
_MAX_ENTRIES = 1024

# tensor id -> (weak reference that proves the id still names the same object,
# content key). The weak reference lets the tensor be freed.
_KEYS: dict[int, tuple[weakref.ReferenceType[torch.Tensor], tuple[int, ...]]] = {}


def index_key(ids: Sequence[int] | torch.Tensor) -> tuple[int, ...]:
    """Return the content key of an index list or index tensor.

    A tensor is read from its device once. Isaac Lab shares one tensor between
    all calls of a term and never changes it in place, so the key stays valid
    for the lifetime of the tensor object.
    """
    if not isinstance(ids, torch.Tensor):
        return tuple(int(index) for index in ids)
    entry = _KEYS.get(id(ids))
    if entry is not None and entry[0]() is ids:
        return entry[1]
    if len(_KEYS) >= _MAX_ENTRIES:
        _KEYS.clear()
    key = tuple(int(index) for index in ids.reshape(-1).tolist())
    _KEYS[id(ids)] = (weakref.ref(ids), key)
    return key
