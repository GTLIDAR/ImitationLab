"""Contract for the env code that follows the Isaac Lab ``develop`` base classes.

Two things are pinned here, all without a simulator:

1. The id helpers pass device index tensors through. Isaac Lab finalizes
   ``SceneEntityCfg.joint_ids`` and ``body_ids`` to device long tensors, and a
   per-element host read of such a tensor would sync the device on every call.
2. The reset-mode events and ``randomize_joint_default_pos`` accept tensor ids
   and slices without copy-construct warnings.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from isaaclab.managers import SceneEntityCfg

from imitationlab.envs.isaaclab.envs import (
    index_keys,
)
from imitationlab.envs.isaaclab.envs.expert_data_plane import ExpertDataPlane
from imitationlab.envs.isaaclab.envs.index_keys import index_key
from imitationlab.envs.isaaclab.tasks.manager_based.imitation.mdp import events

NUM_ENVS = 4


# ---------------------------------------------------------------------------
# index_key
# ---------------------------------------------------------------------------


def test_index_key_is_the_same_for_a_list_and_a_tensor_of_equal_content() -> None:
    assert index_key([3, 1, 2]) == (3, 1, 2)
    assert index_key(torch.tensor([3, 1, 2])) == (3, 1, 2)
    assert index_key(torch.tensor([3, 1, 2], dtype=torch.int32)) == (3, 1, 2)


def test_index_key_reads_each_tensor_once() -> None:
    ids = torch.tensor([5, 6, 7])
    first = index_key(ids)
    # The cached tuple is returned as is; a new read would build a new tuple.
    assert index_key(ids) is first


def test_index_key_never_serves_a_freed_tensors_key_to_a_new_tensor() -> None:
    # CPython reuses the id of a freed object, so the table must prove that an
    # id still names the same tensor.
    for value in range(200):
        assert index_key(torch.tensor([value])) == (value,)


def test_index_key_table_stays_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(index_keys, "_MAX_ENTRIES", 8)
    keep = [torch.tensor([value]) for value in range(32)]
    for ids in keep:
        index_key(ids)
    assert len(index_keys._KEYS) <= 8
    assert [index_key(ids) for ids in keep] == [(value,) for value in range(32)]


# ---------------------------------------------------------------------------
# id helpers
# ---------------------------------------------------------------------------


def _plane(device: str = "cpu") -> Any:
    """A data plane with only the members the id helpers read."""
    plane: Any = ExpertDataPlane.__new__(ExpertDataPlane)
    plane._env = SimpleNamespace(device=device)
    # `_ensure_mdp_fast_paths` returns early once this attribute exists.
    plane._mdp_cache_step = -1
    plane._mdp_body_id_tensor_cache = {}
    plane._mdp_joint_id_tensor_cache = {}
    plane._mdp_expert_motion_cache = {}
    plane._mdp_all_body_ids_key = tuple(range(6))
    return plane


def test_id_helpers_pass_a_long_tensor_through_unchanged() -> None:
    plane = _plane()
    ids = torch.tensor([4, 2, 0], dtype=torch.long)
    assert plane._get_body_ids_tensor_fast(ids) is ids
    assert plane._get_joint_ids_tensor_fast(ids) is ids
    # A tensor must not enter the host-list caches.
    assert not plane._mdp_body_id_tensor_cache
    assert not plane._mdp_joint_id_tensor_cache


def test_id_helpers_convert_an_int32_tensor_and_keep_slices() -> None:
    plane = _plane()
    ids = torch.tensor([4, 2, 0], dtype=torch.int32)
    for helper in (plane._get_body_ids_tensor_fast, plane._get_joint_ids_tensor_fast):
        converted = helper(ids)
        assert isinstance(converted, torch.Tensor)
        assert converted.dtype == torch.long
        assert converted.tolist() == [4, 2, 0]
        assert helper(slice(None)) == slice(None)


def test_id_helpers_still_cache_a_host_list() -> None:
    plane = _plane()
    first = plane._get_joint_ids_tensor_fast([1, 2, 3])
    assert isinstance(first, torch.Tensor)
    assert first.dtype == torch.long
    assert plane._get_joint_ids_tensor_fast([1, 2, 3]) is first


def test_expert_motion_command_shares_one_result_across_equal_tensors() -> None:
    plane = _plane()
    plane._ensure_mdp_step_cache = lambda: None
    plane.current_expert_frame = {
        "joint_pos": torch.arange(12.0).reshape(2, 6),
        "joint_vel": -torch.arange(12.0).reshape(2, 6),
    }
    ids_a = torch.tensor([5, 1, 3])
    ids_b = torch.tensor([5, 1, 3])
    from_list = plane._get_expert_motion_command_fast([5, 1, 3])
    assert plane._get_expert_motion_command_fast(ids_a) is from_list
    assert plane._get_expert_motion_command_fast(ids_b) is from_list
    expected = torch.cat(
        [
            plane.current_expert_frame["joint_pos"][:, [5, 1, 3]],
            plane.current_expert_frame["joint_vel"][:, [5, 1, 3]],
        ],
        dim=-1,
    )
    assert torch.equal(from_list, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_device_tensor_ids_cost_no_host_sync_after_the_first_read() -> None:
    plane = _plane("cuda:0")
    plane._ensure_mdp_step_cache = lambda: None
    plane.current_expert_frame = {
        "joint_pos": torch.zeros(2, 6, device="cuda:0"),
        "joint_vel": torch.zeros(2, 6, device="cuda:0"),
    }
    ids = torch.tensor([5, 1, 3], dtype=torch.long, device="cuda:0")
    plane._get_expert_motion_command_fast(ids)  # reads the tensor once
    torch.cuda.set_sync_debug_mode("error")
    try:
        assert plane._get_body_ids_tensor_fast(ids) is ids
        assert plane._get_joint_ids_tensor_fast(ids) is ids
        plane._get_expert_motion_command_fast(ids)
        assert plane._body_ids_cache_key(ids) == (5, 1, 3)
        assert plane._joint_ids_cache_key(ids) == (5, 1, 3)
    finally:
        torch.cuda.set_sync_debug_mode("default")


# ---------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        (None, [0, 1, 2, 3]),
        (slice(2, None), [2, 3]),
        (torch.tensor([1, 3], dtype=torch.int32), [1, 3]),
    ],
)
def test_event_env_selector_becomes_long_indices(selector, expected) -> None:
    ids = events._as_env_index_tensor(selector, NUM_ENVS, "cpu")
    assert ids.dtype == torch.long
    assert ids.tolist() == expected


class _Scene:
    """The scene surface the joint-default event reads."""

    def __init__(self, asset, num_envs: int) -> None:
        self._asset = asset
        self.num_envs = num_envs

    def __getitem__(self, name: str):
        return self._asset


def _joint_default_env(num_joints: int = 5) -> tuple[Any, torch.Tensor, torch.Tensor]:
    default_joint_pos = torch.zeros(NUM_ENVS, num_joints)
    asset = SimpleNamespace(
        device="cpu",
        data=SimpleNamespace(
            default_joint_pos=SimpleNamespace(torch=default_joint_pos)
        ),
    )
    offset = torch.zeros(NUM_ENVS, 2)
    action_term = SimpleNamespace(
        _offset=offset, _joint_ids=torch.tensor([4, 1], dtype=torch.long)
    )
    env: Any = SimpleNamespace(
        scene=_Scene(asset, NUM_ENVS),
        action_manager=SimpleNamespace(get_term=lambda name: action_term),
    )
    return env, default_joint_pos, offset


@pytest.mark.parametrize(
    "joint_ids",
    [
        slice(None),
        torch.tensor([1, 4], dtype=torch.long),
        [1, 4],
    ],
)
def test_randomize_joint_default_pos_accepts_slice_tensor_and_list_ids(
    joint_ids,
) -> None:
    env, default_joint_pos, offset = _joint_default_env()
    cfg = SceneEntityCfg("robot")
    cfg.joint_ids = joint_ids
    with warnings.catch_warnings():
        # `torch.tensor(<tensor>)` warns; the event must share the tensor.
        warnings.simplefilter("error", UserWarning)
        events.randomize_joint_default_pos(
            env,
            None,
            cfg,
            pos_distribution_params=(0.25, 0.25),
            operation="abs",
        )
    touched = list(range(5)) if isinstance(joint_ids, slice) else [1, 4]
    assert torch.all(default_joint_pos[:, touched] == 0.25)
    untouched = [i for i in range(5) if i not in touched]
    assert torch.all(default_joint_pos[:, untouched] == 0.0)
    # The action offset follows the term's own pinned joints (4 then 1).
    assert torch.all(offset == 0.25)
