"""Unit checks that each collection fast path returns the original tensors.

The end-to-end check (eager vs fast collection from a mature checkpoint, with
one recorded physics trace) is ``test_collect_fastpath_equivalence.py``; these
pin the individual replacements without a simulator.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict
from torchrl.data.tensor_specs import Composite, Unbounded
from torchrl.envs.transforms import VecGymEnvTransform

import imitationlab.envs.isaaclab.envs.torchrl_adapter as torchrl_adapter
from imitationlab.envs.isaaclab.envs import collect_fastpath
from imitationlab.envs.isaaclab.envs.torchrl_adapter import (
    IsaacLabTerminalObsReader,
    IsaacLabVecGymEnvTransform,
    _zero_from_spec,
    attach_terminal_obs_reader,
)
from imitationlab.envs.isaaclab.envs.expert_data_plane import (
    _WXYZ_TO_XYZW,
    _convert_reference_quats_to_xyzw,
)
from imitationlab.envs.isaaclab.tasks.manager_based.imitation.mdp.commands import timer
from imitationlab.envs.isaaclab.tasks.manager_based.imitation.mdp.commands.reference import (
    _zero_rows,
)

N = 6


def _obs_spec() -> Composite:
    return Composite(
        policy=Composite(
            a=Unbounded(shape=(N, 3)),
            b=Unbounded(shape=(N, 2, 2)),
            shape=(N,),
        ),
        critic=Composite(c=Unbounded(shape=(N, 4)), shape=(N,)),
        shape=(N,),
    )


def _assert_same_td(a: TensorDict, b: TensorDict) -> None:
    assert sorted(a.keys(True, True), key=str) == sorted(b.keys(True, True), key=str)
    for key in a.keys(True, True):
        x, y = a.get(key), b.get(key)
        assert x.dtype == y.dtype
        assert x.shape == y.shape
        assert x.device == y.device
        assert torch.equal(x, y), key
    assert a.batch_size == b.batch_size


def test_switches_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    assert collect_fastpath._parse("") == frozenset()
    assert collect_fastpath._parse("0") == frozenset()
    assert collect_fastpath._parse("all") == frozenset(collect_fastpath.FEATURES)
    assert collect_fastpath._parse("reset_gather,read_done") == {
        "reset_gather",
        "read_done",
    }
    with pytest.raises(ValueError, match="unknown fast paths"):
        collect_fastpath._parse("reset_gather,warp_speed")


def _timer_term(low_seconds: float) -> SimpleNamespace:
    calls: list[str] = []
    return SimpleNamespace(
        cfg=SimpleNamespace(resampling_time_range=(low_seconds, low_seconds)),
        time_left=torch.full((3,), 5.0),
        _update_metrics=lambda: calls.append("metrics"),
        _update_command=lambda: calls.append("command"),
        calls=calls,
    )


def test_never_resample_keeps_the_timer_decrement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(timer, "_FAST_NEVER_RESAMPLE", True)
    term = _timer_term(1.0e9)
    assert timer.compute_if_never_resampled(term, 0.5) is True
    assert term.calls == ["metrics", "command"]
    assert torch.equal(term.time_left, torch.full((3,), 4.5))


@pytest.mark.parametrize(("enabled", "low_seconds"), [(False, 1.0e9), (True, 10.0)])
def test_a_live_timer_or_the_off_switch_leaves_the_stock_compute(
    monkeypatch: pytest.MonkeyPatch, enabled: bool, low_seconds: float
) -> None:
    monkeypatch.setattr(timer, "_FAST_NEVER_RESAMPLE", enabled)
    term = _timer_term(low_seconds)
    assert timer.compute_if_never_resampled(term, 0.5) is False
    assert term.calls == []
    assert torch.equal(term.time_left, torch.full((3,), 5.0))


def test_zero_from_spec_matches_spec_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torchrl_adapter, "_FAST_SPEC_ZERO", True)
    spec = _obs_spec()
    _assert_same_td(_zero_from_spec(spec), spec.zero())
    leaf = spec["policy", "a"]
    assert torch.equal(_zero_from_spec(leaf), leaf.zero())


def _step_pair(seed: int, done_ids: list[int], published_ids: list[int]):
    """(next td, published env ids, terminal rows) as the wrapper hands them over."""
    gen = torch.Generator().manual_seed(seed)
    obs = TensorDict(
        {
            "policy": TensorDict(
                {
                    "a": torch.randn(N, 3, generator=gen),
                    "b": torch.randn(N, 2, 2, generator=gen),
                },
                batch_size=[N],
            ),
            "critic": TensorDict(
                {"c": torch.randn(N, 4, generator=gen)}, batch_size=[N]
            ),
        },
        batch_size=[N],
    )
    done = torch.zeros(N, 1, dtype=torch.bool)
    done[done_ids] = True
    next_td = obs.clone()
    next_td.set("done", done)
    next_td.set("terminated", done.clone())
    next_td.set("truncated", torch.zeros_like(done))
    ids = torch.as_tensor(published_ids, dtype=torch.long)
    k = len(published_ids)
    terminal = {
        "policy": {
            "a": torch.randn(k, 3, generator=gen),
            "b": torch.randn(k, 2, 2, generator=gen),
        },
        "critic": {"c": torch.randn(k, 4, generator=gen)},
    }
    return next_td, ids, terminal


# (done rows, rows with a published terminal observation). A done row without
# one is a reset the v2 env's visualizer requested after `_publish_terminal_obs`.
CASES = [
    ([], []),
    ([0], [0]),
    ([1, 4, 5], [1, 4, 5]),
    (list(range(N)), list(range(N))),
    ([2], []),
    ([1, 3, 4], [1, 4]),
]


@pytest.mark.parametrize(("done_ids", "published_ids"), CASES)
def test_index_based_terminal_swap_matches_the_mask_based_transform(
    done_ids, published_ids
) -> None:
    spec = _obs_spec()
    reader = IsaacLabTerminalObsReader(observation_spec=spec, backend="gymnasium")
    next_ref, ids, terminal = _step_pair(3, done_ids, published_ids)
    next_fast = next_ref.clone()
    before = next_ref.clone()

    def info() -> dict:
        # The reader pops what it consumes, so each call gets its own dict.
        if not published_ids:
            return {}
        return {"final_obs": {"_env_ids": ids, "obs": terminal}}

    # Reference: the reader scatters into full-size `final` buffers and the
    # stock transform swaps them in by mask.
    reader(info(), next_ref)
    stock = VecGymEnvTransform()
    stock.__dict__["_done_keys"] = ["done"]
    out_ref = stock._step(TensorDict({}, batch_size=[N]), next_ref)

    reader.hand_off_batched = True
    reader(info(), next_fast)
    fast = IsaacLabVecGymEnvTransform()
    fast.__dict__["_done_keys"] = ["done"]
    fast.__dict__["_terminal_reader_ref"] = reader
    out_fast = fast._step(TensorDict({}, batch_size=[N]), next_fast)

    _assert_same_td(out_fast, out_ref)
    assert torch.equal(fast._memo["done"], stock._memo["done"])
    assert reader.pending_terminal_obs is None
    # The fast path always keeps the (untouched) reset observations; the
    # stock one only on steps with a done row.
    _assert_same_td(fast._memo["saved_next"], before.select("policy", "critic"))
    if done_ids:
        _assert_same_td(fast._memo["saved_next"], stock._memo["saved_next"])
        # The partial-reset merge then gets the same reset observations.
        reset = torch.zeros(N, 1, dtype=torch.bool)
        reset[done_ids] = True
        if not bool(reset.all()):
            reset_ref = TensorDict({}, batch_size=[N])
            reset_fast = TensorDict({}, batch_size=[N])
            stock._reset(TensorDict({"_reset": reset}, batch_size=[N]), reset_ref)
            fast._reset(TensorDict({"_reset": reset}, batch_size=[N]), reset_fast)
            _assert_same_td(reset_fast, reset_ref)
    else:
        assert stock._memo["saved_next"] is None


def test_transform_refuses_to_run_without_its_reader_or_final_buffers() -> None:
    next_td, _ids, _terminal = _step_pair(5, [1], [])
    fast = IsaacLabVecGymEnvTransform()
    fast.__dict__["_done_keys"] = ["done"]
    fast.__dict__["_container"] = None  # no env: the reader cannot be found
    with pytest.raises(RuntimeError, match="no terminal-observation reader"):
        fast._step(TensorDict({}, batch_size=[N]), next_td)


def _object_final_obs(rows: dict[int, dict]) -> np.ndarray:
    """A Gymnasium object array of per-env terminal dicts (non-imitation envs)."""
    final_obs = np.empty(N, dtype=object)
    final_obs[:] = None
    for env_id, obs in rows.items():
        final_obs[env_id] = obs
    return final_obs


def test_hand_off_reader_rejects_an_object_array() -> None:
    reader = IsaacLabTerminalObsReader(
        observation_spec=_obs_spec(), backend="gymnasium"
    )
    reader.hand_off_batched = True
    next_td, _ids, _terminal = _step_pair(7, [1], [])
    row = {
        "policy": {"a": torch.ones(3), "b": torch.ones(2, 2)},
        "critic": {"c": torch.ones(4)},
    }
    with pytest.raises(RuntimeError, match="batched"):
        reader({"final_obs": _object_final_obs({1: row})}, next_td)


def test_an_object_array_keeps_the_stock_swap() -> None:
    reader = IsaacLabTerminalObsReader(
        observation_spec=_obs_spec(), backend="gymnasium"
    )
    next_td, _ids, _terminal = _step_pair(7, [1], [])
    row = {
        "policy": {"a": torch.full((3,), 9.0), "b": torch.full((2, 2), 9.0)},
        "critic": {"c": torch.full((4,), 9.0)},
    }
    reader({"final_obs": _object_final_obs({1: row})}, next_td)
    stock = VecGymEnvTransform()
    stock.__dict__["_done_keys"] = ["done"]
    out = stock._step(TensorDict({}, batch_size=[N]), next_td)
    assert torch.equal(out["policy", "a"][1], torch.full((3,), 9.0))
    assert stock._memo["saved_next"] is not None


class _FakeOuter:
    """TransformedEnv stand-in around a wrapper whose Gym env is ``isaac_env``."""

    def __init__(self, isaac_env: object) -> None:
        self.observation_spec = _obs_spec()
        self.transform = VecGymEnvTransform()
        self.readers: list[object] = []
        inner = type("Wrapper", (), {})()
        inner.__dict__["_env"] = type("Gym", (), {"unwrapped": isaac_env})()
        self.base_env = inner

    def set_info_dict_reader(self, reader):
        self.readers.append(reader)
        return self.base_env


def test_hand_off_is_installed_only_for_the_imitation_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from isaaclab.envs import ManagerBasedRLEnv

    from imitationlab.envs.isaaclab.envs import ImitationRLEnv

    monkeypatch.setattr(torchrl_adapter, "_FAST_TERMINAL_SWAP", True)
    other = _FakeOuter(ManagerBasedRLEnv.__new__(ManagerBasedRLEnv))
    attach_terminal_obs_reader(other)
    assert type(other.transform) is VecGymEnvTransform
    assert other.readers[0].hand_off_batched is False

    imitation = _FakeOuter(ImitationRLEnv.__new__(ImitationRLEnv))
    attach_terminal_obs_reader(imitation)
    assert type(imitation.transform) is IsaacLabVecGymEnvTransform
    assert imitation.readers[0].hand_off_batched is True

    monkeypatch.setattr(torchrl_adapter, "_FAST_TERMINAL_SWAP", False)
    off = _FakeOuter(ImitationRLEnv.__new__(ImitationRLEnv))
    attach_terminal_obs_reader(off)
    assert type(off.transform) is VecGymEnvTransform
    assert off.readers[0].hand_off_batched is False


def _devices() -> list[str]:
    return ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


@pytest.mark.parametrize("device", _devices())
def test_quat_reorder_equals_the_python_list_index(device: str) -> None:
    quat = torch.randn(5, 3, 4, device=device)
    nested = torch.randn(5, 3, 4, device=device)
    reference = TensorDict(
        {"root_quat": quat.clone(), "next": {"xquat": nested.clone()}},
        batch_size=[5],
        device=device,
    )
    converted = _convert_reference_quats_to_xyzw(reference)
    for got, source in (
        (converted.get("root_quat"), quat),
        (converted.get(("next", "xquat")), nested),
    ):
        expected = source[..., _WXYZ_TO_XYZW]
        assert torch.equal(got, expected)
        assert got.stride() == expected.stride() and got.dtype == expected.dtype


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize(
    "rows",
    [torch.tensor([3, 0]), slice(None), slice(1, 3), [2, 4]],
    ids=["tensor", "all", "slice", "list"],
)
def test_zero_rows_equals_the_scalar_index_write(device: str, rows) -> None:
    rows = rows.to(device) if isinstance(rows, torch.Tensor) else rows
    expected = torch.arange(1.0, 6.0, device=device)
    got = expected.clone()
    expected[rows] = 0.0
    _zero_rows(got, rows)
    assert torch.equal(got, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_reference_reset_writes_never_wait_for_the_device() -> None:
    quat = torch.randn(5, 4, device="cuda")
    reference = TensorDict({"root_quat": quat}, batch_size=[5], device="cuda")
    _convert_reference_quats_to_xyzw(reference.clone())  # the one index upload
    rows = torch.tensor([3, 0], device="cuda")
    buffer = torch.ones(5, device="cuda")
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        _convert_reference_quats_to_xyzw(reference)
        _zero_rows(buffer, rows)
    finally:
        torch.cuda.set_sync_debug_mode("default")
