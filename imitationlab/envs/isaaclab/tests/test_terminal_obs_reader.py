"""Regression tests for the terminal-observation plumbing of ``IsaacLabWrapper``.

``IsaacLabWrapper(env)`` returns ``TransformedEnv(wrapper, VecGymEnvTransform)``.
``set_info_dict_reader`` is forwarded to the inner wrapper and returns it, so
assigning its return value dropped the transform. TorchRL then zero-filled the
first observation of every episode after a partial reset.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from imitationlab.envs.isaaclab.envs.torchrl_adapter import (
    IsaacLabTerminalObsReader,
    attach_terminal_obs_reader,
)


class _Inner:
    def __init__(self) -> None:
        self.readers: list[object] = []

    def set_info_dict_reader(self, reader):
        self.readers.append(reader)
        return self  # the real method returns the INNER env


class _Outer:
    """Stand-in for TransformedEnv: forwards unknown attributes to the inner env."""

    def __init__(self, inner: _Inner) -> None:
        self.base_env = inner
        self.observation_spec = None

    def __getattr__(self, name: str):
        return getattr(self.base_env, name)


def test_attach_returns_outer_env_and_registers_reader(monkeypatch):
    made = []

    class _Reader:
        def __init__(self, observation_spec, backend):
            made.append((observation_spec, backend))

    monkeypatch.setattr(
        "imitationlab.envs.isaaclab.envs.torchrl_adapter.IsaacLabTerminalObsReader",
        _Reader,
    )
    inner = _Inner()
    outer = _Outer(inner)
    assert attach_terminal_obs_reader(outer) is outer
    assert len(inner.readers) == 1
    assert made == [(None, "gymnasium")]


def test_extract_nested_value_walks_tensordict_groups():
    obs = TensorDict(
        {"policy": TensorDict({"a": torch.ones(2, 3)}, batch_size=[2])}, batch_size=[2]
    )
    got = IsaacLabTerminalObsReader._extract_nested_value(obs, ("policy", "a"))
    assert isinstance(got, torch.Tensor) and got.shape == (2, 3)
    assert IsaacLabTerminalObsReader._extract_nested_value(obs, ("policy", "x")) is None


def test_extract_nested_value_walks_plain_dicts():
    obs = {"policy": {"a": torch.zeros(1, 2)}}
    got = IsaacLabTerminalObsReader._extract_nested_value(obs, ("policy", "a"))
    assert got is obs["policy"]["a"]


class _FakeWrapper:
    """Minimal ``self`` for ``IsaacLabWrapper._output_transform``."""

    def __init__(self, readers: list[object] | None = None) -> None:
        self.log_infos: list[object] = []
        self.info_dict_reader = readers or []

    def _log_payload(self, info):
        return dict(info)

    def _terminal_swap_is_out_of_place(self) -> bool:
        from imitationlab.envs.isaaclab.envs.torchrl_adapter import IsaacLabWrapper

        return IsaacLabWrapper._terminal_swap_is_out_of_place(self)


def _step_tuple(obs_buf, with_reset: bool):
    n = obs_buf["policy"]["a"].shape[0]
    info: dict[str, object] = {"log": {}}
    if with_reset:
        info["final_obs"] = {"_env_ids": torch.tensor([0]), "obs": obs_buf}
    return (
        obs_buf,
        torch.zeros(n),
        torch.zeros(n, dtype=torch.bool),
        torch.zeros(n, dtype=torch.bool),
        info,
    )


def test_reset_step_observations_do_not_alias_env_obs_buf():
    """VecGymEnvTransform's in-place terminal swap must not reach ``obs_buf``.

    Regression: the swap wrote the terminal observation into the env's
    ``obs_buf``; an episode that ended on its first step then published the
    previous episode's terminal observation as its own.
    """
    from imitationlab.envs.isaaclab.envs.torchrl_adapter import IsaacLabWrapper

    obs_buf = {"policy": {"a": torch.arange(6.0).reshape(2, 3)}}
    out = IsaacLabWrapper._output_transform(_FakeWrapper(), _step_tuple(obs_buf, True))
    returned = out[0]["policy"]["a"]
    assert torch.equal(returned, obs_buf["policy"]["a"])
    returned[0] = -1.0  # what the transform does to done rows
    assert torch.equal(obs_buf["policy"]["a"], torch.arange(6.0).reshape(2, 3))


def test_reset_step_skips_the_copy_when_the_swap_is_out_of_place():
    """With the hand-off reader, `IsaacLabVecGymEnvTransform` swaps by copy.

    It never writes into the step's observation tensors, so the wrapper
    returns ``obs_buf`` itself instead of a defensive copy.
    """
    from imitationlab.envs.isaaclab.envs.torchrl_adapter import IsaacLabWrapper

    reader = IsaacLabTerminalObsReader.__new__(IsaacLabTerminalObsReader)
    reader.hand_off_batched = True
    obs_buf = {"policy": {"a": torch.arange(6.0).reshape(2, 3)}}
    out = IsaacLabWrapper._output_transform(
        _FakeWrapper([reader]), _step_tuple(obs_buf, True)
    )
    assert out[0] is obs_buf
    assert "final_obs" in out[5]


def test_non_reset_step_returns_obs_buf_without_copy():
    from imitationlab.envs.isaaclab.envs.torchrl_adapter import IsaacLabWrapper

    obs_buf = {"policy": {"a": torch.zeros(2, 3)}}
    out = IsaacLabWrapper._output_transform(_FakeWrapper(), _step_tuple(obs_buf, False))
    assert out[0] is obs_buf
