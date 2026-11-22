"""Unit tests of the graph-step building blocks (no simulator)."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from imitationlab.envs.isaaclab.envs.graph_step.reference_frame import (
    ReferenceFrameState,
    masked_row_gather_,
)

CUDA = torch.cuda.is_available()


@pytest.mark.skipif(not CUDA, reason="CUDA required")
def test_host_mapped_gather_equals_cpu_gather() -> None:
    from imitationlab.envs.isaaclab.envs.host_mapped import HostMappedTensor

    host = torch.randn(10_000, 37)
    mapped = HostMappedTensor(host, "cuda")
    try:
        idx = torch.randint(0, 10_000, (777,), device="cuda")
        got = torch.index_select(mapped.device_view, 0, idx).cpu()
        assert torch.equal(got, host[idx.cpu()])
    finally:
        mapped.close()


@pytest.mark.parametrize("device", ["cpu", "cuda"] if CUDA else ["cpu"])
def test_masked_row_gather_writes_only_masked_rows(device: str) -> None:
    src = torch.randn(500, 21, device=device)
    dst = torch.full((64, 21), -7.0, device=device)
    idx = torch.randint(0, 500, (64,), device=device)
    mask = torch.rand(64, device=device) < 0.3
    expected = torch.where(mask[:, None], src[idx], dst)
    masked_row_gather_(dst, src, idx, mask)
    assert torch.equal(dst, expected)


def _frame(num_envs: int = 5) -> tuple[ReferenceFrameState, dict]:
    shapes = {
        "qpos": (36,),
        "qvel": (35,),
        "body_pos_w": (3, 3),
        "body_quat_w": (3, 4),
    }
    order = tuple(shapes)
    # 29 target joints; the reference lacks two of them (NaN-filled).
    ref_to_target = torch.tensor([i for i in range(29) if i not in (4, 17)])
    target_to_ref = torch.arange(27)
    template = torch.full((29,), float("nan"))
    state = ReferenceFrameState(
        num_envs=num_envs,
        device=torch.device("cpu"),
        field_shapes=shapes,
        field_order=order,
        nan_joint_template=template,
        ref_to_target_map=ref_to_target,
        target_to_ref_map=target_to_ref,
    )
    maps = {
        "ref_to_target": ref_to_target,
        "target_to_ref": target_to_ref,
        "template": template,
    }
    return state, maps


def test_derived_fields_match_the_eager_construction() -> None:
    state, maps = _frame()
    state.packed.copy_(torch.randn_like(state.packed))
    state.refresh_derived()
    qpos, qvel = state.raw["qpos"], state.raw["qvel"]
    joint_pos = maps["template"].expand(5, 29).clone()
    joint_pos[..., maps["ref_to_target"]] = qpos[:, 7:].index_select(
        -1, maps["target_to_ref"]
    )
    frame = state.frame
    assert torch.equal(
        torch.nan_to_num(frame["joint_pos"], nan=9.0),
        torch.nan_to_num(joint_pos, nan=9.0),
    )
    assert torch.equal(frame["root_quat"], qpos[:, 3:7][..., [1, 2, 3, 0]])
    assert torch.equal(
        frame["body_quat_w"], state.raw["body_quat_w"][..., [1, 2, 3, 0]]
    )
    assert torch.equal(frame["root_pos"], qpos[:, :3])
    assert torch.equal(frame["root_ang_vel"], qvel[:, 3:6])
    assert frame["qpos"].data_ptr() == qpos.data_ptr()


def test_adopt_round_trips_an_eager_frame_and_rejects_a_wrong_one() -> None:
    source, _ = _frame()
    source.packed.copy_(torch.randn_like(source.packed))
    source.refresh_derived()
    eager = TensorDict(
        {key: value.clone() for key, value in source.frame.items()},
        batch_size=[5],
    )
    target, _ = _frame()
    target.adopt(eager)
    assert torch.equal(target.packed, source.packed)
    eager.set("root_pos", eager.get("root_pos") + 1.0)
    with pytest.raises(RuntimeError, match="mismatched"):
        _frame()[0].adopt(eager)


def test_select_into_keeps_masked_rows() -> None:
    state, _ = _frame()
    state.packed.copy_(torch.randn_like(state.packed))
    before = state.packed.clone()
    other = torch.randn_like(state.packed)
    keep = torch.tensor([True, False, True, False, False])
    state.select_into(keep, other)
    assert torch.equal(state.packed[keep], before[keep])
    assert torch.equal(state.packed[~keep], other[~keep])


def _curriculum_env(terms: list, *, steps: int, num_envs: int = 100):
    from types import SimpleNamespace

    manager = SimpleNamespace(
        _term_names=[f"t{i}" for i in range(len(terms))],
        _term_cfgs=terms,
        active_terms=[f"t{i}" for i in range(len(terms))],
    )
    return SimpleNamespace(
        curriculum_manager=manager,
        common_step_counter=steps,
        num_envs=num_envs,
        cfg=SimpleNamespace(data_parallel_world_size=1),
    )


def test_capture_never_starts_under_a_curriculum() -> None:
    """``curriculum_done`` reads the manager's term list (no get_term_cfg)."""
    from types import SimpleNamespace

    from imitationlab.envs.isaaclab.envs.graph_step.graph import GraphStepper

    # No curriculum terms: capture is never held back by it.
    assert GraphStepper.curriculum_done(
        SimpleNamespace(env=_curriculum_env([], steps=0))
    )
    # Any curriculum term holds capture back indefinitely.
    term = SimpleNamespace(func=lambda *a, **k: None, params={})
    assert not GraphStepper.curriculum_done(
        SimpleNamespace(env=_curriculum_env([term], steps=10**9))
    )


def test_graph_step_names_the_fast_paths_capture_needs(monkeypatch) -> None:
    """The fast paths are opt-in; without them capture would die mid-step."""
    from imitationlab.envs.isaaclab.envs.graph_step import graph
    from imitationlab.envs.isaaclab.envs.graph_step.masked_step import (
        UnsupportedMaskedStep,
    )

    monkeypatch.setattr(
        graph, "fastpath_enabled", lambda name: name == "reward_offsets"
    )
    with pytest.raises(UnsupportedMaskedStep) as caught:
        graph.GraphStepper.validate(object())
    message = str(caught.value)
    assert "device_gather" in message and "never_resample" in message
    assert "reward_offsets" not in message
    assert "IMITATIONLAB_ENV_COLLECT_FASTPATH=1" in message


def _log_stepper(term_names: list[str], dones: torch.Tensor):
    from types import SimpleNamespace

    env = SimpleNamespace(
        reward_manager=SimpleNamespace(_term_names=[]),
        termination_manager=SimpleNamespace(
            _term_names=term_names,
            active_terms=term_names,
            get_term_cfg=lambda name: SimpleNamespace(time_out=name == "time_out"),
            _last_episode_dones=dones,
            _class_term_cfgs=[],
        ),
        command_manager=SimpleNamespace(
            _terms={
                "reference": SimpleNamespace(
                    metrics={"mpjpe_l_mm": None, "mpjpe_g_mm": None}
                )
            }
        ),
    )
    return SimpleNamespace(env=env, device=torch.device("cpu"))


def test_masked_step_publishes_the_contract_entries() -> None:
    """The persistent log carries what ``required_log_entries`` promises."""
    from imitationlab.envs.isaaclab.contracts.training_log import (
        REQUIRED_ENV_LOG_ENTRIES,
        TRACKING_FAILED_ENTRY,
    )
    from imitationlab.envs.isaaclab.envs.graph_step.masked_step import MaskedStepper

    names = ["time_out", "reference_finished", "body_fall", "anchor"]
    dones = torch.tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 0]], dtype=torch.bool
    )
    stepper = _log_stepper(names, dones)
    MaskedStepper._build_log_buffers(stepper)
    assert set(REQUIRED_ENV_LOG_ENTRIES) <= set(stepper.log)
    entry = stepper.log[TRACKING_FAILED_ENTRY]

    MaskedStepper._reset_termination_manager(stepper, torch.zeros(4, dtype=torch.bool))

    # Written in place, so the captured segment and the log dict stay linked;
    # one environment with two failure terms counts once.
    assert stepper.log[TRACKING_FAILED_ENTRY] is entry
    assert float(entry) == 0.25


def test_masked_step_keeps_a_recipe_tracking_failed_term() -> None:
    from imitationlab.envs.isaaclab.contracts.training_log import TRACKING_FAILED_ENTRY
    from imitationlab.envs.isaaclab.envs.graph_step.masked_step import MaskedStepper

    names = ["time_out", "tracking_failed"]
    dones = torch.tensor([[0, 1], [0, 0]], dtype=torch.bool)
    stepper = _log_stepper(names, dones)
    MaskedStepper._build_log_buffers(stepper)
    MaskedStepper._reset_termination_manager(stepper, torch.zeros(2, dtype=torch.bool))
    assert stepper._tracking_failed_log is None
    assert float(stepper.log[TRACKING_FAILED_ENTRY]) == 0.5
