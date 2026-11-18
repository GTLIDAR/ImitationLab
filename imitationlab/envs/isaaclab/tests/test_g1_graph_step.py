"""Captured-step (``env.step_mode`` masked or graph) contracts that need no simulator."""

from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch

import imitationlab.envs.isaaclab.tasks  # noqa: F401  (registers the gym tasks)
from imitationlab.envs.isaaclab.envs.graph_step.g1 import (
    G1MaskedStepper,
    LazyTerminalObs,
)
from imitationlab.envs.isaaclab.envs.g1_env import (
    G1GraphRLEnv,
    G1LatentRLEnv,
    G1RLEnv,
    make_imitation_g1_v1,
)
from imitationlab.envs.isaaclab.tasks.manager_based.imitation.config.g1.imitation_g1_env import (
    ImitationG1EnvCfg,
)

TASK = "imitation-g1-v1"


def _graph_cfg(**fields) -> ImitationG1EnvCfg:
    return ImitationG1EnvCfg(step_mode="graph", **fields)


def test_the_graph_env_refuses_the_eager_step_and_the_eager_env_keeps_its_guard():
    with pytest.raises(ValueError, match="step_mode"):
        G1GraphRLEnv(SimpleNamespace(step_mode="eager"))
    for mode in ("masked", "graph"):
        with pytest.raises(ValueError, match="step_mode='eager'"):
            G1RLEnv(SimpleNamespace(step_mode=mode))


def test_the_graph_step_changes_only_the_step_mode():
    graph, eager = _graph_cfg(), ImitationG1EnvCfg()
    graph.resolve()
    eager.resolve()
    a, b = graph.to_dict(), eager.to_dict()
    assert (a.pop("step_mode"), b.pop("step_mode")) == ("graph", "eager")
    assert a == b


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("graph_reference_source", "bank", "graph_reference_source"),
        ("pinned_reset", True, "pinned_reset"),
    ],
)
def test_the_graph_cfg_refuses_what_its_step_cannot_run(field, value, match):
    cfg = _graph_cfg()
    setattr(cfg, field, value)
    with pytest.raises(ValueError, match=match):
        cfg.resolve()


def test_the_graph_cfg_needs_next_prefetch():
    cfg = _graph_cfg()
    cfg.data.reference_prefetch_mode = "off"
    with pytest.raises(ValueError, match="prefetch"):
        cfg.resolve()
    cfg = _graph_cfg()
    cfg.step_mode = "masked"
    cfg.resolve()


@pytest.mark.parametrize(
    ("fields", "env_class"),
    [
        ({}, G1RLEnv),
        ({"step_mode": "masked"}, G1GraphRLEnv),
        ({"step_mode": "graph"}, G1GraphRLEnv),
        ({"skill_encoder_path": "encoder.pt"}, G1LatentRLEnv),
    ],
)
def test_the_task_selects_the_env_class_from_its_settings(
    monkeypatch, fields, env_class
):
    assert gym.spec(TASK).entry_point.endswith("g1_env:make_imitation_g1_v1")
    built = []
    for cls in (G1RLEnv, G1GraphRLEnv, G1LatentRLEnv):
        monkeypatch.setattr(
            cls, "__init__", lambda self, cfg, _c=cls, **kw: built.append(_c)
        )
    make_imitation_g1_v1(
        SimpleNamespace(**{"step_mode": "eager", "skill_encoder_path": "", **fields})
    )
    assert built == [env_class]


def test_the_frozen_encoders_refuse_the_captured_step():
    with pytest.raises(ValueError, match="step_mode=eager"):
        make_imitation_g1_v1(
            SimpleNamespace(step_mode="graph", skill_encoder_path="encoder.pt")
        )


def _publisher(mask, terminal):
    stepper = SimpleNamespace(
        env=SimpleNamespace(extras={"final_info": 1}),
        _terminal_obs=terminal,
        reset_mask=mask,
        steps_taken=1,
    )
    G1MaskedStepper._publish_terminal_obs(stepper)
    return stepper


def test_final_obs_is_the_eager_index_form_built_on_first_read():
    mask = torch.tensor([False, True, False, True])
    terminal = {"wm": {"state": torch.arange(8.0).view(4, 2)}}
    stepper = _publisher(mask, terminal)
    extras = stepper.env.extras
    final = extras["final_obs"]
    assert isinstance(final, LazyTerminalObs) and isinstance(final, dict)
    assert "unread" in repr(final)
    assert "final_info" not in extras
    assert final["_env_ids"].tolist() == [1, 3]
    assert final["obs"]["wm"]["state"].tolist() == [[2.0, 3.0], [6.0, 7.0]]
    assert set(final) == {"_env_ids", "obs"} and len(final) == 2
    masked = extras["final_obs_masked"]
    assert masked["_env_mask"] is mask and masked["obs"] is terminal


def test_a_step_without_resets_publishes_empty_rows():
    stepper = _publisher(
        torch.zeros(3, dtype=torch.bool), {"wm": {"x": torch.ones(3, 2)}}
    )
    final = stepper.env.extras["final_obs"]
    assert final["_env_ids"].numel() == 0
    assert final["obs"]["wm"]["x"].shape == (0, 2)


def test_final_obs_read_after_the_next_step_is_refused():
    stepper = _publisher(torch.tensor([True]), {"wm": {"x": torch.ones(1, 1)}})
    final = stepper.env.extras["final_obs"]
    stepper.steps_taken += 1
    with pytest.raises(RuntimeError, match="valid only until the next step"):
        final["_env_ids"]


def test_history_rings_shift_in_place_to_the_eager_values():
    n = 3
    torch.manual_seed(0)
    rings = {
        "_g1_states": torch.randn(n, 10, 71),
        "_g1_actions": torch.randn(n, 10, 29),
        "_g1_proprio": torch.randn(n, 10, 99),
    }
    state, action, proprio = torch.randn(n, 71), torch.randn(n, 29), torch.randn(n, 99)

    def env_with(tensors):
        return SimpleNamespace(
            **{k: v.clone() for k, v in tensors.items()},
            g1_state=lambda: state,
            action_manager=SimpleNamespace(action=action),
            g1_cursor=lambda: (None, None),
            _g1_observation_frame=lambda *args: proprio,
        )

    eager, masked = env_with(rings), env_with(rings)
    G1RLEnv._g1_append_history(eager)
    before = {k: getattr(masked, k) for k in rings}
    G1MaskedStepper._append_history(SimpleNamespace(env=masked))
    for key in rings:
        assert getattr(masked, key) is before[key]
        assert torch.equal(getattr(masked, key), getattr(eager, key))
