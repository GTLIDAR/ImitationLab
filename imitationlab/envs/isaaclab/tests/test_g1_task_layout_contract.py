"""Default observation-layout contract of ``imitation-g1-v1``.

The ordered term list of each observation group IS the checkpoint input
contract: reordering, adding, or removing a term silently changes the actor's
input layout and invalidates every checkpoint trained against it. Both command
interfaces are recorded: the explicit reference, and a frozen command encoder
(``env.skill_encoder_path``), which switches the policy and critic features.

Regenerate the recorded default layout only when a layout change is
intentional:

    REGENERATE_G1_TASK_LAYOUT_DEFAULT=1 pixi run -e isaaclab pytest -q \
        imitationlab/envs/isaaclab/tests/test_g1_task_layout_contract.py

and say so in the commit message.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import os
from pathlib import Path

import pytest

import imitationlab.envs.isaaclab.tasks  # noqa: F401  (registers the gym tasks)
import gymnasium as gym

DEFAULT_LAYOUT_PATH = Path(__file__).with_name("g1_task_layout_default.json")

TASK = "imitation-g1-v1"
#: Recorded layouts: the task as registered, and with a frozen command encoder.
VARIANTS = {TASK: {}, f"{TASK} (encoder)": {"skill_encoder_path": "encoder.pt"}}


def _load_env_cfg(fields: dict, monkeypatch: pytest.MonkeyPatch):
    """The task's config as the environment sees it: constructed, then resolved.

    The encoder file is read only to pin its weights hash, so that step is
    skipped; the interface switch follows the path alone.
    """
    entry = gym.spec(TASK).kwargs["env_cfg_entry_point"]
    module_name, class_name = entry.split(":")
    cls = getattr(importlib.import_module(module_name), class_name)
    monkeypatch.setattr(cls, "_bind_skill_encoder", lambda self: None)
    cfg = cls(**fields)
    cfg.resolve()
    return cfg


def _group_terms(group) -> list[str]:
    terms = []
    for field in dataclasses.fields(group):
        value = getattr(group, field.name)
        if value is not None and hasattr(value, "func"):
            terms.append(field.name)
    return terms


def _layout(cfg) -> dict:
    groups = {}
    for field in dataclasses.fields(cfg.observations):
        group = getattr(cfg.observations, field.name)
        if group is None:
            continue
        groups[field.name] = _group_terms(group)
    return {
        "env_cfg_class": type(cfg).__name__,
        "observations_class": type(cfg.observations).__name__,
        "rewards_class": type(cfg.rewards).__name__,
        "terminations_class": type(cfg.terminations).__name__,
        "events_class": type(cfg.events).__name__,
        "actions_class": type(cfg.actions).__name__,
        "curriculum": type(cfg.curriculum).__name__ if cfg.curriculum else None,
        "anchor_body": _anchor_body(cfg),
        "command": _command_surface(cfg),
        "groups": groups,
    }


def _anchor_body(cfg) -> str:
    """The body anchor-relative terms are expressed in (the reference channel's)."""
    return cfg.command_interface.reference.anchor_body_name


def _command_surface(cfg) -> dict:
    """The task's declared command interface."""
    interface = cfg.command_interface
    selection = interface.reference.selection
    encoder = interface.encoder
    return {
        "generation": "interface",
        "actor_kind": interface.actor_kind(),
        "actor_source": interface.actor.source,
        "actor_components": list(interface.actor_components()),
        "actor_latent_dim": getattr(interface.actor, "dim", None),
        "actor_hold_steps": getattr(interface.actor, "hold_steps", None),
        "actor_horizon": getattr(interface.actor, "horizon", None),
        "critic_channels": list(interface.critic_channels),
        "critic_components": list(interface.critic_components()),
        "encoder_components": (
            list(encoder.components) if encoder is not None else None
        ),
        "encoder_window": (
            [int(encoder.past_steps), int(encoder.future_steps)]
            if encoder is not None
            else None
        ),
        "selection_schedule": selection.schedule,
        "selection_start_mode": selection.start_mode,
        "selection_start_frame": int(selection.start_frame),
        "selection_random_step_min": int(selection.random_step_min),
        "selection_random_step_max": int(selection.random_step_max),
        "selection_full_trajectory": bool(selection.full_trajectory),
    }


def test_registered_task_layouts_match_default(monkeypatch) -> None:
    assert [spec for spec in gym.registry if spec.startswith("imitation-")] == [TASK]
    current = {
        name: _layout(_load_env_cfg(fields, monkeypatch))
        for name, fields in VARIANTS.items()
    }
    if os.environ.get("REGENERATE_G1_TASK_LAYOUT_DEFAULT") == "1":
        DEFAULT_LAYOUT_PATH.write_text(
            json.dumps(current, indent=2, sort_keys=True) + "\n"
        )
        pytest.skip(f"regenerated {DEFAULT_LAYOUT_PATH.name}")
    assert DEFAULT_LAYOUT_PATH.is_file(), (
        f"{DEFAULT_LAYOUT_PATH} missing; regenerate with "
        "REGENERATE_G1_TASK_LAYOUT_DEFAULT=1"
    )
    recorded = json.loads(DEFAULT_LAYOUT_PATH.read_text())
    assert sorted(current) == sorted(recorded), (
        "Variant set changed. If intentional, regenerate the recorded default "
        "layout and record the reason in the commit message."
    )
    for name in VARIANTS:
        assert current[name] == recorded[name], (
            f"{name}: layout or protocol drift vs the recorded default. If "
            "intentional (checkpoint-breaking!), regenerate it."
        )


def test_reset_start_mode_config_surface(monkeypatch) -> None:
    """The reference selection validates and normalizes its start-frame policy."""
    cfg = _load_env_cfg({}, monkeypatch)
    selection = cfg.command_interface.reference.selection
    assert selection.start_mode in ("auto", "fixed", "random", "adaptive")
    selection.start_mode = " Adaptive "
    selection.resolve()
    assert selection.start_mode == "adaptive"
    selection.start_mode = "bogus"
    with pytest.raises(ValueError, match="start_mode"):
        selection.resolve()
