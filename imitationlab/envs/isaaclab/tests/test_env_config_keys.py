"""`train.py` refuses `env.*` overrides that name no declared key.

The guard is safe only if every real task passes it, so each registered task is
merged here as `train.py` merges it, with its presets resolved. Keys read with
`getattr` are the trap: they work as undeclared overrides, so a guard that does
not know them refuses working campaigns.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import pytest
from isaaclab_tasks.utils import resolve_task_config

import imitationlab.envs.isaaclab
import imitationlab.envs.isaaclab.tasks  # noqa: F401  (registers the gym tasks)
from imitationlab.paths import REPO_ROOT
from imitationlab.runners.provenance.config_keys import unknown_config_keys
from imitationlab.envs.isaaclab.tasks.manager_based.imitation.config.g1.imitation_g1_env import (
    ImitationG1EnvCfg,
)

TASK_IDS = sorted(
    spec_id
    for spec_id in gym.registry
    if str((gym.spec(spec_id).kwargs or {}).get("env_cfg_entry_point", "")).startswith(
        "imitationlab.envs.isaaclab"
    )
)

TASK = "imitation-g1-v1"
TASK_OPTIONS = {
    "wm_only_observations": "true",
    "reset_fail_share": "0.3",
    "reward_kernel_scale": "0.7",
    "persist_reset_stats": "true",
}
# Every key the data plane reads with `getattr`, with a value a run used.
GETATTR_KEYS = {
    "expert_macro_frame_stride": "5",
    "expert_macro_anchor_mode": "robot_heading",
    "expert_macro_rollout_anchor_mode": "previous_expert_heading",
    "expert_macro_anchor_walk_std": "0.002",
    "expert_macro_anchor_walk_max": "0.1",
    "expert_macro_anchor_jitter": "0.005",
    "expert_macro_anchor_jump_prob": "0.05",
    "expert_macro_anchor_jump": "0.03",
    "achieved_ring_capacity": "101",
}


def _unknown(task_id: str, *overrides: str) -> list[str]:
    env_cfg, _ = resolve_task_config(task_id, None, overrides=list(overrides))
    return unknown_config_keys(env_cfg, prefix="env")


def test_the_scan_found_the_task() -> None:
    assert TASK_IDS == [TASK]


@pytest.mark.parametrize(
    "physics", [(), ("physics=newton_mjwarp",)], ids=["default", "newton"]
)
@pytest.mark.parametrize("task_id", TASK_IDS)
def test_every_registered_task_passes_the_guard(task_id: str, physics: tuple) -> None:
    """Before and after `resolve`, which `train.py` calls later, on both backends."""
    env_cfg, _ = resolve_task_config(task_id, None, overrides=list(physics))
    assert unknown_config_keys(env_cfg, prefix="env") == []
    resolve = getattr(env_cfg, "resolve", None)
    if callable(resolve):
        resolve()
        assert unknown_config_keys(env_cfg, prefix="env") == []


@pytest.mark.parametrize("name", TASK_OPTIONS)
def test_a_mistyped_option_is_reported(name: str) -> None:
    typo = name[:-1]
    assert _unknown(TASK, f"env.{typo}={TASK_OPTIONS[name]}") == [f"env.{typo}"]


@pytest.mark.parametrize("name", TASK_OPTIONS)
def test_the_options_pass(name: str) -> None:
    assert _unknown(TASK, f"env.{name}={TASK_OPTIONS[name]}") == []


def test_the_prefetch_setting_has_one_name() -> None:
    """A top-level key would be overwritten or ignored; the data plane reads `data`."""
    assert _unknown(TASK, "env.reference_prefetch_mode=off") == [
        "env.reference_prefetch_mode"
    ]
    assert _unknown(TASK, "env.data.reference_prefetch_mode=off") == []


@pytest.mark.parametrize("name", GETATTR_KEYS)
def test_the_keys_the_data_plane_reads_stay_accepted(name: str) -> None:
    env_cfg, _ = resolve_task_config(
        TASK, None, overrides=[f"env.{name}={GETATTR_KEYS[name]}"]
    )
    assert unknown_config_keys(env_cfg, prefix="env") == []
    assert str(getattr(env_cfg, name)) == GETATTR_KEYS[name]


def _data_plane_fallbacks() -> dict[str, object]:
    source = (
        Path(imitationlab.envs.isaaclab.__file__).parent / "envs/expert_data_plane.py"
    )
    fallbacks: dict[str, object] = {}
    for node in ast.walk(ast.parse(source.read_text())):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "getattr"
            and len(node.args) == 3
            and getattr(node.args[1], "value", None) in GETATTR_KEYS
        ):
            fallbacks[node.args[1].value] = ast.literal_eval(node.args[2])
    return fallbacks


def test_the_declared_defaults_match_the_data_plane_fallbacks() -> None:
    """The class defaults equal the read sites' fallbacks, so declaring changed nothing."""
    fallbacks = _data_plane_fallbacks()
    assert set(fallbacks) == set(GETATTR_KEYS)
    cfg = ImitationG1EnvCfg()
    assert {name: getattr(cfg, name) for name in fallbacks} == fallbacks


def test_train_py_refuses_a_mistyped_env_override_before_the_simulator() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "imitationlab.runners.train",
            "--task",
            TASK,
            "--algo",
            "WMBPTT",
            "--headless",
            "--assert-kitless",
            "physics=newton_mjwarp",
            "env.reset_fail_shar=0.3",
        ],
        cwd=REPO_ROOT,
        env=dict(os.environ, WANDB_MODE="disabled", OMNI_KIT_ACCEPT_EULA="YES"),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode != 0
    assert "env.reset_fail_shar" in result.stderr + result.stdout
    assert "fields that do not exist" in result.stderr + result.stdout
