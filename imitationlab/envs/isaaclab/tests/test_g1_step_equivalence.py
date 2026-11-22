"""G1 control steps against one recorded physics trace, bit for bit.

The first arm records physics with default settings. Every other arm replays
that trace, so the only differences left are the options under test:

- ``replay``: the same settings, the harness's self-check;
- ``fast``: ``IMITATIONLAB_ENV_COLLECT_FASTPATH=1``. Every leaf must match, so
  this is the counterpart of ``test_collect_fastpath_equivalence.py``;
- ``wm_only``: ``env.wm_only_observations=true``. The `wm` group, its terminal
  rows, rewards, dones, sampler state and both random streams must match, and the
  policy and critic groups must be absent;
- ``fail_next`` and ``fail_off``: ``env.reset_fail_share=0.3`` under each
  ``reference_prefetch_mode`` the task accepts. They must agree with each other and
  differ from the default, so this arm is also the harness's negative control.

Tolerance: none. See ``imitationlab.evaluation.validation.graph_step_equivalence``.

Opt-in, because it needs the GPU and a local bank: set
``IMITATION_G1_EQUIV_BANK`` to ``g1_lafan1_corrected_40_d972c37c_v1``. To compare
the default path with an older commit, set ``IMITATION_G1_EQUIV_BASE_TREE`` to an
export of it (``git archive <commit>``); the recording
arm then imports that tree. Every arm must prove where it imported from, because
a ``PYTHONPATH`` entry that misses falls through to the editable install and the
test would then compare this tree with itself.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import imitationlab.evaluation.validation.graph_step_equivalence as harness_module
from imitationlab.evaluation.validation.graph_step_equivalence import (
    compare_records,
    imported_packages,
)
from imitationlab.paths import REPO_ROOT

BANK = os.environ.get("IMITATION_G1_EQUIV_BANK", "")
BASE_TREE = os.environ.get("IMITATION_G1_EQUIV_BASE_TREE", "")
BODIES = (
    "[pelvis,left_hip_roll_link,left_knee_link,left_ankle_roll_link,"
    "right_hip_roll_link,right_knee_link,right_ankle_roll_link,torso_link,"
    "left_shoulder_roll_link,left_elbow_link,left_wrist_yaw_link,"
    "right_shoulder_roll_link,right_elbow_link,right_wrist_yaw_link]"
)
STEPS = 200
NUM_ENVS = 256


def _skip_reason() -> str | None:
    if not BANK:
        return "set IMITATION_G1_EQUIV_BANK to the LAFAN1 reference bank"
    if not Path(BANK).is_dir():
        return f"bank {BANK} does not exist"
    if not torch.cuda.is_available():
        return "CUDA is required"
    return None


def _overrides(prefetch: str = "next") -> list[str]:
    return [
        "physics=newton_mjwarp",
        "env.data.manifest=null",
        f"env.data.reference_arrays_dir={BANK}",
        "env.data.persist_id="
        + json.loads((Path(BANK) / "reference_arrays_manifest.json").read_text())[
            "key"
        ]["source"]["persist_id"],
        "env.data.reference_arrays_resident=true",
        "env.data.reference_arrays_warm_workers=8",
        "env.data.runtime_cache_device=cpu",
        f"env.data.reference_prefetch_mode={prefetch}",
        f"env.data.runtime_cache_body_names={BODIES}",
    ]


def _assert_imported_from(record: dict, tree: Path) -> None:
    root = tree.resolve()
    outside = [
        path
        for path in record["imported"].values()
        if not Path(path).is_relative_to(root)
    ]
    assert not outside, f"this arm did not import {root}: {outside}"


def _run_arm(
    tmp_path: Path,
    name: str,
    physics: str,
    *options: str,
    fast: str = "0",
    tree: str = "",
    prefetch: str = "next",
) -> dict:
    assert not tree or Path(tree).is_dir(), f"base tree {tree} does not exist"
    out = tmp_path / f"{name}.json"
    env = dict(
        os.environ,
        IMITATIONLAB_ENV_COLLECT_FASTPATH=fast,
        WANDB_MODE="disabled",
        OMNI_KIT_ACCEPT_EULA="YES",
        TERM=os.environ.get("TERM", "xterm"),
    )
    harness = ["-m", "imitationlab.evaluation.validation.graph_step_equivalence"]
    if tree:
        # The older tree has no harness module, so run this one as a script.
        env["PYTHONPATH"] = f"{tree}:{tree}/envs/isaaclab:{tree}/experiments"
        script = tmp_path / "harness.py"
        script.write_bytes(Path(harness_module.__file__).read_bytes())
        harness = [str(script)]
    cmd = [
        sys.executable,
        *harness,
        "run",
        "--out",
        str(out),
        "--steps",
        str(STEPS),
        "--num_envs",
        str(NUM_ENVS),
        "--physics",
        physics,
        "--",
        *_overrides(prefetch),
        *options,
    ]
    log = tmp_path / f"{name}.log"
    with log.open("w") as handle:
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            timeout=1800,
        )
    assert result.returncode == 0, f"{name} arm failed; see {log}"
    record = json.loads(out.read_text())
    _assert_imported_from(record, Path(tree) if tree else REPO_ROOT)
    return record


@pytest.fixture(scope="module")
def recorded(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, dict]:
    tmp_path = tmp_path_factory.mktemp("g1_equivalence")
    trace = tmp_path / "physics.pt"
    return (
        tmp_path,
        trace,
        _run_arm(tmp_path, "record", f"record:{trace}", tree=BASE_TREE),
    )


def _exercised(record: dict) -> None:
    """A comparison proves nothing about terminal rows that never occurred."""
    assert record["prefetch_mode"] == "next"
    assert record["physics_steps"] == STEPS
    assert record["terminated"] > 0 and record["truncated"] > 0


@pytest.mark.skipif(_skip_reason() is not None, reason=str(_skip_reason()))
def test_replay_reproduces_the_recording(recorded) -> None:
    tmp_path, trace, base = recorded
    _exercised(base)
    replay = _run_arm(tmp_path, "replay", f"replay:{trace}")
    assert compare_records(base, replay) == []


@pytest.mark.skipif(_skip_reason() is not None, reason=str(_skip_reason()))
def test_fast_paths_are_bit_identical_on_the_g1_step(recorded) -> None:
    tmp_path, trace, base = recorded
    fast = _run_arm(tmp_path, "fast", f"replay:{trace}", fast="1")
    _exercised(fast)
    assert compare_records(base, fast) == []


@pytest.mark.skipif(_skip_reason() is not None, reason=str(_skip_reason()))
def test_the_per_env_reference_chunk_is_bit_identical_on_the_eager_step(
    recorded,
) -> None:
    """``env.library_env_window``: the step's reads come from per-env chunks of
    the bank (refilled every K - 13 steps and at resets); every value is the bank's.
    A 16-frame chunk refills every 3 steps, so refills and resets both occur."""
    tmp_path, trace, base = recorded
    chunked = _run_arm(
        tmp_path,
        "env_window",
        f"replay:{trace}",
        "env.library_env_window=16",
        fast="1",
    )
    _exercised(chunked)
    assert compare_records(base, chunked) == []


@pytest.mark.skipif(_skip_reason() is not None, reason=str(_skip_reason()))
@pytest.mark.parametrize("fast", ["0", "1"])
def test_wm_only_observations_keep_the_wm_transition_bit_identical(
    recorded, fast: str
) -> None:
    tmp_path, trace, base = recorded
    arm = _run_arm(
        tmp_path,
        f"wm_only_{fast}",
        f"replay:{trace}",
        "env.wm_only_observations=true",
        fast=fast,
    )
    _exercised(arm)
    problems = compare_records(
        base,
        arm,
        require=("obs/wm", "final/wm", "final/_env_ids", "out/", "sampler/", "rng/"),
    )
    assert problems == []
    for row in arm["steps"]:
        assert not any(k.startswith(("obs/policy", "obs/critic")) for k in row)
        assert not any(k.startswith(("final/policy", "final/critic")) for k in row)


@pytest.mark.skipif(_skip_reason() is not None, reason=str(_skip_reason()))
def test_reset_fail_share_is_exact_under_both_prefetch_modes(recorded) -> None:
    tmp_path, trace, base = recorded
    arms = {
        mode: _run_arm(
            tmp_path,
            f"fail_{mode}",
            f"replay:{trace}",
            "env.reset_fail_share=0.3",
            prefetch=mode,
        )
        for mode in ("next", "off")
    }
    assert arms["next"]["prefetch_mode"] == "next"
    assert arms["off"]["prefetch_mode"] == "off"
    assert compare_records(arms["next"], arms["off"]) == []
    for arm in arms.values():
        assert arm["bin_fail_mass"] > 0
    # Negative control: the option must move the reset stream.
    assert compare_records(base, arms["next"], require=("sampler/bin_G",))


@pytest.mark.skipif(_skip_reason() is not None, reason=str(_skip_reason()))
def test_reset_fail_share_charges_the_frame_the_cursor_reports(tmp_path: Path) -> None:
    """On live physics, every ended episode's derived frame equals the cursor's."""
    arm = _run_arm(tmp_path, "fail_live", "", "env.reset_fail_share=0.3")
    assert arm["ended_rows"] > 0
    assert arm["frame_mismatches"] == 0
    assert arm["terminated"] > 0 and arm["truncated"] > 0


def test_a_base_tree_that_does_not_exist_is_refused(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="does not exist"):
        _run_arm(tmp_path, "record", "record:x", tree=str(tmp_path / "gone"))


def test_a_record_imported_from_another_tree_is_refused(tmp_path: Path) -> None:
    """The silent case: the tree exists, but the imports fell through to this one."""
    record = {"imported": imported_packages()}
    with pytest.raises(AssertionError, match="did not import"):
        _assert_imported_from(record, tmp_path)


def test_the_record_names_the_tree_it_imported() -> None:
    imported = imported_packages()
    assert set(imported) == {"imitationlab", "imitationlab.envs.isaaclab"}
    _assert_imported_from({"imported": imported}, REPO_ROOT)


def test_compare_flags_a_changed_leaf_and_names_the_step() -> None:
    a = {"reset": {"a": "1"}, "steps": [{"a": "1"}]}
    b = {"reset": {"a": "1"}, "steps": [{"a": "2"}]}
    assert compare_records(a, b) == ["step 1: 1/1 differ: ['a']"]


def test_compare_needs_identical_leaves_unless_a_subset_is_named() -> None:
    full, part = (
        {"out": "r", "obs/wm/x": "1", "obs/policy/y": "2"},
        {"out": "r", "obs/wm/x": "1"},
    )
    a = {"reset": full, "steps": [full]}
    b = {"reset": part, "steps": [part]}
    assert compare_records(a, b)
    assert compare_records(a, b, require=("obs/wm",)) == []


def test_compare_fails_when_a_required_prefix_was_never_compared() -> None:
    row = {"out": "r"}
    a = {"reset": row, "steps": [row]}
    assert compare_records(a, a, require=("obs/wm",)) == [
        "no compared leaf starts with 'obs/wm'"
    ]


def test_compare_detects_a_mismatch_inside_the_required_subset() -> None:
    a = {"reset": {"obs/wm/x": "1"}, "steps": [{"obs/wm/x": "1"}]}
    b = {"reset": {"obs/wm/x": "1"}, "steps": [{"obs/wm/x": "2"}]}
    assert compare_records(a, b, require=("obs/wm",))
