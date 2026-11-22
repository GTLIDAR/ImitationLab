"""The captured step on the GPU (``env.step_mode``): graph = masked bit for bit, no sync.

- The masked step records physics, and the graph step replays it; every leaf
  of ``imitationlab.evaluation.validation.graph_step_equivalence`` (observations, the
  indexed terminal rows, reward, dones, cursors, histories, every sampler
  tensor and both random-generator states) must match over 210 steps at 256
  envs, with resets on most steps. One pair runs the ``wm`` group alone; the
  other every group with ``env.reset_fail_share=0.3``, where every masked
  reset's ending frame must also equal the cursor's.
- The replay overwrites the simulator state after each ``sim.step``, so the
  pair above cannot see whether the reset reached the physics. With live
  physics, graph, masked and (as reference) eager: on the reset rows the
  simulator holds the written state, forward kinematics ran and the solver
  state was cleared (``graph_step_statistics writes``).
- The graph step, and the masked step, run 100 steps under
  ``torch.cuda.set_sync_debug_mode("error")`` reading what WMBPTT reads
  (``imitationlab.evaluation.validation.graph_step_statistics sync``); the indexed
  ``final_obs`` read must raise there, which shows the check is live.
- Opt-in, about half an hour: with live physics, the masked step against the
  eager step of ``imitation-g1-v1``, and the graph step against the
  masked step, as distributions over four runs per arm
  (``IMITATION_G1_GRAPH_STATS=1``).

Opt-in like ``test_g1_step_equivalence.py``: set ``IMITATION_G1_EQUIV_BANK``
to ``g1_lafan1_corrected_40_d972c37c_v1``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from imitationlab.evaluation.validation.graph_step_statistics import compare_groups
from imitationlab.evaluation.validation.graph_step_equivalence import compare_records
from imitationlab.paths import REPO_ROOT

from test_g1_step_equivalence import _overrides, _skip_reason  # noqa: E402

TASK = "imitation-g1-v1"
STEPS = 210
NUM_ENVS = 256


def _run(tmp_path: Path, name: str, module: str, mode: str, *args: str) -> Path:
    out = tmp_path / f"{name}.json"
    env = dict(
        os.environ,
        IMITATIONLAB_ENV_COLLECT_FASTPATH="1",
        WANDB_MODE="disabled",
        OMNI_KIT_ACCEPT_EULA="YES",
        TERM=os.environ.get("TERM", "xterm"),
    )
    cmd = [
        sys.executable,
        "-m",
        f"imitationlab.evaluation.validation.{module}",
        mode,
        "--out",
        str(out),
        *args,
    ]
    log = tmp_path / f"{name}.log"
    with log.open("w") as handle:
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            timeout=2400,
        )
    assert result.returncode == 0, f"{name} arm failed; see {log}"
    return out


def _equivalence_arm(tmp_path, name, step_mode, physics, *options) -> dict:
    out = _run(
        tmp_path,
        name,
        "graph_step_equivalence",
        "run",
        "--task",
        TASK,
        "--steps",
        str(STEPS),
        "--num_envs",
        str(NUM_ENVS),
        "--physics",
        physics,
        "--",
        *_overrides(),
        f"env.step_mode={step_mode}",
        *options,
    )
    if step_mode == "graph":
        assert "Captured the G1 graph step" in (tmp_path / f"{name}.log").read_text()
    return json.loads(out.read_text())


@pytest.mark.skipif(_skip_reason() is not None, reason=str(_skip_reason()))
@pytest.mark.parametrize(
    "options",
    [
        ("env.wm_only_observations=true",),
        ("env.reset_fail_share=0.3",),
        # The reference frame derived from the library instead of the packed arrays
        # (``graph_reference_source=library``): the same bit-for-bit contract.
        ("env.wm_only_observations=true", "env.graph_reference_source=library"),
        # The per-env chunk of reference rows (``library_env_window``) beside the
        # library frame: the same bit-for-bit contract, resets included.
        (
            "env.reset_fail_share=0.3",
            "env.graph_reference_source=library",
            "env.library_env_window=16",
        ),
    ],
    ids=[
        "wm_only",
        "all_groups_fail_share",
        "wm_only_library_frame",
        "library_frame_env_window",
    ],
)
def test_graph_step_is_bit_identical_to_the_masked_step(tmp_path, options) -> None:
    trace = tmp_path / "physics.pt"
    masked = _equivalence_arm(tmp_path, "masked", "masked", f"record:{trace}", *options)
    graph = _equivalence_arm(tmp_path, "graph", "graph", f"replay:{trace}", *options)
    assert masked["physics_steps"] == graph["physics_steps"] == STEPS
    assert masked["terminated"] > 0 and masked["truncated"] > 0
    assert masked["resets"] > STEPS
    assert compare_records(masked, graph) == []
    if "env.reset_fail_share=0.3" in options:
        # The ending frame each masked reset charges is the cursor's. The graph
        # arm is checked only before its capture; it equals this arm above.
        assert masked["frame_checks"] >= STEPS
        assert masked["ended_rows"] > 0
        assert masked["frame_mismatches"] == graph["frame_mismatches"] == 0


@pytest.mark.skipif(_skip_reason() is not None, reason=str(_skip_reason()))
@pytest.mark.parametrize("step_mode", ["graph", "masked"])
def test_the_step_runs_without_a_host_synchronization(tmp_path, step_mode) -> None:
    out = _run(
        tmp_path,
        "sync",
        "graph_step_statistics",
        "sync",
        "--task",
        TASK,
        "--steps",
        "100",
        "--num_envs",
        str(NUM_ENVS),
        "--",
        *_overrides(),
        f"env.step_mode={step_mode}",
        "env.wm_only_observations=true",
    )
    record = json.loads(out.read_text())
    assert record["step_mode"] == step_mode
    assert record["resets"] > 0 and record["finite"]
    assert record["indexed_final_obs_refused"]
    log = (tmp_path / "sync.log").read_text()
    assert f"100 {step_mode} steps ran without a host synchronization." in log


WRITE_CHECKS = (
    "root_position",
    "root_orientation",
    "joint_position",
    "joint_velocity",
    "fk_root",
    "reset_warmstart",
)


@pytest.mark.skipif(_skip_reason() is not None, reason=str(_skip_reason()))
@pytest.mark.parametrize(
    ("task", "step_mode"),
    [(TASK, "graph"), (TASK, "masked"), (TASK, "eager")],
    ids=["graph", "masked", "eager"],
)
def test_the_reset_reaches_the_simulator_on_live_physics(
    tmp_path, task, step_mode
) -> None:
    """The bitwise test replays physics over the reset; this one does not.

    On the rows a step reset, the simulator holds the written state, forward
    kinematics ran, and the MuJoCo-Warp solver state was cleared
    (``graph_step_statistics writes``). The eager step is the reference.
    """
    mode = () if step_mode == "eager" else (f"env.step_mode={step_mode}",)
    out = _run(
        tmp_path,
        "writes",
        "graph_step_statistics",
        "writes",
        "--task",
        task,
        "--steps",
        "100",
        "--num_envs",
        str(NUM_ENVS),
        "--",
        *_overrides(),
        "env.wm_only_observations=true",
        *mode,
    )
    record = json.loads(out.read_text())
    assert record["step_mode"] == step_mode
    if step_mode == "graph":
        assert record["captured_at_step"] is not None
    assert record["solver_warmstart_checked"]
    assert record["reset_rows"] > 100
    assert record["max_error"] == {name: 0.0 for name in WRITE_CHECKS}, record
    assert record["pending_masks"] == 0
    # Liveness: the reset moves a root far more than a physics step does, and
    # the rows it kept carry a warm start.
    assert record["reset_root_move_mean_m"] > 10 * record["kept_root_move_mean_m"]
    assert record["kept_warmstart_nonzero_share"] > 0.5


@pytest.mark.skipif(
    _skip_reason() is not None or os.environ.get("IMITATION_G1_GRAPH_STATS") != "1",
    reason=str(_skip_reason() or "set IMITATION_G1_GRAPH_STATS=1"),
)
def test_the_masked_and_graph_steps_match_the_eager_step_in_distribution(
    tmp_path,
) -> None:
    """Four runs per arm, live physics; each run's mean after the transient is one sample.

    Masked is compared with eager, and graph with masked. A within-run block
    test does not hold here: two eager seeds alone differ beyond their block
    errors (``graph_step_statistics`` docstring).
    """

    def arm(name, task, seed, *options):
        out = _run(
            tmp_path,
            name,
            "graph_step_statistics",
            "run",
            "--task",
            task,
            "--steps",
            "1200",
            "--num_envs",
            "4096",
            "--seed",
            str(seed),
            "--",
            *_overrides(),
            "env.wm_only_observations=true",
            *options,
        )
        return json.loads(out.read_text())

    eager = [arm(f"eager{s}", TASK, s) for s in range(4)]
    masked = [arm(f"masked{s}", TASK, s, "env.step_mode=masked") for s in range(4)]
    graph = [arm(f"graph{s}", TASK, s, "env.step_mode=graph") for s in range(4)]
    for a, b in ((eager, masked), (masked, graph)):
        result = compare_groups(a, b, skip=300)
        # |t| 5 at about 6 degrees of freedom is p ~ 0.003 per scalar.
        assert max(abs(row["t"]) for row in result["scalars"]) < 5.0, result
        hists = {row["name"]: row for row in result["hists"]}
        # The clip and start histograms must be able to differ. On LAFAN1 the
        # clip-length histogram has one occupied bin and compares nothing.
        assert hists["reset_clip"]["bins_with_mass"] > 1, result
        assert hists["reset_start_position"]["bins_with_mass"] > 1, result
        for row in result["hists"]:
            assert row["tv"] < 2.0 * row["within_tv"] + 1e-3, result
