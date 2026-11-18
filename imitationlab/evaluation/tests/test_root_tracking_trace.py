"""Row bookkeeping of the per-step root tracking trace (no simulator)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from imitationlab.evaluation.root_tracking_trace import (
    RootTrackingTrace,
    episode_rows,
    field,
    load_root_trace,
    root_error,
    summaries,
    take_arrays,
)

IDENTITY = torch.tensor([0.0, 0.0, 0.0, 1.0])


def _pose(xyz: list[list[float]]) -> torch.Tensor:
    pos = torch.tensor(xyz, dtype=torch.float32)
    return torch.cat([pos, IDENTITY.expand(pos.shape[0], 4)], dim=-1)


def _trace(record_joints: bool = True) -> RootTrackingTrace:
    """Two environments with origins 10 m apart; env 1 finishes after step 2."""

    trace = RootTrackingTrace(
        env_origins=torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        trajectory_ranks=[7, 3],
        motion_names=["walk", "jog"],
        step_dt=0.02,
        provenance={"evaluator": "test"},
        joint_names=["a", "b"] if record_joints else None,
    )
    # step t: robot at x = 0.1 t, reference at x = 0.1 t + drift t * 0.01 (env 0)
    for t in range(4):
        robot = _pose([[0.1 * t, 0.0, 0.8], [10.0 + 0.2 * t, 0.0, 0.8]])
        reference = _pose([[0.1 * t + 0.01 * t, 0.0, 0.8], [10.0 + 0.2 * t, 0.0, 0.8]])
        active = torch.tensor([True, t <= 2])
        joints = torch.full((2, 2), float(t)) if record_joints else None
        trace.append(
            robot_pose_w=robot,
            reference_pose_w=reference,
            mpjpe_local_m=torch.tensor([0.02, 0.03]),
            mpjpe_global_m=torch.tensor([0.05, 0.03]),
            active=active,
            local_step=torch.tensor([t, t if t <= 2 else 0]),
            joint_pos=joints,
            reference_joint_pos=None if joints is None else joints + 1.0,
        )
    return trace


def test_origins_are_removed_and_errors_follow_the_rows(tmp_path):
    path = _trace().write(tmp_path / "trace.npz")
    trace = load_root_trace(path)
    assert trace["rows"].shape == (4, 2, 16)
    np.testing.assert_allclose(
        field(trace, "robot_x")[:, 1], [0.0, 0.2, 0.4, 0.6], atol=1e-6
    )
    np.testing.assert_allclose(
        root_error(trace)[:, 0], [0.0, 0.01, 0.02, 0.03], atol=1e-6
    )
    assert trace["provenance"]["evaluator"] == "test"
    assert trace["step_dt"] == pytest.approx(0.02)


def test_episode_rows_stop_at_the_first_inactive_row():
    trace = {**_trace().arrays(), "provenance": {}, "step_dt": 0.02}
    np.testing.assert_array_equal(episode_rows(trace, 0), [0, 1, 2, 3])
    np.testing.assert_array_equal(episode_rows(trace, 1), [0, 1, 2])


def test_summaries_skip_the_reset_row():
    trace = {**_trace().arrays(), "provenance": {}, "step_dt": 0.02}
    rows = summaries(trace)
    assert rows[0]["scored_rows"] == 3 and rows[1]["scored_rows"] == 2
    assert rows[0]["root_xyz_error_mm"] == pytest.approx(20.0, abs=1e-3)
    assert rows[0]["root_xyz_error_final_mm"] == pytest.approx(30.0, abs=1e-3)
    assert rows[1]["mpjpe_l_mm"] == pytest.approx(30.0, abs=1e-3)
    assert rows[0]["trajectory_rank"] == 7 and rows[1]["motion_name"] == "jog"


def test_take_arrays_match_the_renderer_layout():
    trace = {**_trace().arrays(), "provenance": {"evaluator": "test"}, "step_dt": 0.02}
    take = take_arrays(trace, 1)
    assert take["root_pose"].shape == (3, 7) and take["joint_pos"].shape == (3, 2)
    np.testing.assert_allclose(take["root_pose"][:, 0], [0.0, 0.2, 0.4], atol=1e-6)
    reference = take_arrays(trace, 1, reference=True)
    np.testing.assert_allclose(reference["joint_pos"][:, 0], [1.0, 2.0, 3.0])
    provenance = json.loads(take["provenance"])
    assert provenance["trajectory_rank"] == 3 and provenance["env_origin"] == [
        0.0,
        0.0,
        0.0,
    ]
    np.testing.assert_allclose(take["time_s"], [0.0, 0.02, 0.04])


def test_take_arrays_need_joints():
    trace = {**_trace(record_joints=False).arrays(), "provenance": {}, "step_dt": 0.02}
    with pytest.raises(ValueError, match="without joints"):
        take_arrays(trace, 0)


def test_rank_count_must_match_environments():
    with pytest.raises(ValueError, match="one rank"):
        RootTrackingTrace(
            env_origins=torch.zeros(2, 3),
            trajectory_ranks=[1],
            motion_names=["a", "b"],
            step_dt=0.02,
            provenance={},
        )
