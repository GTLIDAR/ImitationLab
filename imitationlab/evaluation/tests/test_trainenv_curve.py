from __future__ import annotations

import math

import pytest

from imitationlab.evaluation.trainenv_curve import eval_json_for, row_from_eval


def _keys(**means: float) -> dict:
    return {"reset_logs": {"keys": {k: {"settled_mean": v} for k, v in means.items()}}}


def test_row_reads_settled_means_and_sums_success() -> None:
    summary = _keys(
        **{
            "Metrics/reference/mpjpe_l_mm": 30.0,
            "Metrics/reference/mpjpe_g_mm": 120.0,
            "Episode_Termination/reference_finished": 0.1,
            "Episode_Termination/time_out": 0.6,
            "Episode_Termination/foot_pos_xyz": 0.2,
            "Episode_Termination/anchor_ori": 0.1,
            "Metrics/reference/reset_return_norm_undiscounted": 0.7,
        }
    )
    row = row_from_eval(summary, arm="wm", frames=100, seconds=5400.0)
    assert row["hours"] == 1.5
    assert row["mpjpe_l_mm"] == 30.0 and row["mpjpe_g_mm"] == 120.0
    assert row["success"] == pytest.approx(0.7)
    assert row["fail_foot_pos_xyz"] == 0.2 and "fail_time_out" not in row


def test_missing_metric_is_nan_and_missing_logs_raise() -> None:
    row = row_from_eval(
        _keys(**{"Episode_Termination/time_out": 1.0}), arm="ppo", frames=1, seconds=1.0
    )
    assert math.isnan(row["mpjpe_l_mm"])
    with pytest.raises(ValueError):
        row_from_eval({"reset_logs": None}, arm="ppo", frames=1, seconds=1.0)


def test_sonic_release_schema_is_read() -> None:
    summary = {
        "episode_reward": {
            "Metrics/reference/mpjpe_l_mm": {"settled_mean": 28.8},
            "Episode_Termination/time_out": {"settled_mean": 0.5},
            "Episode_Termination/reference_finished": {"settled_mean": 0.3},
        }
    }
    row = row_from_eval(summary, arm="sonic", frames=0, seconds=0.0)
    assert row["mpjpe_l_mm"] == 28.8 and row["success"] == pytest.approx(0.8)


def test_eval_json_for_reads_both_naming_conventions(tmp_path) -> None:
    ckpt = tmp_path / "model_step_250.pt"
    assert eval_json_for(ckpt, tmp_path) is None
    (tmp_path / "wm_z_seed0_milestone_f250.json").write_text("{}")
    (tmp_path / "wm_z_seed0_milestone_f2500.json").write_text("{}")
    assert eval_json_for(ckpt, tmp_path).name == "wm_z_seed0_milestone_f250.json"
    (tmp_path / "model_step_250.json").write_text("{}")
    assert eval_json_for(ckpt, tmp_path).name == "model_step_250.json"
