"""ResetLogRecorder: reset-time snapshots summarized like a training curve."""

from __future__ import annotations

import pytest
import torch

from imitationlab.evaluation.reset_logs import ResetLogRecorder


def test_settled_mean_uses_the_second_half_only() -> None:
    recorder = ResetLogRecorder()
    for step in range(10):
        recorder.record(
            step,
            {
                "Episode_Reward/a": torch.tensor(float(step)),
                "Metrics/reference/b": 1.0,
                "Other/c": 5.0,
            },
            reset_count=2,
        )
    summary = recorder.summary(10, bins=2)
    keys = summary["keys"]
    assert set(keys) == {"Episode_Reward/a", "Metrics/reference/b"}
    assert keys["Episode_Reward/a"]["settled_mean"] == pytest.approx(7.0)
    assert keys["Episode_Reward/a"]["bin_means"] == [2.0, 7.0]
    assert keys["Metrics/reference/b"]["settled_snapshots"] == 5
    assert summary["snapshots"] == 10


def test_keys_missing_from_some_snapshots_keep_their_own_steps() -> None:
    recorder = ResetLogRecorder()
    recorder.record(0, {"Metrics/x": 1.0}, reset_count=1)
    recorder.record(9, {"Metrics/x": 3.0, "Metrics/y": 4.0}, reset_count=1)
    keys = recorder.summary(10)["keys"]
    assert keys["Metrics/x"]["settled_mean"] == pytest.approx(3.0)
    assert keys["Metrics/y"]["settled_mean"] == pytest.approx(4.0)
