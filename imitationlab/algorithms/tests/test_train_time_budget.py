# ruff: noqa: E402 -- filter upstream import deprecations before importing TorchRL.
"""``trainer.max_train_seconds``: walltime budget of the PPO loop."""

from __future__ import annotations

import warnings

warnings.filterwarnings(
    "ignore",
    message="`torch.jit.script_method` is deprecated.*",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Creating .* which inherits from WeightUpdaterBase is deprecated.*",
    category=DeprecationWarning,
)
from pathlib import Path

import pytest
import torch

from imitationlab.algorithms.ppo.ppo import PPO, PPOTrainerConfig
from imitationlab.algorithms.common.config import TrainerConfig
from imitationlab.algorithms.common.env_utils import make_parallel_env

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:The env passed to Collector is missing transforms.*:FutureWarning"
    ),
]

NUM_ENVS = 4


def _steps(agent) -> list[int]:
    return sorted(
        int(p.stem.rsplit("_", 1)[1]) for p in agent.log_dir.glob("**/model_step_*.pt")
    )


def _budget_after(agent, frames: int) -> None:
    """Make the budget run out once ``frames`` cumulative frames are processed.

    Wraps the real check so the clock start and the stop path are exercised,
    without making the test depend on machine speed.
    """

    def reached() -> bool:
        assert agent._train_clock_start is not None, "clock must run before checks"
        return agent._frames_after_collect() >= frames

    agent._train_time_budget_reached = reached  # type: ignore[method-assign]


def _capture_metrics(agent) -> list[tuple[dict, int]]:
    logged: list[tuple[dict, int]] = []
    original = agent.log_metrics

    def log_metrics(metrics, step=None, **kwargs):
        logged.append((dict(metrics), int(step) if step is not None else -1))
        return original(metrics, step=step, **kwargs)

    agent.log_metrics = log_metrics  # type: ignore[method-assign]
    return logged


# ------------------------------------------------------------------ config/base
def test_trainer_default_has_no_budget():
    assert TrainerConfig().max_train_seconds is None


def test_loop_support_marker():
    assert getattr(PPO.train, "_honours_train_time_budget", False)


# ------------------------------------------------------------------- PPO
def _ppo_cfg(tmp_path: Path, total_frames: int) -> PPOTrainerConfig:
    cfg = PPOTrainerConfig()
    cfg.env.env_name = "Pendulum-v1"
    cfg.env.device = cfg.device = "cpu"
    cfg.collector.frames_per_batch = 4
    cfg.collector.total_frames = total_frames
    cfg.replay_buffer.size = 4
    cfg.loss.mini_batch_size = 4
    cfg.loss.epochs = 1
    cfg.compile.compile = False
    cfg.logger.backend = ""
    cfg.logger.log_dir = str(tmp_path)
    cfg.save_interval = 1_000_000
    cfg.policy.input_keys = ["observation"]
    cfg.policy.num_cells = [8]
    cfg.value_function.input_keys = ["observation"]
    cfg.value_function.num_cells = [8]
    return cfg


def _ppo(tmp_path: Path, total_frames: int, budget: float | None) -> PPO:
    cfg = _ppo_cfg(tmp_path, total_frames)
    cfg.trainer.max_train_seconds = budget
    agent = PPO(make_parallel_env(cfg), cfg)
    agent._frames_after_collect = lambda: agent._probe_frames  # type: ignore[attr-defined]
    agent._probe_frames = 0  # type: ignore[attr-defined]
    original_collect = agent.collect

    def collect(run, idx):
        it = original_collect(run, idx)
        agent._probe_frames = run.frames_processed  # type: ignore[attr-defined]
        return it

    agent.collect = collect  # type: ignore[method-assign]
    return agent


def test_ppo_budget_stops_at_iteration_boundary_and_checkpoints(tmp_path):
    agent = _ppo(tmp_path, total_frames=40, budget=3600.0)
    assert agent.supports_train_time_budget
    _budget_after(agent, 12)
    logged = _capture_metrics(agent)
    agent.train()  # returns normally: a budget stop is a successful run
    assert agent._probe_frames == 12  # 3 of 10 iterations
    assert _steps(agent) == [12]
    ckpt = next(agent.log_dir.glob("**/model_step_12.pt"))
    data = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert data["cumulative_env_frames"] == 12
    assert data["cumulative_train_seconds"] > 0.0
    final = [m for m, _ in logged if m.get("train/stopped_by_time_budget") == 1.0]
    assert len(final) == 1
    assert final[0]["train/cumulative_env_frames"] == 12.0
    assert final[0]["time/train_elapsed_s"] > 0.0
    # Every per-iteration flush carries the time axis.
    per_iter = [m for m, _ in logged if "time/speed" in m]
    assert per_iter
    assert all("time/train_elapsed_s" in m for m in per_iter)
    assert [m["train/cumulative_env_frames"] for m in per_iter] == [4.0, 8.0, 12.0]


def test_ppo_without_budget_runs_the_frame_budget(tmp_path):
    agent = _ppo(tmp_path, total_frames=16, budget=None)
    agent.train()
    assert agent._probe_frames == 16
    assert agent._max_train_seconds() is None
    assert not agent._train_time_budget_reached()
    assert _steps(agent) == []  # save_interval not crossed; no budget checkpoint


def test_ppo_real_clock_budget(tmp_path):
    """A tiny real budget stops after the first iteration (clock is real)."""
    agent = _ppo(tmp_path, total_frames=400, budget=1e-6)
    agent.train()
    assert agent._probe_frames == 4
    assert _steps(agent) == [4]


def test_ppo_resume_continues_the_train_clock(tmp_path):
    agent = _ppo(tmp_path, total_frames=40, budget=3600.0)
    _budget_after(agent, 8)
    agent.train()
    ckpt = next(agent.log_dir.glob("**/model_step_8.pt"))
    used = torch.load(ckpt, map_location="cpu", weights_only=False)[
        "cumulative_train_seconds"
    ]
    resumed = _ppo(tmp_path / "resume", total_frames=40, budget=3600.0)
    resumed.load_model(str(ckpt))
    assert resumed._train_seconds_offset == pytest.approx(used)
    assert resumed.train_elapsed_seconds() == pytest.approx(used)
    # The budget is cumulative: a budget below the restored seconds stops at once.
    resumed.config.trainer.max_train_seconds = used / 2
    resumed.train()
    assert resumed._probe_frames == 12  # one iteration after the 8-frame resume


def test_ppo_interrupt_still_writes_resume_checkpoint_with_clock(tmp_path):
    agent = _ppo(tmp_path, total_frames=40, budget=3600.0)
    original = agent.collect
    calls = {"n": 0}

    def interrupted(run, idx):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt  # what the SIGTERM handler raises
        return original(run, idx)

    agent.collect = interrupted  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt):
        agent.train()
    assert _steps(agent) == [8]
    data = torch.load(
        next(agent.log_dir.glob("**/model_step_8.pt")),
        map_location="cpu",
        weights_only=False,
    )
    assert data["cumulative_env_frames"] == 8
    assert data["cumulative_train_seconds"] > 0.0
