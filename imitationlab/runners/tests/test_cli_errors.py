"""Command-line error reporting, help names, and shared task constants."""

from __future__ import annotations

import subprocess
import sys
from importlib import metadata

import pytest
import torch

import imitationlab
from imitationlab.cli import main
from imitationlab.runners.errors import UserError
from imitationlab.runners.registry import (
    DEFAULT_TASK,
    ENTRY_POINT_ALGORITHM_MAP,
    agent_entry_point_key,
    is_eager_task,
    resolve_agent_cfg_entry_point,
)


def _checkpoint(tmp_path):
    model = tmp_path / "model.pt"
    torch.save(
        {"imitationlab": {"argv": ["--task", DEFAULT_TASK, "--algo", "PPO"]}}, model
    )
    return model


def test_missing_checkpoint_is_reported_by_the_subcommand(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["evaluate", "--checkpoint", str(tmp_path / "absent.pt"), "--dry-run"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "usage: imitationlab evaluate" in err
    assert "imitationlab evaluate: error: Checkpoint does not exist" in err


def test_unexpected_errors_keep_their_traceback(tmp_path, monkeypatch):
    from imitationlab.runners import artifacts

    def broken(_):
        raise ValueError("internal defect")

    monkeypatch.setattr(artifacts, "checkpoint_run_spec", broken)
    with pytest.raises(ValueError, match="internal defect"):
        main(["evaluate", "--checkpoint", str(_checkpoint(tmp_path)), "--dry-run"])


def test_evaluate_refuses_to_overwrite_a_result(tmp_path, capsys):
    model = _checkpoint(tmp_path)
    output = tmp_path / "result.json"
    output.write_text("{}")
    argv = ["evaluate", "--checkpoint", str(model), "--output", str(output)]
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2
    assert "--overwrite" in capsys.readouterr().err
    # A dry run only prints the command, and --overwrite opts in.
    assert main([*argv, "--dry-run"]) == 0
    assert main([*argv, "--overwrite", "--dry-run"]) == 0
    assert output.read_text() == "{}"


@pytest.mark.parametrize(
    ("argv", "prog"),
    [
        (["train"], "imitationlab train"),
        (["benchmark"], "imitationlab benchmark"),
        (["benchmark", "run"], "imitationlab benchmark run"),
        (["benchmark", "report"], "imitationlab benchmark report"),
    ],
)
def test_help_names_the_invoked_command(argv, prog):
    result = subprocess.run(
        [sys.executable, "-m", "imitationlab", *argv, "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.startswith(f"usage: {prog} ")


def test_top_level_help_describes_every_command():
    result = subprocess.run(
        [sys.executable, "-m", "imitationlab", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    from imitationlab.cli import COMMANDS

    for name, text in COMMANDS.items():
        assert name in result.stdout and text in result.stdout


def test_version_comes_from_package_metadata():
    try:
        expected = metadata.version("imitationlab")
    except metadata.PackageNotFoundError:
        expected = "0.0.0+unknown"
    assert imitationlab.__version__ == expected


def test_benchmark_rejects_reused_output_with_subcommand_usage(tmp_path, capsys):
    output = tmp_path / "out"
    output.mkdir()
    (output / "result.json").write_text("{}")
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "benchmark",
                "run",
                "--dataset",
                "lafan1",
                "--data",
                str(tmp_path),
                "--policy",
                "ppo",
                "--output",
                str(output),
            ]
        )
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "usage: imitationlab benchmark run" in err
    assert "existing benchmark results are immutable" in err


def test_benchmark_failure_prints_the_error(tmp_path, capsys):
    from imitationlab.runners.benchmark import _report_failure

    path = tmp_path / "result.json"
    assert _report_failure({"status": "completed"}, path) == 0
    assert (
        _report_failure({"status": "failed", "error": "Training exited 3"}, path) == 1
    )
    err = capsys.readouterr().err
    assert "Training exited 3" in err and str(path) in err


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        (DEFAULT_TASK, True),
        (f"isaaclab_tasks:{DEFAULT_TASK}", True),
        ("CartPole-v1", False),
    ],
)
def test_eager_task_predicate(task, expected):
    assert is_eager_task(task) is expected


def test_evaluate_applies_pinned_reset_settings_to_the_task(tmp_path, capsys):
    model = tmp_path / "model.pt"
    torch.save(
        {
            "imitationlab": {
                "argv": ["--task", DEFAULT_TASK, "--algo", "PPO"],
                "task": DEFAULT_TASK,
            }
        },
        model,
    )
    assert main(["evaluate", "--checkpoint", str(model), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "env.reset_fail_share=0.0" in out and "env.pinned_reset=true" in out


def test_agent_entry_point_keys_are_stable():
    # Saved runs record these registry keys through --agent.
    assert ENTRY_POINT_ALGORITHM_MAP == {
        "rlopt_ppo_cfg_entry_point": "PPO",
        "rlopt_wmbptt_cfg_entry_point": "WMBPTT",
    }
    assert agent_entry_point_key("WMBPTT") == "rlopt_wmbptt_cfg_entry_point"


def test_agent_entry_point_resolution(monkeypatch):
    import gymnasium as gym

    task = "ImitationLabResolutionProbe-v0"
    gym.register(
        id=task,
        entry_point="gymnasium.envs.classic_control:CartPoleEnv",
        kwargs={"rlopt_ppo_cfg_entry_point": "pkg:Cfg"},
    )
    try:
        assert resolve_agent_cfg_entry_point(task, "PPO") == "rlopt_ppo_cfg_entry_point"
        assert resolve_agent_cfg_entry_point(task, "PPO", "custom_key") == "custom_key"
        assert resolve_agent_cfg_entry_point(None, "WMBPTT") == (
            "rlopt_wmbptt_cfg_entry_point"
        )
        with pytest.raises(UserError, match=r"no agent config.*\['PPO'\]"):
            resolve_agent_cfg_entry_point(task, "WMBPTT")
        with pytest.raises(UserError, match="not registered"):
            resolve_agent_cfg_entry_point("NoSuchImitationTask-v0", "PPO")
    finally:
        gym.registry.pop(task, None)


def test_user_error_is_a_value_error():
    # Callers that caught ValueError from artifact helpers keep working.
    assert issubclass(UserError, ValueError)
