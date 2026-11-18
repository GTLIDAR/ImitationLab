"""Public workflow contracts independent of the simulator."""

import json
import shlex
import shutil
import subprocess
import sys

import pytest
import torch

from imitationlab.cli import main
from imitationlab.runners.artifacts import checkpoint_run_spec, file_sha256
from imitationlab.runners.export import main as export


@pytest.mark.parametrize(
    "command",
    [
        [],
        ["train"],
        ["pretrain"],
        ["evaluate"],
        ["export"],
        ["data"],
        ["hardware"],
        ["tune"],
    ],
)
def test_help_does_not_require_simulator(command):
    result = subprocess.run(
        [sys.executable, "-m", "imitationlab", *command, "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "usage:" in result.stdout
    assert "Isaac Sim" not in result.stderr


def test_legacy_checkpoint_recovers_architecture_and_evaluation_data(tmp_path, capsys):
    model = tmp_path / "models/model_step_32.pt"
    model.parent.mkdir()
    torch.save({"policy_state_dict": {}}, model)
    (tmp_path / "command.txt").write_text(
        "python scripts/tracking/train.py --task imitation-g1-v1 --algo WMBPTT --seed 17 "
        "--max_iterations 1 agent.wm.dynamics_model=diffsr agent.wm.actor_hidden=[64,64,64] "
        "env.data.reference_arrays_dir=/old/bank agent.logger.log_dir=/old/run"
    )
    manifest = tmp_path / "eval motions.json"
    manifest.write_text("{}")
    assert (
        main(
            [
                "evaluate",
                "--checkpoint",
                str(model),
                "--data",
                str(manifest),
                "--dry-run",
            ]
        )
        == 0
    )
    args = shlex.split(capsys.readouterr().out)
    assert args[args.index("--algo") + 1] == "WMBPTT"
    assert args[args.index("--seed") + 1] == "17"
    assert "agent.wm.actor_hidden=[64,64,64]" in args
    assert "agent.wm.dynamics_model=diffsr" in args
    assert not any("/old/" in arg for arg in args)
    assert "--max_iterations" not in args
    assert f"env.data.manifest={json.dumps(str(manifest))}" in args


def test_resume_recovers_original_learner_and_respects_new_budget(tmp_path, capsys):
    model = tmp_path / "model.pt"
    torch.save(
        {
            "imitationlab": {
                "argv": [
                    "--task",
                    "imitation-g1-v1",
                    "--algo",
                    "WMBPTT",
                    "agent.wm.actor_source=real",
                    "--max_iterations",
                    "1",
                ]
            }
        },
        model,
    )
    assert (
        main(
            ["train", "--checkpoint", str(model), "--max-iterations", "10", "--dry-run"]
        )
        == 0
    )
    args = shlex.split(capsys.readouterr().out)
    assert "agent.wm.actor_source=real" in args
    assert args[-2:] == ["--max_iterations", "10"]


def test_native_export_is_relocatable_and_preserves_weights(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    encoder = source / "encoder.pt"
    encoder.write_bytes(b"test encoder artifact")
    model = source / "model.pt"
    weight = torch.arange(12).reshape(3, 4)
    metadata = {
        "argv": ["--task", "imitation-g1-v1", f"env.skill_encoder_path={encoder}"],
        "encoder": {"path": str(encoder), "sha256": file_sha256(encoder)},
    }
    torch.save(
        {"policy_state_dict": {"weight": weight}, "imitationlab": metadata}, model
    )
    bundle = tmp_path / "export"
    assert export(["--checkpoint", str(model), "--output", str(bundle)]) == 0
    moved = tmp_path / "moved bundle"
    bundle.rename(moved)
    shutil.rmtree(source)
    spec = checkpoint_run_spec(moved / "model.pt")
    assert spec["encoder"]["path"] == str(moved / "encoder.pt")
    assert "${BUNDLE}" not in " ".join(spec["argv"])
    payload = torch.load(moved / "model.pt", weights_only=False)
    assert torch.equal(payload["policy_state_dict"]["weight"], weight)
    (moved / "encoder.pt").write_bytes(b"different encoder")
    with pytest.raises(ValueError, match="hash mismatch"):
        checkpoint_run_spec(moved / "model.pt")
