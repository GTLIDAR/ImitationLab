"""Hardware sizing, caller precedence, and calibration evidence contracts."""

from dataclasses import replace
import json
import shlex

import pytest
import torch

from imitationlab.cli import main
from imitationlab.runners.args_file import expand_args_files
from imitationlab.runners.artifacts import data_overrides, flag_value
from imitationlab.runners.calibration import CalibrationClock
from imitationlab.runners.hardware import (
    GIB,
    Hardware,
    MotionBank,
    choose_plan,
    inspect_bank,
)
from imitationlab.runners import tuning


def pc(vram=16, ram=32, free=None):
    return Hardware(
        16,
        ram * GIB,
        ram * GIB,
        gpu_name="test GPU",
        vram_total=vram * GIB,
        vram_free=int((free if free is not None else vram * 0.9) * GIB),
        compute_capability=(8, 6),
    )


BONES = MotionBank("packed", "bones-test", 47491234, 47491234 * 71 * 4, "bank-hash")
LAFAN = MotionBank("packed", "lafan-test", 441080, 441080 * 71 * 4, "small-bank-hash")


@pytest.mark.parametrize(
    "vram,profile,host",
    [
        (8, "small", True),
        (16, "standard", True),
        (24, "standard", True),
        (32, "standard", False),
        (48, "large", False),
        (80, "xlarge", False),
    ],
)
def test_full_bank_placement_and_hardware_tiers(vram, profile, host):
    plan = choose_plan(pc(vram, ram=64), BONES)
    assert not plan.errors
    assert plan.profile == profile
    assert f"env.library_host_mapped={str(host).lower()}" in plan.overrides
    assert f"env.library_env_window={64 if host else 0}" in plan.overrides


def test_free_memory_and_host_limits_change_selection():
    assert choose_plan(pc(48, free=18), BONES).profile == "standard"
    assert choose_plan(pc(16, free=5), BONES).errors
    assert choose_plan(pc(16, ram=8), BONES).errors
    resident = choose_plan(pc(32, ram=8), BONES)
    assert not resident.errors
    assert "env.library_host_mapped=false" in resident.overrides
    assert choose_plan(pc(24, ram=8), BONES).profile == "small"
    assert choose_plan(pc(16), BONES, profile="xlarge").errors


def test_ppo_full_rollouts_need_more_headroom_than_world_model_training():
    assert choose_plan(pc(20), BONES, policy="supermimic").profile == "standard"
    ppo = choose_plan(pc(20), BONES, policy="ppo")
    assert ppo.profile == "small"
    assert not ppo.errors
    assert choose_plan(pc(20), BONES, policy="ppo", profile="standard").errors


@pytest.mark.parametrize("bank", [LAFAN, BONES])
@pytest.mark.parametrize("policy", ["svg", "diffsr"])
@pytest.mark.parametrize("interface", ["fsq", "spectral"])
def test_h100_latent_learners_use_measured_profile_with_manual_escape(
    bank, policy, interface
):
    hardware = replace(pc(80, ram=128, free=79), gpu_name="NVIDIA H100 80GB HBM3")
    plan = choose_plan(hardware, bank, policy=policy, interface=interface)
    assert not plan.errors and plan.profile == "standard"
    assert "agent.wm.imagine_batch=2048" in plan.overrides
    manual = choose_plan(
        hardware, bank, policy=policy, interface=interface, profile="large"
    )
    assert not manual.errors and manual.profile == "large"
    limited = choose_plan(
        replace(hardware, vram_free=20 * GIB), bank, policy=policy, interface=interface
    )
    assert not limited.errors and limited.profile == "small"


def test_h100_measured_cap_does_not_change_other_workloads():
    hardware = replace(pc(80, ram=128, free=79), gpu_name="NVIDIA H100 80GB HBM3")
    assert (
        choose_plan(hardware, LAFAN, policy="supermimic", interface="fsq").profile
        == "xlarge"
    )
    assert (
        choose_plan(hardware, LAFAN, policy="ppo", interface="fsq").profile == "large"
    )
    assert choose_plan(hardware, LAFAN, policy="svg").profile == "xlarge"
    assert choose_plan(hardware, LAFAN, policy="diffsr").profile == "xlarge"
    assert (
        choose_plan(
            hardware, LAFAN, policy="diffsr", interface="fsq", stage="pretrain"
        ).profile
        == "xlarge"
    )
    assert (
        choose_plan(
            replace(hardware, gpu_name="NVIDIA H200"),
            LAFAN,
            policy="diffsr",
            interface="fsq",
        ).profile
        == "xlarge"
    )


@pytest.mark.parametrize("policy", ["ppo", "supermimic", "svg", "diffsr"])
@pytest.mark.parametrize("interface", ["explicit", "fsq", "spectral"])
@pytest.mark.parametrize("bank", [BONES, LAFAN], ids=["bones", "lafan1"])
def test_all_combinations_only_change_execution_sizes(policy, interface, bank):
    plan = choose_plan(pc(), bank, policy=policy, interface=interface)
    assert not plan.errors
    assert ("agent.wm.imagine_batch=" in " ".join(plan.overrides)) == (policy != "ppo")
    assert all(
        value.startswith(
            (
                "env.library_",
                "agent.wm.imagine_batch=",
                "agent.wm.wm_batch_size=",
                "--num-envs",
            )
        )
        or value.isdigit()
        for value in plan.overrides
    )


def test_no_cuda_is_actionable_and_off_needs_no_gpu():
    cpu = Hardware(4, 16 * GIB, 8 * GIB)
    assert choose_plan(cpu, LAFAN).errors == ["No CUDA GPU detected."]
    off = choose_plan(cpu, LAFAN, profile="off")
    assert not off.errors and not off.overrides
    assert choose_plan(pc(), LAFAN, stage="pretrain").overrides == [
        "--num-envs",
        "4",
        "--batch-size",
        "128",
    ]
    assert (
        choose_plan(replace(pc(), cpu_count=2), LAFAN).environment[
            "TORCHINDUCTOR_COMPILE_THREADS"
        ]
        == "2"
    )


def bank_at(tmp_path):
    bank = tmp_path / "bank with spaces"
    bank.mkdir()
    (bank / "reference_arrays_manifest.json").write_text(
        json.dumps(
            {
                "key": {"source": {"persist_id": "test"}},
                "traj_info": {"written": 441080, "capacity": 500000},
            }
        )
    )
    return bank


def test_bank_uses_actual_written_frames(tmp_path):
    bank = inspect_bank(bank_at(tmp_path))
    assert bank.rows == 441080
    assert bank.library_bytes == LAFAN.library_bytes
    assert len(bank.manifest_sha256) == 64


def test_auto_is_default_and_caller_args_file_wins(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(
        "imitationlab.runners.hardware.detect_hardware", lambda device: pc()
    )
    bank = bank_at(tmp_path)
    user_args = tmp_path / "user.args"
    user_args.write_text(
        "--num-envs 32\nagent.wm.imagine_batch=16\nenv.library_host_mapped=true\n"
    )
    main(["train", "--data", str(bank), "--args-file", str(user_args), "--dry-run"])
    argv = expand_args_files(shlex.split(capsys.readouterr().out), environ={})
    values = dict(x.split("=", 1) for x in argv if "=" in x)
    assert flag_value(argv, "--num-envs") == "32"
    assert values["agent.wm.imagine_batch"] == "16"
    assert values["env.library_host_mapped"] == "true"
    assert "agent.wm.wm_batch_size=2048" in argv
    assert "env.step_mode=graph" in argv


def test_resume_and_off_skip_hardware_detection(tmp_path, capsys, monkeypatch):
    def fail(*args):
        pytest.fail("Resume and off must not redetect or resize hardware")

    monkeypatch.setattr("imitationlab.runners.hardware.detect_hardware", fail)
    main(["train", "--hardware", "off", "--dry-run"])
    capsys.readouterr()
    ckpt = tmp_path / "model.pt"
    torch.save(
        {"imitationlab": {"argv": ["--task", "imitation-g1-v1", "--num-envs", "77"]}},
        ckpt,
    )
    main(["train", "--checkpoint", str(ckpt), "--dry-run"])
    assert flag_value(shlex.split(capsys.readouterr().out), "--num-envs") == "77"
    with pytest.raises(SystemExit):
        main(["train", "--checkpoint", str(ckpt), "--hardware", "small", "--dry-run"])


def test_pretraining_override_precedence(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(
        "imitationlab.runners.hardware.detect_hardware", lambda device: pc()
    )
    main(
        [
            "pretrain",
            "--interface",
            "fsq",
            "--batch-size",
            "16",
            "--num-envs",
            "2",
            "--dry-run",
        ]
    )
    argv = expand_args_files(shlex.split(capsys.readouterr().out), environ={})
    assert flag_value(argv, "--batch-size") == "16"
    assert flag_value(argv, "--num-envs") == "2"


def test_calibration_excludes_warmup_and_requires_two_complete_updates(tmp_path):
    clock = CalibrationClock(str(tmp_path / "timing.json"), seconds=10, warmup=2)
    assert not clock.step(100, now=10)
    assert not clock.step(200, now=100)
    assert not clock.step(300, now=110)
    assert clock.step(400, now=120)
    measured = json.loads(clock.output.read_text())
    assert measured == {
        "warmup_iterations": 2,
        "measured_iterations": 2,
        "measured_seconds": 20,
        "measured_frames": 200,
        "frames_per_second": 10,
    }
    assert clock.step(500, now=200)
    assert json.loads(clock.output.read_text()) == measured


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf")])
def test_calibration_requires_finite_positive_duration(seconds):
    with pytest.raises(ValueError):
        CalibrationClock("unused.json", seconds, 5)


def test_saved_selection_rejects_workload_or_memory_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(tuning, "source_fingerprint", lambda: "source")
    bank = bank_at(tmp_path)
    argv = data_overrides(bank)
    plan = choose_plan(pc(), inspect_bank(bank))
    saved = {
        "context": tuning.context_signature(plan, argv, None),
        "winner": {
            "profile": plan.profile,
            "overrides": plan.overrides,
            "frames_per_second": 100.0,
        },
    }
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(saved))
    assert (
        tuning.load_selection(path, plan, argv + ["--seed", "3"], None).calibration[
            "frames_per_second"
        ]
        == 100
    )
    with pytest.raises(ValueError, match="does not match"):
        tuning.load_selection(path, plan, argv + ["agent.wm.horizon=5"], None)
    low_memory = choose_plan(replace(pc(), vram_free=7 * GIB), plan.bank)
    with pytest.raises(ValueError, match="Available memory"):
        tuning.load_selection(path, low_memory, argv, None)
    monkeypatch.setattr(tuning, "source_fingerprint", lambda: "different-source")
    with pytest.raises(ValueError, match="does not match"):
        tuning.load_selection(path, plan, argv, None)


def test_calibration_requires_finite_updated_checkpoint(tmp_path):
    path = tmp_path / "model.pt"

    def save(step, tensor):
        torch.save(
            {
                "weight": tensor,
                "optimizers": {
                    "actor": {"nested_adam": {"state": {0: {"step": step}}}}
                },
            },
            path,
        )

    save(3, torch.ones(2))
    assert tuning.validate_checkpoint(path)["optimizer_steps"] == {"actor": 3}
    save(0, torch.ones(2))
    with pytest.raises(ValueError, match="Missing optimizer"):
        tuning.validate_checkpoint(path)
    save(3, torch.tensor([float("nan")]))
    with pytest.raises(ValueError, match="Non-finite"):
        tuning.validate_checkpoint(path)


def test_tune_records_failures_and_only_selects_verified_runs(tmp_path, monkeypatch):
    from pathlib import Path

    monkeypatch.setattr(tuning, "detect_hardware", lambda device: pc())
    monkeypatch.setattr(tuning, "source_fingerprint", lambda: "source")
    bank = bank_at(tmp_path)
    output = tmp_path / "tuning"

    def trial(command, env, log, timeout):
        assert "--max-train-seconds" not in command
        assert "--max-train-seconds" not in command
        assert "agent.wm.wm_warmup_iters=0" in command
        assert "agent.save_interval=1000000000000" in command
        profile = flag_value(command, "--hardware")
        log.write_text("simulated subprocess output")
        if profile == "small":
            return 1
        settings = json.loads(env["IMITATIONLAB_CALIBRATION"])
        Path(settings["output"]).write_text(json.dumps({"frames_per_second": 100.0}))
        destination = next(
            x.split("=", 1)[1] for x in command if x.startswith("agent.logger.log_dir=")
        )
        checkpoint = Path(destination) / "models" / "model_step_100.pt"
        checkpoint.parent.mkdir(parents=True)
        torch.save(
            {
                "weight": torch.ones(2),
                "optimizer_state_dict": {"state": {0: {"step": 3}}},
            },
            checkpoint,
        )
        return 0

    monkeypatch.setattr(tuning, "run_trial", trial)
    assert (
        tuning.main(
            [
                "--data",
                str(bank),
                "--output",
                str(output),
                "--candidates",
                "small",
                "standard",
            ],
            calibrate=True,
        )
        == 0
    )
    saved = json.loads((output / "selection.json").read_text())
    assert saved["winner"]["profile"] == "standard"
    assert [x["status"] for x in saved["trials"]] == ["failed", "passed"]
    plan = choose_plan(pc(), inspect_bank(bank))
    assert (
        tuning.load_selection(
            output / "selection.json", plan, data_overrides(bank), None
        ).profile
        == "standard"
    )
    saved["winner"] = None
    (output / "selection.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="no valid winning"):
        tuning.load_selection(
            output / "selection.json", plan, data_overrides(bank), None
        )
