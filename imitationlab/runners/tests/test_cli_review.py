"""Regression tests for restoring saved runs through the public CLI."""

import json
import os
import shlex
import sys

import pytest
import torch

from imitationlab.cli import main


def _overrides(command: str) -> dict[str, str]:
    return dict(
        item.split("=", 1)
        for item in shlex.split(command)
        if item.startswith("env.data.") and "=" in item
    )


@pytest.mark.parametrize("action", ["train", "evaluate"])
@pytest.mark.parametrize("packed", [False, True], ids=["manifest", "packed"])
def test_replacing_saved_data_switches_source_modes(tmp_path, capsys, action, packed):
    old_source = (
        [
            "env.data.manifest=/old/motions.json",
            "env.data.reference_arrays_dir=null",
        ]
        if packed
        else [
            "env.data.manifest=null",
            "env.data.reference_arrays_dir=/old/packed",
            "env.data.persist_id=previous",
        ]
    )
    model = tmp_path / "model.pt"
    torch.save(
        {
            "imitationlab": {
                "argv": ["--task", "imitation-g1-v1", "--algo", "PPO", *old_source]
            }
        },
        model,
    )
    if packed:
        source = tmp_path / "packed bank"
        source.mkdir()
        (source / "reference_arrays_manifest.json").write_text(
            json.dumps({"key": {"source": {"persist_id": "replacement"}}})
        )
    else:
        source = tmp_path / "motions.json"
        source.write_text("{}")

    assert (
        main([action, "--checkpoint", str(model), "--data", str(source), "--dry-run"])
        == 0
    )
    overrides = _overrides(capsys.readouterr().out)
    if packed:
        assert overrides["env.data.manifest"] == "null"
        assert json.loads(overrides["env.data.reference_arrays_dir"]) == str(source)
        assert json.loads(overrides["env.data.persist_id"]) == "replacement"
    else:
        assert json.loads(overrides["env.data.manifest"]) == str(source)
        assert overrides["env.data.reference_arrays_dir"] == "null"
        assert overrides["env.data.persist_id"] == "null"


def test_legacy_resume_restores_recipe_environment_before_exec(tmp_path, monkeypatch):
    model = tmp_path / "model.pt"
    torch.save({"policy_state_dict": {}}, model)
    recipe = tmp_path / "settings.args"
    recipe.write_text(
        "%env ISAACLAB_IMITATION_COLLECT_FASTPATH=1\n"
        "%env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n"
    )
    (tmp_path / "command.txt").write_text(
        shlex.join(
            [
                "python",
                "scripts/tracking/train.py",
                "--task",
                "imitation-g1-v1",
                "--args_file",
                str(recipe),
            ]
        )
    )
    monkeypatch.setattr(os, "environ", os.environ.copy())
    monkeypatch.delenv("ISAACLAB_IMITATION_COLLECT_FASTPATH", raising=False)
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")

    class ExecCalled(Exception):
        pass

    def execv(executable, command):
        assert os.environ["ISAACLAB_IMITATION_COLLECT_FASTPATH"] == "1"
        assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:False"
        assert command[-2:] == ["--checkpoint", str(model)]
        raise ExecCalled

    monkeypatch.setattr("os.execv", execv)
    with pytest.raises(ExecCalled):
        main(["train", "--checkpoint", str(model)])


def test_evaluate_does_not_ignore_encoder_identity_when_overriding_data(tmp_path):
    encoder = tmp_path / "encoder.pt"
    encoder.write_bytes(b"changed")
    model = tmp_path / "model.pt"
    torch.save(
        {
            "imitationlab": {
                "argv": ["--task", "imitation-g1-v1", "--algo", "PPO"],
                "encoder": {"path": str(encoder), "sha256": "previous hash"},
            }
        },
        model,
    )
    data = tmp_path / "motions.json"
    data.write_text("{}")

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "evaluate",
                "--checkpoint",
                str(model),
                "--data",
                str(data),
                "--policy",
                "ppo",
                "--dry-run",
            ]
        )
    assert exc.value.code == 2


@pytest.mark.parametrize("action", ["train", "evaluate"])
def test_checkpoint_restores_resolved_seed_and_wmbptt(tmp_path, capsys, action):
    model = tmp_path / "model.pt"
    torch.save(
        {
            "imitationlab": {
                "task": "imitation-g1-v1",
                "policy": "svg",
                "seed": 1729,
                "argv": [
                    "--task",
                    "imitation-g1-v1",
                    "--algorithm",
                    "WMBPTT",
                    "--seed",
                    "-1",
                ],
            }
        },
        model,
    )

    assert main([action, "--checkpoint", str(model), "--dry-run"]) == 0
    argv = shlex.split(capsys.readouterr().out)
    seed_indices = [index for index, value in enumerate(argv) if value == "--seed"]
    assert argv[seed_indices[-1] + 1] == "1729"
    if action == "evaluate":
        assert argv[argv.index("--algo") + 1] == "WMBPTT"


def test_installed_recipes_are_available_without_a_checkout(tmp_path, monkeypatch):
    from imitationlab import paths

    installed = tmp_path / "prefix/share/imitationlab/recipes"
    installed.mkdir(parents=True)
    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path / "workspace")
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "prefix"))
    assert paths.recipes_root() == installed


def test_every_recipe_file_is_installed_with_the_package():
    """A wheel ships only the listed data files; an unlisted recipe is missing there."""
    import tomllib

    from imitationlab.paths import REPO_ROOT

    data_files = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["tool"][
        "setuptools"
    ]["data-files"]
    listed = {path for paths in data_files.values() for path in paths}
    recipes = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "experiments" / "recipes").rglob("*.args")
    }
    assert recipes <= listed, sorted(recipes - listed)


@pytest.mark.parametrize("action", ["train", "evaluate"])
def test_replacing_manifest_does_not_reuse_saved_cache(tmp_path, capsys, action):
    model = tmp_path / "model.pt"
    torch.save(
        {
            "imitationlab": {
                "argv": [
                    "--task",
                    "imitation-g1-v1",
                    "--algo",
                    "PPO",
                    "env.data.manifest=/old/motions.json",
                    "env.data.cache_dir=/old/zarr",
                    "env.data.persist_dir=/old/persisted",
                ]
            }
        },
        model,
    )
    source = tmp_path / "new_motions.json"
    source.write_text("{}")

    assert (
        main([action, "--checkpoint", str(model), "--data", str(source), "--dry-run"])
        == 0
    )
    overrides = _overrides(capsys.readouterr().out)
    assert overrides.get("env.data.cache_dir", "null") == "null"
    assert overrides.get("env.data.persist_dir", "null") == "null"


def test_checkpoint_directory_ignores_scratch_files_and_subdirectories(tmp_path):
    from imitationlab.runners.artifacts import resolve_checkpoint

    valid = tmp_path / "model_step_640.pt"
    valid.touch()
    os.utime(valid, (1700000000, 1700000000))
    (tmp_path / "model_step_scratch.pt").touch()
    (tmp_path / "model_step_960.pt").mkdir()
    assert resolve_checkpoint(tmp_path) == valid


@pytest.mark.parametrize("embedded_selection", [False, True])
def test_repeating_resume_selectors_preserves_saved_architecture(
    tmp_path, capsys, embedded_selection
):
    model = tmp_path / "model.pt"
    spec = {
        "argv": [
            "--task",
            "imitation-g1-v1",
            "--algorithm=PPO",
            "agent.policy.num_cells=[96,48]",
            "--max_iterations=25",
        ],
    }
    if embedded_selection:
        spec.update(task="imitation-g1-v1", policy="ppo", interface="explicit")
    torch.save({"imitationlab": spec}, model)

    assert (
        main(
            [
                "train",
                "--checkpoint",
                str(model),
                "--policy=ppo",
                "--interface",
                "explicit",
                "--task",
                "imitation-g1-v1",
                "--max-iterations",
                "30",
                "--dry-run",
            ]
        )
        == 0
    )
    argv = shlex.split(capsys.readouterr().out)
    assert "agent.policy.num_cells=[96,48]" in argv
    # New-run normalization defaults must not change a saved policy's layers.
    assert not any(
        "normalize_input" in item or item.endswith("/ppo.args") for item in argv
    )
    assert "--algorithm=PPO" in argv
    assert argv[-2:] == ["--max_iterations", "30"]


@pytest.mark.parametrize(
    ("selector", "value", "saved"),
    [
        ("policy", "svg", "ppo"),
        ("interface", "fsq", "explicit"),
        ("task", "another-task", "imitation-g1-v1"),
    ],
)
def test_resume_rejects_conflicting_saved_contract(
    tmp_path, capsys, selector, value, saved
):
    model = tmp_path / "model.pt"
    torch.save(
        {
            "imitationlab": {
                "argv": ["--task", "imitation-g1-v1", "--algo", "PPO"],
                "task": "imitation-g1-v1",
                "policy": "ppo",
                "interface": "explicit",
            }
        },
        model,
    )

    with pytest.raises(SystemExit) as exc:
        main(["train", "--checkpoint", str(model), f"--{selector}", value, "--dry-run"])
    assert exc.value.code == 2
    assert (
        f"conflicts with the checkpoint's saved {selector} {saved!r}"
        in capsys.readouterr().err
    )


def test_standalone_legacy_resume_accepts_policy_and_data(tmp_path, capsys):
    model = tmp_path / "model.pt"
    torch.save({"policy_state_dict": {}}, model)
    data = tmp_path / "motions.json"
    data.write_text("{}")
    assert (
        main(
            [
                "train",
                "--checkpoint",
                str(model),
                "--policy",
                "ppo",
                "--data",
                str(data),
                "agent.policy.num_cells=[96,48]",
                "--dry-run",
            ]
        )
        == 0
    )
    argv = shlex.split(capsys.readouterr().out)
    assert argv[argv.index("--algo") + 1] == "PPO"
    assert "agent.policy.num_cells=[96,48]" in argv


def test_train_does_not_ignore_encoder_identity_when_repeating_policy(tmp_path, capsys):
    encoder = tmp_path / "encoder.pt"
    encoder.write_bytes(b"changed")
    model = tmp_path / "model.pt"
    torch.save(
        {
            "imitationlab": {
                "argv": ["--task", "imitation-g1-v1", "--algo", "PPO"],
                "encoder": {"path": str(encoder), "sha256": "previous hash"},
            }
        },
        model,
    )
    data = tmp_path / "motions.json"
    data.write_text("{}")

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "train",
                "--checkpoint",
                str(model),
                "--data",
                str(data),
                "--policy",
                "ppo",
                "--dry-run",
            ]
        )
    assert exc.value.code == 2
    assert "Checkpoint encoder hash mismatch" in capsys.readouterr().err


@pytest.mark.parametrize("interface", ["fsq", "spectral"])
def test_packed_pretraining_maps_arrays_on_host(tmp_path, capsys, interface):
    bank = tmp_path / "bank"
    bank.mkdir()
    (bank / "reference_arrays_manifest.json").write_text(
        json.dumps({"key": {"source": {"persist_id": "large-bank"}}})
    )
    main(["pretrain", "--interface", interface, "--data", str(bank), "--dry-run"])
    overrides = _overrides(capsys.readouterr().out)
    assert overrides["env.data.runtime_cache_device"] == "cpu"
    assert overrides["env.data.reference_arrays_resident"] == "false"
    assert overrides["env.data.reference_arrays_warm_workers"] == "0"


@pytest.mark.parametrize("interface", ["fsq", "spectral"])
@pytest.mark.parametrize("policy", ["ppo", "supermimic", "svg", "diffsr"])
def test_packed_latent_training_keeps_reference_bank_off_gpu(
    tmp_path, capsys, interface, policy
):
    bank = tmp_path / "bank"
    bank.mkdir()
    (bank / "reference_arrays_manifest.json").write_text(
        json.dumps({"key": {"source": {"persist_id": "large-bank"}}})
    )
    argv = [
        "train",
        "--interface",
        interface,
        "--policy",
        policy,
        "--data",
        str(bank),
        "--encoder",
        str(tmp_path / "encoder.pt"),
        "--dry-run",
    ]
    main(argv)
    resolved = shlex.split(capsys.readouterr().out)
    overrides = dict(
        item.split("=", 1)
        for item in resolved
        if item.startswith("env.") and "=" in item
    )
    assert overrides["env.data.runtime_cache_device"] == "cpu"
    assert overrides["env.library_host_mapped"] == "true"
    assert overrides["env.library_env_window"] == "64"
    assert "env.step_mode=graph" not in resolved
    main(
        [
            *argv,
            "env.data.runtime_cache_device=cuda",
            "env.library_host_mapped=false",
        ]
    )
    resolved = shlex.split(capsys.readouterr().out)
    overrides = dict(
        item.split("=", 1)
        for item in resolved
        if item.startswith("env.") and "=" in item
    )
    assert overrides["env.data.runtime_cache_device"] == "cuda"
    assert overrides["env.library_host_mapped"] == "false"


@pytest.mark.parametrize("interface", ["explicit", "fsq", "spectral"])
def test_public_ppo_normalizes_unscaled_observations(tmp_path, capsys, interface):
    from imitationlab.runners.args_file import expand_args_files

    argv = [
        "train",
        "--interface",
        interface,
        "--policy",
        "ppo",
        "--hardware",
        "off",
        "--dry-run",
    ]
    if interface != "explicit":
        argv += ["--encoder", str(tmp_path / "encoder.pt")]

    def resolved(extra):
        main([*argv, *extra])
        args = expand_args_files(shlex.split(capsys.readouterr().out), environ={})
        return dict(
            item.split("=", 1)
            for item in args
            if item.startswith("agent.") and "=" in item
        )

    values = resolved([])
    assert values["agent.policy.normalize_input"] == "true"
    assert values["agent.value_function.normalize_input"] == "true"
    assert (
        resolved(["agent.policy.normalize_input=false"])["agent.policy.normalize_input"]
        == "false"
    )
    assert "agent.policy.normalize_input" not in resolved(["--task", "custom-task-v0"])


def test_legacy_ppo_resume_does_not_add_normalizer_layers(tmp_path, capsys):
    from imitationlab.runners.args_file import expand_args_files

    checkpoint = tmp_path / "old.pt"
    torch.save({"policy_state_dict": {}}, checkpoint)
    data = tmp_path / "motions.json"
    data.write_text("{}")
    main(
        [
            "train",
            "--checkpoint",
            str(checkpoint),
            "--data",
            str(data),
            "--policy",
            "ppo",
            "--hardware",
            "off",
            "--dry-run",
        ]
    )
    args = expand_args_files(shlex.split(capsys.readouterr().out), environ={})
    assert not any("normalize_input" in item for item in args)


@pytest.mark.parametrize("interface", ["explicit", "fsq", "spectral"])
@pytest.mark.parametrize("policy", ["supermimic", "svg", "diffsr"])
@pytest.mark.parametrize("packed", [False, True])
def test_command_interfaces_share_fast_learner_defaults(
    tmp_path, capsys, interface, policy, packed
):
    from imitationlab.runners.args_file import expand_args_files

    bank = tmp_path / "bank"
    bank.mkdir()
    (bank / "reference_arrays_manifest.json").write_text(
        json.dumps({"key": {"source": {"persist_id": "test"}}})
    )
    data = bank if packed else tmp_path / "motions.json"
    if not packed:
        data.write_text("{}")
    argv = [
        "train",
        "--interface",
        interface,
        "--policy",
        policy,
        "--data",
        str(data),
        "--hardware",
        "off",
        "--dry-run",
    ]
    if interface != "explicit":
        argv += ["--encoder", str(tmp_path / "encoder.pt")]
    main(argv)
    resolved = expand_args_files(shlex.split(capsys.readouterr().out), environ={})
    values = dict(
        item.split("=", 1) for item in resolved if item.startswith("agent.wm.")
    )
    assert values["agent.wm.wm_warmup_iters"] == "0"
    assert values["agent.wm.wm_hidden"] == "[1024,1024,1024]"
    assert values["agent.wm.actor_lr"] == "0.0004"
    assert values["agent.wm.actor_source"] == (
        "real" if policy == "svg" else "imagined"
    )
    if policy == "supermimic":
        assert values["agent.wm.update_graphs"] == "true"
    if policy == "diffsr":
        assert values["agent.wm.wm_opt"] == "adam"
    main([*argv, "agent.wm.wm_warmup_iters=7"])
    resolved = expand_args_files(shlex.split(capsys.readouterr().out), environ={})
    values = dict(
        item.split("=", 1) for item in resolved if item.startswith("agent.wm.")
    )
    assert values["agent.wm.wm_warmup_iters"] == "7"
