"""Interface names: ``spectral`` is the continuous skill latent; ``diffsr`` names only the learner."""

from __future__ import annotations

import pytest

from imitationlab.runners.registry import INTERFACES


def test_interfaces_name_the_spectral_skill_latent() -> None:
    assert INTERFACES == ("explicit", "fsq", "spectral")


def test_the_command_line_accepts_only_the_current_names(capsys) -> None:
    from imitationlab.cli import main

    with pytest.raises(SystemExit) as raised:
        main(["train", "--policy", "supermimic", "--interface", "diffsr", "--dry-run"])
    assert raised.value.code == 2
    assert "invalid choice: 'diffsr'" in capsys.readouterr().err


def test_a_run_without_a_recorded_interface_infers_spectral(tmp_path) -> None:
    from imitationlab.runners.launch import _saved_selection

    spec = {
        "argv": [
            "--algo",
            "WMBPTT",
            f"env.skill_encoder_path={tmp_path / 'encoder.pt'}",
            "env.skill_encoder_mode=deterministic",
        ]
    }
    assert _saved_selection(spec)["interface"] == "spectral"


def test_the_spectral_encoder_recipe_is_the_only_spectral_recipe() -> None:
    from imitationlab.paths import recipes_root

    encoders = recipes_root() / "encoders"
    assert "--latent_mode deterministic" in (encoders / "spectral.args").read_text()
    assert not (encoders / "diffsr.args").exists()
