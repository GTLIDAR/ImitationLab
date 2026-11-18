"""The ``diffsr`` command interface is now ``spectral``; old records still resolve."""

from __future__ import annotations

import argparse
import json

import pytest

from imitationlab.runners.registry import (
    INTERFACES,
    canonical_interface,
    interface_arg,
)


def test_interfaces_name_the_spectral_skill_latent() -> None:
    assert INTERFACES == ("explicit", "fsq", "spectral")
    assert canonical_interface("diffsr") == "spectral"
    for name in INTERFACES:
        assert canonical_interface(name) == name


def test_the_old_flag_value_parses_with_a_deprecation_warning() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface", type=interface_arg, choices=INTERFACES)
    with pytest.warns(FutureWarning, match="--interface spectral"):
        assert parser.parse_args(["--interface", "diffsr"]).interface == "spectral"
    assert parser.parse_args(["--interface", "spectral"]).interface == "spectral"


def test_a_run_recorded_with_the_old_name_resumes_as_spectral(tmp_path) -> None:
    from imitationlab.runners.launch import _saved_selection

    spec = {
        "argv": ["--task", "imitation-g1-v1", "--algo", "WMBPTT"],
        "policy": "supermimic",
        "interface": "diffsr",
        "encoder": {"path": str(tmp_path / "encoder.pt")},
    }
    assert _saved_selection(spec)["interface"] == "spectral"


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


def test_a_frozen_encoder_recorded_as_diffsr_matches_spectral(tmp_path) -> None:
    from imitationlab.runners.artifacts import file_sha256
    from imitationlab.runners.benchmark import frozen_encoder_record

    weights = tmp_path / "encoder.pt"
    weights.write_bytes(b"frozen encoder")
    record = dict(
        status="completed",
        interface="diffsr",
        seed=42,
        source_sha256="a" * 64,
        board=dict(bank_sha256="bank"),
        encoder=str(weights),
        encoder_sha256=file_sha256(weights),
    )
    (tmp_path / "pretrain.json").write_text(json.dumps(record))
    assert (
        frozen_encoder_record(
            tmp_path,
            interface="spectral",
            seed=42,
            source_sha256="a" * 64,
            board=dict(bank_sha256="bank"),
        )
        == record
    )


def test_the_spectral_encoder_recipe_keeps_the_old_file_as_an_alias() -> None:
    from imitationlab.paths import recipes_root

    encoders = recipes_root() / "encoders"
    assert "--latent_mode deterministic" in (encoders / "spectral.args").read_text()
    assert (encoders / "diffsr.args").read_text().split() == [
        "--args_file",
        "spectral.args",
    ]
