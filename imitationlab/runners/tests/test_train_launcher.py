"""Public training launcher contract."""

import itertools
import shlex
import subprocess
import sys

import pytest

from imitationlab.paths import REPO_ROOT
from imitationlab.runners.args_file import expand_args_files


# The historical script entry point; it must keep accepting the public options.
LAUNCHER = REPO_ROOT / "scripts" / "train.py"


@pytest.mark.parametrize(
    ("interface", "policy"),
    itertools.product(
        ("explicit", "spectral", "fsq"), ("ppo", "supermimic", "svg", "diffsr")
    ),
)
def test_select_interface_and_policy(interface, policy, tmp_path):
    command = [
        sys.executable,
        str(LAUNCHER),
        "--dry-run",
        "--interface",
        interface,
        "--policy",
        policy,
    ]
    encoder = tmp_path / "encoder with space.pt"
    if interface != "explicit":
        command.extend(("--encoder", str(encoder)))
    command.extend(("--num-envs", "4"))
    resolved = shlex.split(subprocess.check_output(command, text=True))

    assert resolved[resolved.index("--task") + 1] == "imitation-g1-v1"
    assert resolved[resolved.index("--algo") + 1] == (
        "PPO" if policy == "ppo" else "WMBPTT"
    )
    if interface != "explicit":
        mode = {"fsq": "sonic_fsq", "spectral": "deterministic"}[interface]
        assert f"env.skill_encoder_mode={mode}" in resolved
        assert f'env.skill_encoder_path="{encoder}"' in resolved
    if policy != "ppo":
        source = "real" if policy == "svg" else "imagined"
        assert f"agent.wm.actor_source={source}" in resolved
    assert ("agent.wm.dynamics_model=diffsr" in resolved) == (policy == "diffsr")
    assert "--args-file" in resolved
    if policy == "ppo":
        assert any(arg.endswith("/g1/ppo.args") for arg in resolved)
    assert resolved[-2:] == ["--num-envs", "4"]


def test_latent_interface_requires_encoder():
    result = subprocess.run(
        [sys.executable, str(LAUNCHER), "--interface", "fsq", "--dry-run"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "requires --encoder" in result.stderr


@pytest.mark.parametrize("policy", [None, "ppo", "supermimic", "svg", "diffsr"])
def test_packed_motions_enable_the_compatible_training_defaults(policy, tmp_path):
    user_args = tmp_path / "train.args"
    user_args.write_text("env.data.reference_arrays_dir=/motions\n--num-envs 16\n")
    command = [sys.executable, str(LAUNCHER), "--dry-run"]
    if policy is not None:
        command.extend(("--policy", policy))
    command.extend(("--args-file", str(user_args)))

    resolved = shlex.split(subprocess.check_output(command, text=True))
    environ = {}
    expanded = expand_args_files(resolved[2:], environ=environ)
    assert environ["IMITATIONLAB_ENV_COLLECT_FASTPATH"] == "1"
    assert "env.step_mode=graph" in expanded
    assert expanded[-2:] == ["--num-envs", "16"]
    imagined_defaults = policy in (None, "supermimic")
    assert ("agent.wm.update_graphs=true" in expanded) == imagined_defaults
    assert ("env.wm_only_observations=true" in expanded) == (policy != "ppo")
    assert ("agent.wm.wm_batch_size=2048" in expanded) == (policy != "ppo")
    assert ("agent.wm.actor_source=real" in expanded) == (policy == "svg")
    if policy == "diffsr":
        assert "agent.wm.dynamics_model=diffsr" in expanded
        optimizers = [arg for arg in expanded if arg.startswith("agent.wm.wm_opt=")]
        assert optimizers[-1] == "agent.wm.wm_opt=adam"
        assert "agent.wm.wm_structured=true" not in expanded


def test_learned_commands_share_learner_defaults_without_explicit_simulator_overrides(
    tmp_path,
):
    command = [
        sys.executable,
        str(LAUNCHER),
        "--dry-run",
        "--interface",
        "fsq",
        "--encoder",
        str(tmp_path / "encoder.pt"),
        "env.data.reference_arrays_dir=/motions",
    ]
    resolved = shlex.split(subprocess.check_output(command, text=True))
    assert resolved[resolved.index("--algo") + 1] == "WMBPTT"
    expanded = expand_args_files(resolved, environ={})
    assert "env.step_mode=graph" not in expanded
    assert "agent.wm.wm_warmup_iters=0" in expanded
    assert "agent.wm.update_graphs=true" in expanded
