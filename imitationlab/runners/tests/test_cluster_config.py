"""Gate: cluster profiles and campaign specs resolve deterministically.

The frozen-env design only removes the allow-list failure class if every key
in a profile reaches the rendered env file. The packaged example profile stands
in for site profiles, which live outside the package.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from imitationlab.runners.pipeline.cluster.common import PipelineError
from imitationlab.runners.pipeline.cluster.config import (
    EXAMPLE_PROFILE_NAME,
    PROFILE_DIR_ENV,
    _package_profile_path,
    load_campaign,
    load_profile,
    resolve_profile_path,
    resolved_env,
)
from imitationlab.runners.pipeline.cluster.envfile import (
    render_job_env,
)

CAMPAIGN_YAML = """\
name: demo
profile: mycluster
wandb_project: proj
wandb_group: grp
vars:
  frame_cap: 100
preflight:
  require_container_paths: [/data/ref]
  output_container_path: /data/out/${vars.arm}_seed${vars.seed}
shared_env:
  CLUSTER_WANDB_TAGS: demo,${vars.arm}
arms:
  a1:
    stages:
      - name: pretrain
        executable: scripts/tracking/train_hl_skill_diffsr.py
        args: ["--seed", "${vars.seed}", "cap=${vars.frame_cap}"]
        env: {STAGE_ONLY: pre}
      - name: lowlevel
        executable: scripts/tracking/train.py
        args: ["cap=${vars.frame_cap}"]
        depends_on: pretrain
  a2:
    vars: {frame_cap: 999}
    stages:
      - name: pretrain
        executable: x.py
        args: ["cap=${vars.frame_cap}"]
"""


def _campaign_file(tmp_path: Path) -> Path:
    path = tmp_path / "campaign.yaml"
    path.write_text(CAMPAIGN_YAML)
    return path


def _load_example_profile():
    """The packaged example profile; site profiles live outside the package."""
    return load_profile(EXAMPLE_PROFILE_NAME)


def test_load_example_profile_from_package() -> None:
    profile = _load_example_profile()
    assert profile.name == "example"
    assert profile.slurm.log_dir.startswith("/")
    assert profile.extra_bind_paths
    assert all(path.startswith("/") for path in profile.extra_bind_paths)
    assert profile.min_free_gb > 0


def test_package_ships_only_the_example_profile() -> None:
    conf_dir = _package_profile_path(EXAMPLE_PROFILE_NAME).parent
    assert sorted(p.name for p in conf_dir.glob("profile_*.yaml")) == [
        "profile_example.yaml"
    ]


def _write_profile(directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    text = _package_profile_path(EXAMPLE_PROFILE_NAME).read_text(encoding="utf-8")
    path = directory / f"profile_{name}.yaml"
    path.write_text(text.replace("name: example", f"name: {name}"), encoding="utf-8")
    return path


def test_named_profile_resolves_env_dir_before_user_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_dir = tmp_path / "env_dir"
    config_home = tmp_path / "config"
    monkeypatch.setenv(PROFILE_DIR_ENV, str(env_dir))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    user_path = _write_profile(config_home / "imitationlab" / "cluster", "site")
    assert resolve_profile_path("site") == user_path
    env_path = _write_profile(env_dir, "site")
    assert resolve_profile_path("site") == env_path
    assert load_profile("site").name == "site"


def test_missing_named_profile_names_search_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(PROFILE_DIR_ENV, str(tmp_path / "none"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    with pytest.raises(PipelineError, match="profile_absent.yaml"):
        load_profile("absent")


def test_packaged_site_profile_is_deprecated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from imitationlab.runners.pipeline.cluster import config as config_module

    monkeypatch.delenv(PROFILE_DIR_ENV, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    legacy = _write_profile(tmp_path / "pkg", "legacy")
    monkeypatch.setattr(
        config_module,
        "_package_profile_path",
        lambda name: tmp_path / "pkg" / f"profile_{name}.yaml",
    )
    with pytest.warns(DeprecationWarning, match="deprecated"):
        assert resolve_profile_path("legacy") == legacy


def test_profile_rejects_relative_log_dir(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "name: t\nlogin: t\ncontrol_root: /a\ndata_dir: /b\nshared_sif_path: /c.sif\n"
        "isaac_cache_dir: /d\nslurm:\n  gres: gpu:a40:1\n  cpus_per_task: 4\n"
        "  mem: 8G\n  log_dir: logs/slurm\n"
    )
    with pytest.raises(PipelineError, match="log_dir must be absolute"):
        load_profile(str(bad))


def test_profile_rejects_fqdn_login(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "name: t\nlogin: login.cluster.example.edu\ncontrol_root: /a\ndata_dir: /b\n"
        "shared_sif_path: /c.sif\nisaac_cache_dir: /d\nslurm:\n  gres: gpu:a40:1\n"
        "  cpus_per_task: 4\n  mem: 8G\n  log_dir: /logs\n"
    )
    with pytest.raises(PipelineError, match="ssh alias"):
        load_profile(str(bad))


def test_campaign_resolution_interpolates_arm_and_seed(tmp_path: Path) -> None:
    jobset = load_campaign(_campaign_file(tmp_path), arm="a1", seed=3)
    assert jobset.output_container_path == "/data/out/a1_seed3"
    pretrain = jobset.stages[0]
    assert pretrain.args == ("--seed", "3", "cap=100")
    assert jobset.shared_env["CLUSTER_WANDB_TAGS"] == "demo,a1"
    assert jobset.stages[1].depends_on == "pretrain"


def test_campaign_arm_vars_and_cli_override(tmp_path: Path) -> None:
    path = _campaign_file(tmp_path)
    arm_local = load_campaign(path, arm="a2", seed=0)
    assert arm_local.stages[0].args == ("cap=999",)
    overridden = load_campaign(path, arm="a2", seed=0, overrides=["vars.frame_cap=555"])
    assert overridden.stages[0].args == ("cap=555",)


def test_campaign_unknown_arm_raises(tmp_path: Path) -> None:
    with pytest.raises(PipelineError, match="unknown arm"):
        load_campaign(_campaign_file(tmp_path), arm="nope", seed=0)


def test_resolved_env_merge_order_and_determinism(tmp_path: Path) -> None:
    profile = _load_example_profile()
    profile.env["CLUSTER_WANDB_TAGS"] = "from-profile"
    jobset = load_campaign(_campaign_file(tmp_path), arm="a1", seed=0)
    env = resolved_env(profile, jobset, jobset.stages[0])
    # campaign shared_env beats profile.env; stage executable always wins.
    assert env["CLUSTER_WANDB_TAGS"] == "demo,a1"
    assert (
        env["CLUSTER_PYTHON_EXECUTABLE"] == "scripts/tracking/train_hl_skill_diffsr.py"
    )
    assert env["STAGE_ONLY"] == "pre"
    assert env["CLUSTER_EXTRA_BIND_PATHS"] == ":".join(profile.extra_bind_paths)
    again = resolved_env(profile, jobset, jobset.stages[0])
    assert list(env.items()) == list(again.items())


def test_render_job_env_round_trips_through_bash(tmp_path: Path) -> None:
    rendered = render_job_env(
        {"CLUSTER_A": "plain", "CLUSTER_B": "has spaces 'and quotes'"},
        plan_sha="abc123",
        stage="pretrain",
    )
    env_file = tmp_path / "job_env.resolved.sh"
    env_file.write_text(rendered)
    proc = subprocess.run(
        ["bash", "-c", f'source "{env_file}" && printf %s "$CLUSTER_B"'],
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.stdout == "has spaces 'and quotes'"
    assert "plan_sha=abc123" in rendered


def test_render_job_env_rejects_bad_key() -> None:
    with pytest.raises(PipelineError, match="invalid environment variable name"):
        render_job_env({"BAD-KEY": "x"}, plan_sha="s", stage="pretrain")
