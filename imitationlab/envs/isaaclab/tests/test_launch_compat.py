"""Kit-less tests for the launcher helpers of the Isaac Lab develop API.

``isaaclab.app.add_launcher_args`` imports ``isaacsim`` and so bootstraps the
Kit kernel. The in-process tests below replace it with a small stand-in so this
test process stays free of Kit. One subprocess test runs the real function
under the strict kit-less import guard of ``imitationlab.runners.runtime_bootstrap``.

These tests cover ``imitationlab.runners.launch_compat``. They live here, not in
``experiments/tests/``, because they import Isaac Lab and so
need the ``isaaclab`` Pixi environment (``pixi run -e isaaclab test-isaaclab``).
The default environment, which runs ``pixi run test-experiments``, has no
Isaac Lab.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import gymnasium as gym
import isaaclab.app
import pytest
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.visualizers.visualizer_cfg import parse_visualizer_csv
from imitationlab.runners import launch_compat
from isaaclab_newton.physics import NewtonCfg
from isaaclab_physx.physics import PhysxCfg

TASK_ID = "LaunchCompatProbe-v0"
REPO_ROOT = Path(__file__).resolve().parents[4]


def _stand_in_launcher_args(parser: argparse.ArgumentParser) -> None:
    """Define the launcher arguments these tests read, without importing Kit."""
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--visualizer", "--viz", type=parse_visualizer_csv, default=None
    )


@pytest.fixture(autouse=True)
def kit_free_launcher_args(monkeypatch):
    monkeypatch.setattr(isaaclab.app, "add_launcher_args", _stand_in_launcher_args)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    launch_compat.add_launcher_args(parser)
    return parser


def test_legacy_launcher_flags_are_accepted_and_hidden():
    parser = _parser()
    args, rest = parser.parse_known_args(
        ["--headless", "--enable_cameras", "--device", "cpu", "env.scene.num_envs=16"]
    )
    assert args.headless is True
    assert args.enable_cameras is True
    assert args.device == "cpu"
    assert rest == ["env.scene.num_envs=16"]

    defaults, _ = parser.parse_known_args([])
    assert defaults.headless is True
    assert defaults.enable_cameras is False

    help_text = parser.format_help()
    assert "--headless" not in help_text
    assert "--enable_cameras" not in help_text
    assert "--visualizer" in help_text


def test_script_flag_defined_before_the_call_wins():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", default=False)
    launch_compat.add_launcher_args(parser)
    args, _ = parser.parse_known_args([])
    assert args.headless is False
    assert args.enable_cameras is False


def test_preset_cli_keeps_hydra_tokens_next_to_legacy_flags():
    from isaaclab_tasks.utils import setup_preset_cli

    args, remaining = setup_preset_cli(
        _parser(),
        [
            "--headless",
            "--visualizer",
            "newton_gl",
            "physics=newton_mjwarp",
            "env.scene.num_envs=16",
            "agent.logger.backend=",
        ],
    )
    assert args.visualizer == ["newton_gl"]
    assert remaining == [
        "physics=newton_mjwarp",
        "env.scene.num_envs=16",
        "agent.logger.backend=",
    ]


@configclass
class _SceneCfg:
    num_envs: int = 4


@configclass
class _EnvCfg:
    scene: _SceneCfg = _SceneCfg()
    sim: SimulationCfg = SimulationCfg(physics=NewtonCfg())
    decimation: int = 2


@configclass
class _AgentCfg:
    seed: int = 0


class _Probe:
    """Register a tiny task so the real ``resolve_task_config`` runs."""

    def __enter__(self):
        gym.register(
            id=TASK_ID,
            entry_point="gymnasium.envs.classic_control:CartPoleEnv",
            kwargs={"env_cfg_entry_point": _EnvCfg, "probe_cfg_entry_point": _AgentCfg},
        )
        return self

    def __exit__(self, *exc):
        gym.registry.pop(TASK_ID, None)


@pytest.fixture
def probe_task():
    with _Probe():
        yield


@pytest.fixture
def restored_logging():
    """Hydra replaces the root handlers; put the original ones back."""
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    for handler in list(root.handlers):
        if handler not in handlers:
            handler.close()
    root.handlers[:] = handlers
    root.setLevel(level)


def test_hydra_task_config_passes_configs_and_overrides(monkeypatch, probe_task):
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "env.scene.num_envs=16", "agent.seed=7", "env.decimation=3"],
    )

    @launch_compat.hydra_task_config(TASK_ID, "probe_cfg_entry_point")
    def main(env_cfg, agent_cfg, label, *, scale=1):
        """Docstring is kept."""
        return env_cfg, agent_cfg, label, scale

    assert main.__name__ == "main"
    assert main.__doc__ == "Docstring is kept."
    env_cfg, agent_cfg, label, scale = main("tag", scale=2)
    assert isinstance(env_cfg, _EnvCfg)
    assert isinstance(agent_cfg, _AgentCfg)
    assert env_cfg.scene.num_envs == 16
    assert env_cfg.decimation == 3
    assert agent_cfg.seed == 7
    assert (label, scale) == ("tag", 2)
    # The caller owns sys.argv; the decorator leaves it as it found it.
    assert sys.argv[1] == "env.scene.num_envs=16"


def test_hydra_task_config_reads_argv_when_called_not_when_decorated(
    monkeypatch, probe_task
):
    monkeypatch.setattr(sys, "argv", ["prog"])

    @launch_compat.hydra_task_config(TASK_ID, "probe_cfg_entry_point")
    def main(env_cfg, agent_cfg):
        return env_cfg.scene.num_envs

    assert main() == 4
    monkeypatch.setattr(sys, "argv", ["prog", "env.scene.num_envs=9"])
    assert main() == 9


def test_hydra_task_config_applies_typed_physics_selector(monkeypatch, probe_task):
    # The probe env has no `physics` preset, so a typed selector must fail
    # loudly instead of being dropped.
    monkeypatch.setattr(sys, "argv", ["prog", "physics=newton_mjwarp"])

    @launch_compat.hydra_task_config(TASK_ID, "probe_cfg_entry_point")
    def main(env_cfg, agent_cfg):
        raise AssertionError("must not run")

    with pytest.raises(ValueError, match="newton_mjwarp"):
        main()


def test_hydra_task_config_runs_hydra_for_other_overrides(
    monkeypatch, tmp_path, probe_task, restored_logging
):
    run_dir = tmp_path / "hydra_run"
    monkeypatch.setattr(
        sys, "argv", ["prog", f"hydra.run.dir={run_dir}", "env.scene.num_envs=5"]
    )
    monkeypatch.chdir(tmp_path)

    @launch_compat.hydra_task_config(TASK_ID, "probe_cfg_entry_point")
    def main(env_cfg, agent_cfg):
        return env_cfg.scene.num_envs

    assert main() == 5
    assert (run_dir / ".hydra").is_dir()


def _launch_args(*argv: str) -> argparse.Namespace:
    args, _ = _parser().parse_known_args(list(argv))
    return args


def test_needs_kit_follows_physics_and_visualizer():
    newton_cfg = _EnvCfg()
    assert launch_compat.needs_kit(newton_cfg, _launch_args()) is False

    physx_cfg = _EnvCfg(sim=SimulationCfg(physics=PhysxCfg()))
    assert launch_compat.needs_kit(physx_cfg, _launch_args()) is True

    kit_args = _launch_args("--visualizer", "kit")
    assert launch_compat.needs_kit(_EnvCfg(), kit_args) is True
    # scan() rewrites the visualizer selection in place and is idempotent.
    assert kit_args.visualizer == ["kit"]
    assert launch_compat.needs_kit(_EnvCfg(), kit_args) is True


def test_needs_kit_accepts_a_dict_and_none():
    assert launch_compat.needs_kit(_EnvCfg(), {"visualizer": ["newton_gl"]}) is False
    assert launch_compat.needs_kit(_EnvCfg(), None) is False


def test_needs_kit_applies_the_physics_selector_in_place():
    cfg = _EnvCfg(sim=SimulationCfg(physics=PhysxCfg()))
    assert launch_compat.needs_kit(cfg, {"physics": "newton_mjwarp"}) is False
    assert isinstance(cfg.sim.physics, NewtonCfg)


def test_needs_kit_counts_an_explicit_experience_and_require_kit():
    """Both are Kit sources of ``launch_simulation`` that the config scan misses."""
    assert launch_compat.needs_kit(_EnvCfg(), {"experience": "my.python.kit"}) is True
    assert launch_compat.needs_kit(_EnvCfg(), {"require_kit": True}) is True
    # The `AppLauncher` adapter sets require_kit itself. Unset or false values
    # are not a Kit source.
    unset = {"experience": "", "require_kit": False}
    assert launch_compat.needs_kit(_EnvCfg(), unset) is False


def _launcher_kit_decision(cfg, args: dict) -> bool:
    """The Kit decision of ``launch_simulation``, read from its own private helper."""
    from isaaclab.app import scan, sim_launcher

    helper = getattr(sim_launcher, "_get_kit_runtime_sources", None)
    assert helper is not None, (
        "isaaclab.app.sim_launcher._get_kit_runtime_sources moved. Read how "
        "launch_simulation decides to start Kit and update launch_compat.needs_kit."
    )
    return bool(helper(scan(cfg, args), args))


@pytest.mark.parametrize(
    "args",
    [
        {},
        {"visualizer": ["kit"]},
        {"visualizer": ["newton_gl"]},
        {"experience": "my.python.kit"},
        {"require_kit": True},
        {"livestream": 1},
        {"physics": "newton_mjwarp"},
    ],
    ids=lambda args: ",".join(f"{key}={value}" for key, value in args.items())
    or "none",
)
@pytest.mark.parametrize("physics_cfg", [NewtonCfg, PhysxCfg])
def test_needs_kit_matches_the_launcher_decision(args, physics_cfg):
    """Run this test after every ``pixi update isaaclab``: it catches drift."""

    def cfg():
        return _EnvCfg(sim=SimulationCfg(physics=physics_cfg()))

    expected = _launcher_kit_decision(cfg(), dict(args))
    assert launch_compat.needs_kit(cfg(), dict(args)) is expected


def _video_cfg(num_envs: int) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        scene=types.SimpleNamespace(num_envs=num_envs),
        sim=SimulationCfg(physics=NewtonCfg()),
        video_recorders=[],
    )


def test_attach_video_recorder_builds_recorder_and_headless_visualizer(tmp_path):
    cfg = _video_cfg(num_envs=8)
    out_dir = launch_compat.attach_video_recorder(
        cfg, output_dir=tmp_path / "videos", video_length=120, video_interval=0
    )
    assert out_dir == (tmp_path / "videos").resolve()
    assert out_dir.is_absolute() and out_dir.is_dir()

    (recorder,) = cfg.video_recorders
    assert recorder.source == "viz:newton_gl"
    assert recorder.output_dir == str(out_dir)
    assert (recorder.video_length, recorder.video_interval) == (120, 0)

    (visualizer,) = cfg.sim.visualizer_cfgs
    assert visualizer.visualizer_type == "newton_gl"
    assert visualizer.headless is True
    assert visualizer.visible_env_indices == list(range(8))
    assert (visualizer.window_width, visualizer.window_height) == (1280, 720)


def test_attach_video_recorder_takes_camera_overrides_and_keeps_other_visualizers(
    tmp_path,
):
    from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

    cfg = _video_cfg(num_envs=4096)
    old_gl = NewtonGLVisualizerCfg(eye=(1.0, 1.0, 1.0))
    other = types.SimpleNamespace(visualizer_type="rerun")
    cfg.sim.visualizer_cfgs = [old_gl, other]

    launch_compat.attach_video_recorder(
        cfg,
        output_dir=tmp_path,
        video_length=50,
        video_interval=1000,
        eye=(3.0, 3.0, 2.0),
        lookat=(0.0, 0.0, 1.0),
        window_width=640,
        window_height=480,
        headless=False,
    )
    assert cfg.sim.visualizer_cfgs[0] is other
    visualizer = cfg.sim.visualizer_cfgs[1]
    assert visualizer is not old_gl and len(cfg.sim.visualizer_cfgs) == 2
    assert visualizer.eye == (3.0, 3.0, 2.0)
    assert visualizer.lookat == (0.0, 0.0, 1.0)
    assert (visualizer.window_width, visualizer.window_height) == (640, 480)
    assert visualizer.headless is True
    visible = visualizer.visible_env_indices
    assert visible is not None and len(visible) == 16
    assert all(0 <= index < 4096 for index in visible)
    assert cfg.video_recorders[0].video_interval == 1000


def test_attach_video_recorder_retires_clips_of_earlier_runs(tmp_path):
    """A reused directory must hold only the clips of the new run."""
    (tmp_path / "clip_0000.mp4").write_bytes(b"old")
    (tmp_path / "notes.txt").write_text("keep")

    launch_compat.attach_video_recorder(
        _video_cfg(4), output_dir=tmp_path, video_length=10, video_interval=0
    )

    # No ``*.mp4`` pattern finds an old clip, not even a recursive search.
    assert list(tmp_path.rglob("*.mp4")) == []
    assert (tmp_path / "notes.txt").read_text() == "keep"
    retired = sorted(path.name for path in tmp_path.glob("*.superseded-*"))
    assert [name.split(".superseded-")[0] for name in retired] == ["clip_0000.mp4"]
    (old_clip,) = tmp_path.glob("clip_0000.mp4.superseded-*")
    assert old_clip.read_bytes() == b"old"


def test_attach_video_recorder_leaves_a_clean_directory_alone(tmp_path):
    launch_compat.attach_video_recorder(
        _video_cfg(4), output_dir=tmp_path, video_length=10, video_interval=0
    )
    assert list(tmp_path.iterdir()) == []


def test_attach_video_recorder_gives_each_prefix_its_own_clip_names(tmp_path):
    """Labels that share one directory keep each other's clips."""
    launch_compat.attach_video_recorder(
        _video_cfg(4),
        output_dir=tmp_path,
        video_length=10,
        video_interval=0,
        output_filename_prefix="kneel_ours",
    )
    (tmp_path / "kneel_ours_0000.mp4").write_bytes(b"ours, first run")

    cfg = _video_cfg(4)
    launch_compat.attach_video_recorder(
        cfg,
        output_dir=tmp_path,
        video_length=10,
        video_interval=0,
        output_filename_prefix="kneel_sonic",
    )
    (tmp_path / "kneel_sonic_0000.mp4").write_bytes(b"sonic")
    assert cfg.video_recorders[0].output_filename_prefix == "kneel_sonic"
    # The second label did not touch the first label's clip.
    assert (tmp_path / "kneel_ours_0000.mp4").read_bytes() == b"ours, first run"

    # A longer label that starts with another label is a different label.
    launch_compat.attach_video_recorder(
        _video_cfg(4),
        output_dir=tmp_path,
        video_length=10,
        video_interval=0,
        output_filename_prefix="kneel_ours_full",
    )
    (tmp_path / "kneel_ours_full_0000.mp4").write_bytes(b"full")
    launch_compat.attach_video_recorder(
        _video_cfg(4),
        output_dir=tmp_path,
        video_length=10,
        video_interval=0,
        output_filename_prefix="kneel_ours",
    )
    # A rerun of the first label retires only its own clip.
    assert not (tmp_path / "kneel_ours_0000.mp4").exists()
    assert list(tmp_path.glob("kneel_ours_0000.mp4.superseded-*"))
    assert (tmp_path / "kneel_ours_full_0000.mp4").read_bytes() == b"full"
    assert (tmp_path / "kneel_sonic_0000.mp4").read_bytes() == b"sonic"


def test_attach_video_recorder_retires_only_the_clips_of_its_label(tmp_path):
    (tmp_path / "kneel_ours_0000.mp4").write_bytes(b"earlier run")
    (tmp_path / "kneel_sonic_0000.mp4").write_bytes(b"other label")

    launch_compat.attach_video_recorder(
        _video_cfg(4),
        output_dir=tmp_path,
        video_length=10,
        video_interval=0,
        output_filename_prefix="kneel_ours",
    )

    assert not (tmp_path / "kneel_ours_0000.mp4").exists()
    assert (tmp_path / "kneel_sonic_0000.mp4").exists()


@pytest.mark.parametrize("prefix", ["", "../escape", "a/b", ".hidden", "with space"])
def test_attach_video_recorder_rejects_a_prefix_that_is_not_a_name_part(
    tmp_path, prefix
):
    with pytest.raises(ValueError, match="output_filename_prefix"):
        launch_compat.attach_video_recorder(
            _video_cfg(4),
            output_dir=tmp_path,
            video_length=10,
            video_interval=0,
            output_filename_prefix=prefix,
        )


def test_attach_video_recorder_rejects_bad_clip_timing(tmp_path):
    with pytest.raises(ValueError, match="video_length"):
        launch_compat.attach_video_recorder(
            _video_cfg(4), output_dir=tmp_path, video_length=0, video_interval=0
        )


def test_attached_recorder_survives_the_launcher_scan(tmp_path):
    """The launcher must find the recorder and keep a headless GL visualizer."""
    from isaaclab.visualizers.visualizer_cfg import resolve_visualizer_cfgs

    cfg = _video_cfg(num_envs=4)
    launch_compat.attach_video_recorder(
        cfg, output_dir=tmp_path, video_length=10, video_interval=0, eye=(2.0, 2.0, 2.0)
    )
    args = _launch_args()
    assert launch_compat.needs_kit(cfg, args) is False
    assert cfg.video_recorders[0].source == "viz:newton_gl"

    from isaaclab.app import scan

    video_visualizers = scan(cfg, args).video_visualizers
    assert video_visualizers == ["newton_gl"]
    (resolved,) = resolve_visualizer_cfgs(
        cfg.sim.visualizer_cfgs, args.visualizer, None, video_visualizers
    )
    assert getattr(resolved, "headless") is True
    assert resolved.eye == (2.0, 2.0, 2.0)


def test_recorder_attached_inside_launch_simulation_keeps_a_headless_gl_visualizer(
    tmp_path,
):
    """``train_impl.py`` attaches the recorder inside the ``launch_simulation`` block.

    The recorder source is ``viz:newton_gl``, which needs no Kit. The launch has
    then already resolved the visualizers, and the attached ones must still be
    what the env builds: one headless GL visualizer and one recorder.
    """
    from isaaclab.app import launch_simulation

    cfg = _video_cfg(num_envs=4)
    args = _launch_args()
    with launch_simulation(cfg, args):
        launch_compat.attach_video_recorder(
            cfg, output_dir=tmp_path, video_length=10, video_interval=0
        )

    (recorder,) = cfg.video_recorders
    assert recorder.source == "viz:newton_gl"
    (visualizer,) = cfg.sim.visualizer_cfgs
    assert visualizer.visualizer_type == "newton_gl"
    assert visualizer.headless is True
    # Another scan after the launch still needs no Kit.
    assert launch_compat.needs_kit(cfg, args) is False


def _run_python(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent,
        timeout=300,
        check=False,
    )


def test_importing_the_module_does_not_load_kit_or_usd():
    result = _run_python(
        """
        import sys
        from imitationlab.runners import launch_compat

        loaded = [m for m in sys.modules if m.split(".")[0] in ("isaacsim", "carb")]
        loaded += [m for m in sys.modules if m.startswith("omni.kit")]
        assert not loaded, loaded
        # A pxr import before Kit starts breaks the Kit USD extensions. The
        # Isaac Lab task registration imports pxr, so it must stay out.
        early = [
            name for name in ("pxr", "imitationlab.envs.isaaclab.tasks")
            if name in sys.modules
        ]
        assert not early, early
        print("import ok")
        """
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "import ok" in result.stdout


def test_real_launcher_args_run_under_the_strict_kitless_guard():
    result = _run_python(
        """
        import argparse
        from types import SimpleNamespace

        from imitationlab.runners.runtime_bootstrap import (
            assert_kit_not_loaded,
            install_kit_import_guard,
        )

        install_kit_import_guard()
        from isaaclab.sim import SimulationCfg
        from imitationlab.runners import launch_compat
        from isaaclab_newton.physics import NewtonCfg
        from isaaclab_tasks.utils import setup_preset_cli

        parser = argparse.ArgumentParser()
        launch_compat.add_launcher_args(parser)
        args, remaining = setup_preset_cli(
            parser,
            ["--headless", "--enable_cameras", "--viz", "newton_gl",
             "physics=newton_mjwarp", "env.scene.num_envs=16"],
        )
        assert args.headless is True and args.enable_cameras is True
        assert args.visualizer == ["newton_gl"]
        assert remaining == ["physics=newton_mjwarp", "env.scene.num_envs=16"]

        cfg = SimpleNamespace(sim=SimulationCfg(physics=NewtonCfg()))
        assert launch_compat.needs_kit(cfg, args) is False
        assert_kit_not_loaded()
        print("guard ok")
        """
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "guard ok" in result.stdout
