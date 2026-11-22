"""Launcher helpers for the Isaac Lab ``develop`` API.

``develop`` removed or moved four launcher tools that scripts in this repo use:

- ``isaaclab_tasks.utils.add_launcher_args`` is now ``isaaclab.app.add_launcher_args``.
  The new parser has no ``--headless`` and no ``--enable-cameras`` flag.
  ``--headless`` is gone for good. ``enable_cameras`` is still a launcher key
  that Kit reads (see :func:`add_launcher_args`).
- ``isaaclab_tasks.utils.compute_kit_requirements`` is gone. Use ``scan``.
- ``isaaclab_tasks.utils.hydra.hydra_task_config`` is gone. Use
  ``isaaclab_tasks.utils.resolve_task_config``.
- ``gym.wrappers.RecordVideo`` does not record any more. The env records video
  itself from ``env_cfg.video_recorders``.

This module keeps one repo-owned wrapper for each tool. A call site then
changes by one import, not by a rewrite. The module targets ``develop`` only.

This module lives in ``imitationlab.runners``, not in ``imitationlab.envs``.
Importing the task package ``imitationlab.envs.isaaclab.tasks`` registers every
task. That import loads ``pxr`` from the pip wheel. Kit must start before any
``pxr`` import, or Kit fails to load its own USD extensions. A script needs
``add_launcher_args`` before it starts Kit, so the helper must stay outside the
env package. Keep this module import-light.

Every Isaac Lab import in this module is lazy, and importing the module does
not start Omniverse Kit. ``add_launcher_args`` does import ``isaacsim`` through
``isaaclab.app``. That import loads the Kit kernel but does not start an app,
and the strict kit-less import guard (``imitationlab.runners.runtime_bootstrap``)
blocks that import, which Isaac Lab accepts. Only
``isaaclab.app.launch_simulation`` starts Kit, and only when the config needs
it.
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import math
import os
import re
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from imitationlab.cli_args import kebab_options

__all__ = [
    "add_launcher_args",
    "attach_video_recorder",
    "hydra_task_config",
    "needs_kit",
    "run_then_close",
]

_NEWTON_GL_SOURCE = "viz:newton_gl"
"""Video source for a headless Newton GL visualizer: no Kit and no scene camera."""

DEFAULT_CLIP_PREFIX = "clip"
"""Default clip file prefix of ``VideoRecorderCfg``: ``clip_0000.mp4``, ``clip_0001.mp4``."""

_CLIP_PREFIX_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
"""Allowed clip prefix: a plain file name part, so a prefix cannot leave the video directory."""


def add_launcher_args(parser: argparse.ArgumentParser) -> None:
    """Add the Isaac Lab launcher arguments, and keep the old launcher flags.

    Calls ``isaaclab.app.add_launcher_args``. Then registers ``--headless`` and
    ``--enable-cameras`` as hidden flags. ``develop`` removed both flags. About
    60 launcher lines (pixi tasks, campaign scripts, subprocess commands) still
    pass them, and an unknown flag would otherwise reach Hydra and stop the run.
    The two flags are not equal:

    - ``--headless`` is ignored. ``develop`` chooses headless mode from
      ``--visualizer`` and the ``HEADLESS`` environment variable.
    - ``--enable-cameras`` is still honored when Kit starts. The flag sets the
      ``enable_cameras`` launcher key, and the Kit launcher reads that key to
      choose the rendering experience and the offscreen render path. This costs
      startup time and GPU memory. It has no effect when Kit does not start (for
      example a Newton run without a Kit source). ``launch_simulation`` also sets
      the key by itself when the config has a Kit camera, a Kit streaming view,
      or a Kit video recorder.

    A script that already defines one of these flags keeps its own definition.
    Define the script flag before this call; a later definition conflicts.
    """
    from isaaclab.app import add_launcher_args as _add_launcher_args

    _add_launcher_args(parser)
    # Isaac Lab declares ``--kit_args`` in snake case.
    kebab_options(parser)
    for flag, default in (("--headless", True), ("--enable-cameras", False)):
        try:
            parser.add_argument(
                flag, action="store_true", default=default, help=argparse.SUPPRESS
            )
        except argparse.ArgumentError:
            pass  # the script already defines this flag


def needs_kit(env_cfg: Any, args: argparse.Namespace | dict | None) -> bool:
    """Return whether the launch of ``env_cfg`` with ``args`` starts Kit.

    This is the repo's stand-in for the removed ``compute_kit_requirements``.
    It copies the Kit decision of ``isaaclab.app.launch_simulation``. Kit starts
    for Isaac Sim PhysX physics, for a Kit renderer camera, for the Kit
    visualizer (chosen with ``--visualizer`` or recorded by a video recorder),
    for an explicit Kit ``experience``, and when the caller sets
    ``require_kit`` (``AppLauncher`` always does). Livestreaming adds the Kit
    visualizer, so ``scan`` already reports it. ``launch_simulation`` keeps its
    own list in a private helper, so ``test_needs_kit_matches_the_launcher_decision``
    compares the two. Run that test after every ``pixi update isaaclab``.

    Warning:
        ``isaaclab.app.scan`` changes its inputs in place. It applies a
        ``physics`` selector from ``args``, resolves automatic PhysX and RTX
        placeholders in ``env_cfg``, rewrites the visualizer source of each
        video recorder, and turns ``args.visualizer`` into a list of canonical
        names. A second call gives the same result.
    """
    from isaaclab.app import scan

    config_scan = scan(env_cfg, args)
    launcher_args = vars(args) if isinstance(args, argparse.Namespace) else args or {}
    visualizers = launcher_args.get("visualizer") or []
    return bool(
        config_scan.needs_kit
        or "kit" in visualizers
        or "kit" in config_scan.video_visualizers
        or launcher_args.get("experience")
        or launcher_args.get("require_kit")
    )


def hydra_task_config(task_name: str, agent_cfg_entry_point: str | None) -> Callable:
    """Decorate a function that receives the resolved task configs.

    Keeps the contract of the removed
    ``isaaclab_tasks.utils.hydra.hydra_task_config``: the decorated function is
    called as ``func(env_cfg, agent_cfg, *args, **kwargs)``. It is built on
    ``isaaclab_tasks.utils.resolve_task_config``.

    ``resolve_task_config`` reads the overrides from ``sys.argv[1:]`` when the
    wrapper runs, not when the decorator runs. A script must therefore remove
    its own flags first, as it did before: ``sys.argv = [sys.argv[0]] +
    hydra_args``. Then ``env.*`` and ``agent.*`` values, ``physics=NAME``,
    ``renderer=NAME``, and ``presets=NAME`` reach the configs. Any other
    override goes to Hydra.

    The old decorator ran ``func`` inside the Hydra job. This one runs it after
    Hydra returns. No script in this repo reads ``HydraConfig``, and an error
    in ``func`` now shows its normal traceback. The wrapper returns what
    ``func`` returns.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            from isaaclab_tasks.utils import resolve_task_config

            env_cfg, agent_cfg = resolve_task_config(task_name, agent_cfg_entry_point)
            return func(env_cfg, agent_cfg, *args, **kwargs)

        return wrapper

    return decorator


def _center_env_block(num_envs: int, *, size: int = 4) -> list[int]:
    """Return env indices for a ``size x size`` block near the grid center.

    A small block keeps the frame capture cheap at any scale. The full-scene
    viewer needs about 18 s per frame at 8192 environments.
    """
    if num_envs <= size * size:
        return list(range(num_envs))
    grid_cols = max(math.ceil(math.sqrt(num_envs)), 1)
    center = (grid_cols // 2) * grid_cols + grid_cols // 2
    block = [
        index
        for row in range(size)
        for col in range(size)
        if 0 <= (index := center + row * grid_cols + col) < num_envs
    ]
    return block or list(range(min(size * size, num_envs)))


def _retire_stale_clips(
    out_dir: Path, output_filename_prefix: str = DEFAULT_CLIP_PREFIX
) -> list[Path]:
    """Rename the clips of earlier runs that have this prefix, so no ``*.mp4`` pattern finds them.

    ``VideoRecorder`` never overwrites: it numbers a new clip ``max(existing) + 1``.
    A reused directory would therefore keep the old clips next to the new ones, and a reader that
    takes the first ``*.mp4`` (even one that searches subdirectories) could
    report a stale clip. This function appends ``.superseded-<time>`` to each old
    file name. The data stays on disk. Nothing is deleted.

    Only the clips of this prefix are retired (``<prefix>_*.mp4``). Several
    runs may therefore share one directory, each with its own prefix, and keep
    every clip. Files of other prefixes stay as they are.

    Returns:
        The new paths of the renamed files.
    """
    # Match the index exactly: prefix "ours" must not take "ours_full_0000.mp4".
    own_clip = re.compile(rf"{re.escape(output_filename_prefix)}_\d{{4,}}\.mp4")
    stale = sorted(
        path
        for path in out_dir.glob("*.mp4")
        if path.is_file() and own_clip.fullmatch(path.name)
    )
    stamp = time.strftime("%Y%m%d_%H%M%S")
    retired = []
    for clip in stale:
        target = clip.with_name(f"{clip.name}.superseded-{stamp}")
        clip.rename(target)
        retired.append(target)
    return retired


def attach_video_recorder(
    env_cfg: Any,
    *,
    output_dir: str | Path,
    video_length: int,
    video_interval: int,
    output_filename_prefix: str = DEFAULT_CLIP_PREFIX,
    **camera_kwargs: Any,
) -> Path:
    """Record video from a headless Newton GL visualizer; return the output directory.

    Replaces ``render_mode="rgb_array"`` with ``gym.wrappers.RecordVideo``. Call
    it on ``env_cfg`` before ``gym.make``. The env then records inside its
    ``step`` call. It writes ``<output_dir>/<output_filename_prefix>_0000.mp4``,
    ``<output_filename_prefix>_0001.mp4``, and so on. The default prefix is
    ``clip``. A new clip starts at the first step and then every
    ``video_interval`` steps. Use ``video_interval=0`` for one clip. Each clip
    holds ``video_length`` steps.

    ``output_dir`` may hold the clips of earlier runs. The function renames the
    ones that have the same prefix to ``<name>.mp4.superseded-<time>`` (see
    ``_retire_stale_clips``) and leaves the clips of other prefixes. Give each
    label its own prefix when several runs write into one directory. Without
    this, every run would write ``clip_0000.mp4`` and retire the clip of the
    run before it.

    Call order. The recorder source is ``viz:newton_gl``. It needs neither Kit
    nor a scene camera, so the call is valid before ``launch_simulation`` and
    inside its ``with`` block. ``train_impl.py`` calls it inside the block, and
    the AppLauncher-based evaluators call it after Kit has started. Both orders
    ran in the 2026-10-07 smokes. The launcher decides whether to start Kit
    from the config it scans, so a recorder whose source needs Kit (``viz:kit``
    or ``sensor:...`` set directly on ``env_cfg.video_recorders``, not by this
    function) must be on the config before ``launch_simulation``. In every case
    the call must come before ``gym.make``, and after ``env_cfg.scene.num_envs``
    is set.

    The function replaces ``env_cfg.video_recorders``
    and the ``newton_gl`` entry of ``env_cfg.sim.visualizer_cfgs``. Entries of
    other visualizer types stay. ``launch_simulation`` runs that visualizer
    headless, even when ``--visualizer`` selects it.

    The camera defaults to a static bird view: window 1280 x 720, ``eye`` (11, 11,
    7), ``lookat`` (0, 0, 0.8). It shows a 4 x 4 block of environments near the grid
    center, or all environments when there are 16 or fewer. Set
    ``env_cfg.scene.num_envs`` first. ``camera_kwargs`` go to
    ``NewtonGLVisualizerCfg`` and override these defaults, for example
    ``eye=...``, ``lookat=...``, ``window_width=...``, ``window_height=...``,
    or ``visible_env_indices=[...]``. ``headless`` is always ``True``.

    Note:
        The env must call ``recorder.step()`` for every entry of
        ``self.video_recorders`` in each step. The base ``ManagerBasedRLEnv.step``
        does. An env class that overrides ``step`` must do it itself.

    Args:
        env_cfg: Environment config to change in place.
        output_dir: Directory for the clips. The function creates it and renames
            the earlier clips that have the same prefix.
        video_length: Number of env steps in each clip.
        video_interval: Env steps between clip starts, or ``0`` for one clip.
        output_filename_prefix: File name prefix of the clips. A plain name part
            of letters, digits, ``.``, ``_`` and ``-``.
        **camera_kwargs: Extra ``NewtonGLVisualizerCfg`` fields.

    Returns:
        The absolute output directory. Print it, with the clip names, when the
        evaluation finishes: the retained video path must be visible on stdout.

    Raises:
        ImportError: If ``moviepy`` is missing. The pixi ``isaaclab``
            environment installs ``moviepy<2`` through the ``video`` extra of
            ``isaaclab``.
        ValueError: If a clip timing value is out of range, or the prefix is not
            a plain name part.
    """
    if importlib.util.find_spec("moviepy") is None:
        raise ImportError(
            "Video recording needs moviepy<2. Add the 'video' extra to the "
            "isaaclab dependency in pixi.toml and run `pixi install -e isaaclab`."
        )

    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
    from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

    if not _CLIP_PREFIX_PATTERN.fullmatch(output_filename_prefix):
        raise ValueError(
            "output_filename_prefix must be a plain file name part (letters, digits, "
            f"'.', '_', '-'), got {output_filename_prefix!r}."
        )
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    recorder_cfg = VideoRecorderCfg(
        source=_NEWTON_GL_SOURCE,
        output_dir=str(out_dir),
        video_length=int(video_length),
        video_interval=int(video_interval),
        output_filename_prefix=output_filename_prefix,
    )
    recorder_cfg.validate_config()
    _retire_stale_clips(out_dir, output_filename_prefix)

    num_envs = int(getattr(env_cfg.scene, "num_envs", 16) or 16)
    visualizer_kwargs: dict[str, Any] = {
        "window_width": 1280,
        "window_height": 720,
        "eye": (11.0, 11.0, 7.0),
        "lookat": (0.0, 0.0, 0.8),
        "visible_env_indices": _center_env_block(num_envs),
        # A headless capture needs no stream panel and no live plots.
        "streaming_view": False,
        "enable_live_plots": False,
        **camera_kwargs,
        "headless": True,
    }
    existing = env_cfg.sim.visualizer_cfgs
    if existing is None:
        existing = []
    elif not isinstance(existing, list):
        existing = [existing]
    env_cfg.sim.visualizer_cfgs = [
        cfg for cfg in existing if cfg.visualizer_type != "newton_gl"
    ] + [NewtonGLVisualizerCfg(**visualizer_kwargs)]
    env_cfg.video_recorders = [recorder_cfg]
    return out_dir


def run_then_close(main: Callable[[], object], close: Callable[[], object]) -> None:
    """Run an entrypoint's ``main``, then ``close`` (``simulation_app.close``).

    Kit's ``close()`` ends the process itself with exit code 0, so no code after it
    runs, and a crash would look like a pass to any caller that reads ``$?``. When
    ``main`` raises, this prints the traceback, flushes both streams and leaves with
    exit code 1 before ``close``.
    """
    try:
        main()
    except BaseException:  # noqa: BLE001 - every failure must reach the exit code
        traceback.print_exc(file=sys.stderr)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    close()
