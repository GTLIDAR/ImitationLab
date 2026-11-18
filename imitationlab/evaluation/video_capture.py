"""Video capture for evaluations on the Isaac Lab ``develop`` API.

``gym.wrappers.RecordVideo`` does not record any more. The env writes video
itself from ``env_cfg.video_recorders``, inside its ``step`` call. A clip file
exists only when the env stepped its recorder. This module attaches the
recorder to the env config and checks, after the run, that a clip exists.

The entrypoint rule in ``scripts/CONTEXT.md`` requires that an evaluation which
produces a video prints the absolute retained path to stdout.
:meth:`VideoCapture.report` does that, and it fails when no new clip exists. A
silent run without video is a defect.

Isaac Lab imports stay lazy. This module also loads in the default Pixi
environment, where Isaac Lab is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import time
from typing import Any

__all__ = [
    "CLIP_NAME",
    "FOLLOW_EYE_OFFSET",
    "FOLLOW_LOOKAT_OFFSET",
    "LEGACY_CLIP_NAME",
    "VideoCapture",
    "aim_camera_at_robot",
    "attach_video_capture",
    "follow_robot_camera",
    "clip_for_stem",
    "clip_prefix_for_label",
    "retained_clip",
]

VIDEO_LOG_PREFIX = "[VIDEO]"
"""Prefix of each stdout line that names a retained clip."""

DEFAULT_CLIP_PREFIX = "clip"
"""File name prefix of a clip when the caller gives none."""

CLIP_NAME = f"{DEFAULT_CLIP_PREFIX}_0000.mp4"
"""First clip file the Isaac Lab ``develop`` recorder writes into a directory."""

LEGACY_CLIP_NAME = "rl-video-step-0.mp4"
"""Clip file that ``gym.wrappers.RecordVideo`` wrote before the ``develop`` move."""

FOLLOW_EYE_OFFSET = (2.6, 2.6, 1.6)
"""Camera position of a chase view, relative to the robot root, in world axes [m]."""

FOLLOW_LOOKAT_OFFSET = (0.0, 0.0, 0.7)
"""Camera target of a chase view, relative to the robot root, in world axes [m]."""

_CLOCK_SLACK_SECONDS = 5.0
"""Allowed gap between this host's clock and a network file system's timestamps.

A run takes much longer than this, so a clip from an earlier run is still excluded.
"""


@dataclass(frozen=True)
class VideoCapture:
    """A video recorder attached to an env config, and where its clips go."""

    directory: Path
    """Absolute output directory of the clips."""

    started_at: float
    """Wall-clock time (``time.time()``) when the recorder was attached."""

    prefix: str = DEFAULT_CLIP_PREFIX
    """File name prefix of the clips of this run: ``<prefix>_0000.mp4``, and so on."""

    def clips(self) -> list[Path]:
        """Return the clips of this run written since the recorder was attached, in name order.

        Only files named ``<prefix>_<index>.mp4`` count. Runs with other prefixes
        may share the directory, and their clips are not this run's clips.
        """
        own_clip = re.compile(rf"{re.escape(self.prefix)}_\d{{4,}}\.mp4")
        return sorted(
            clip
            for clip in self.directory.glob("*.mp4")
            if own_clip.fullmatch(clip.name)
            and clip.stat().st_mtime >= self.started_at - _CLOCK_SLACK_SECONDS
        )

    def report(self, *, require_clip: bool = True) -> list[Path]:
        """Print the absolute path of each retained clip and return the paths.

        Call it after ``env.close()``. ``env.close()`` flushes the last clip.

        Args:
            require_clip: If true, fail when no clip exists. A training script
                passes false: it has already saved its checkpoints, and a
                missing clip must not turn a finished run into a failed job. It
                then prints a warning instead.

        Raises:
            RuntimeError: If ``require_clip`` is true and no clip was written
                since the recorder was attached. The usual cause is an env whose
                ``step`` does not call ``recorder.step()`` for each of its
                ``video_recorders``.
        """
        clips = self.clips()
        if not clips:
            message = (
                f"Video was requested, but no .mp4 clip was written to {self.directory}. "
                "Check that the env steps each recorder in env.video_recorders."
            )
            if require_clip:
                raise RuntimeError(message)
            print(f"[WARN] {message}", flush=True)
            return []
        for clip in clips:
            print(f"{VIDEO_LOG_PREFIX} {clip}", flush=True)
        return clips


def attach_video_capture(
    env_cfg: Any,
    *,
    output_dir: str | Path,
    video_length: int,
    video_interval: int = 0,
    output_filename_prefix: str = DEFAULT_CLIP_PREFIX,
    **camera_kwargs: Any,
) -> VideoCapture:
    """Attach a recorder for clips of ``video_length`` env steps to ``env_cfg``.

    The first clip starts at the first env step. With ``video_interval`` of 0 it
    is the only clip. Call this before ``isaaclab.app.launch_simulation`` or
    ``gym.make`` and after ``env_cfg.scene.num_envs`` is set. ``output_dir`` is
    created.

    Args:
        env_cfg: Environment config. It changes in place.
        output_dir: Directory for the clip files.
        video_length: Number of env steps in each clip. Must be positive.
        video_interval: Env steps between clip starts, or 0 for one clip.
        output_filename_prefix: File name prefix of the clips. Give each label
            its own prefix when several runs write into one ``output_dir``
            (see :func:`clip_prefix_for_label`). The recorder renames only the
            earlier clips that have this prefix.
        **camera_kwargs: Camera fields for the visualizer, such as ``eye``,
            ``lookat``, ``window_width``, and ``window_height``. See
            :func:`~imitationlab.runners.launch_compat.attach_video_recorder`.

    Returns:
        The capture handle. Call :meth:`VideoCapture.report` after ``env.close()``.
    """
    if int(video_length) <= 0:
        raise ValueError(f"video_length must be positive, got {video_length}.")
    from imitationlab.runners.launch_compat import attach_video_recorder

    started_at = time.time()
    directory = attach_video_recorder(
        env_cfg,
        output_dir=output_dir,
        video_length=int(video_length),
        video_interval=int(video_interval),
        output_filename_prefix=output_filename_prefix,
        **camera_kwargs,
    )
    return VideoCapture(
        directory=Path(directory),
        started_at=started_at,
        prefix=output_filename_prefix,
    )


def clip_prefix_for_label(label: str | None, default: str = "eval") -> str:
    """Return a clip file name prefix for a run label.

    Replaces each character outside letters, digits, ``.``, ``_`` and ``-`` with
    ``_``, so any label gives a valid prefix. An empty or missing label gives
    ``default``. The prefix keeps runs that share one video directory apart:
    ``<prefix>_0000.mp4``.
    """
    text = re.sub(r"[^A-Za-z0-9._-]", "_", str(label or "")).lstrip("._-")
    return text or default


def aim_camera_at_robot(
    env: Any,
    env_index: int,
    *,
    asset_name: str = "robot",
    eye_offset: tuple[float, float, float] = FOLLOW_EYE_OFFSET,
    lookat_offset: tuple[float, float, float] = FOLLOW_LOOKAT_OFFSET,
) -> None:
    """Aim the recording camera at the root of one robot.

    The ``develop`` recorder has no per-frame camera hook, so a chase view needs
    one call before each ``env.step``. The camera then lags the robot by one
    control step. The call changes the camera of every visualizer through
    ``env.sim.set_camera_view``. It stores the pose when the visualizer does not
    exist yet, so it is also valid before the first step.

    The call reads one robot position to the host, so use it for a video run
    with few environments, not for a throughput run.

    Args:
        env: Unwrapped Isaac Lab env with ``scene`` and ``sim``.
        env_index: Index of the environment whose robot the camera follows.
        asset_name: Scene name of the robot articulation.
        eye_offset: Camera position relative to the robot root, in world axes [m].
        lookat_offset: Camera target relative to the robot root, in world axes [m].
    """
    root_pos = env.scene[asset_name].data.root_pos_w
    root = getattr(root_pos, "torch", root_pos)[env_index].tolist()
    env.sim.set_camera_view(
        tuple(position + offset for position, offset in zip(root, eye_offset)),
        tuple(position + offset for position, offset in zip(root, lookat_offset)),
    )


def follow_robot_camera(
    env: Any,
    env_index: int,
    *,
    eye_offset: tuple[float, float, float] = FOLLOW_EYE_OFFSET,
    lookat_offset: tuple[float, float, float] = FOLLOW_LOOKAT_OFFSET,
) -> Any:
    """Wrap a gym env so the recording camera follows one robot.

    The wrapper calls :func:`aim_camera_at_robot` before each ``step``. It must
    hook ``step``, not ``render``: the ``develop`` recorder captures frames
    inside the env step and never calls ``render``, so a wrapper that overrides
    ``render`` (the old ``RecordVideo`` design) silently does nothing.

    Args:
        env: Gym env whose ``unwrapped`` is an Isaac Lab env.
        env_index: Index of the environment whose robot the camera follows.
        eye_offset: Camera position relative to the robot root, in world axes [m].
        lookat_offset: Camera target relative to the robot root, in world axes [m].
    """
    import gymnasium as gym

    class _FollowRobotCamera(gym.Wrapper):
        def step(self, action: Any) -> Any:
            aim_camera_at_robot(
                self.env.unwrapped,
                env_index,
                eye_offset=eye_offset,
                lookat_offset=lookat_offset,
            )
            return self.env.step(action)

    return _FollowRobotCamera(env)


def retained_clip(directory: str | Path) -> Path | None:
    """Return the first clip in a run's video directory, or ``None`` if it has none.

    A reader that needs one clip of a finished run, such as a post-run check or
    an annotation file, uses this instead of a fixed file name. It prefers the
    first ``clip_*.mp4`` of the ``develop`` recorder. It falls back to the
    ``rl-video-step-0.mp4`` that older renders hold.

    :func:`~imitationlab.runners.launch_compat.attach_video_recorder` renames the
    clips of earlier runs before a run starts, so the first clip belongs to the
    latest run.
    """
    directory = Path(directory)
    clips = sorted(directory.glob("clip_*.mp4"))
    if clips:
        return clips[0]
    legacy = directory / LEGACY_CLIP_NAME
    return legacy if legacy.is_file() else None


def clip_for_stem(
    directory: str | Path, stem: str, *, required: bool = False
) -> Path | None:
    """Return the newest rendered clip of ``stem`` in ``directory``.

    A report builder reads a clip by the name of the thing it shows. Two file
    names exist for one stem. Renders made with the old ``RecordVideo`` wrapper
    are ``<stem>.mp4``. The ``develop`` recorder writes ``<stem>_0000.mp4``
    (``<stem>_0001.mp4`` for later clips of the same recorder). Old renders stay
    on disk under the old name, so a reader must accept both. When several files
    match, the newest by modification time wins; a tie goes to the higher name.
    ``<name>.mp4.superseded-<time>`` files never match.

    Args:
        directory: Directory that holds the rendered clips.
        stem: Clip stem, without extension and without the ``_0000`` index.
        required: If true, raise when no clip matches instead of returning ``None``.

    Raises:
        FileNotFoundError: If ``required`` is true and no clip matches.
    """
    directory = Path(directory)
    wanted = re.compile(rf"{re.escape(stem)}(_\d{{4,}})?\.mp4")
    matches = [
        path
        for path in directory.glob("*.mp4")
        if path.is_file() and wanted.fullmatch(path.name)
    ]
    if matches:
        return max(matches, key=lambda path: (path.stat().st_mtime, path.name))
    if required:
        raise FileNotFoundError(
            f"No rendered clip for stem {stem!r} in {directory}. Looked for "
            f"{stem}.mp4 and {stem}_<index>.mp4."
        )
    return None
