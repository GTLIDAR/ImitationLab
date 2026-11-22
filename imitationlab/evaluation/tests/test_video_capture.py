"""Tests for the evaluation video capture contract."""

from __future__ import annotations

import os
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from imitationlab.evaluation.video_capture import (
    FOLLOW_EYE_OFFSET,
    FOLLOW_LOOKAT_OFFSET,
    VideoCapture,
    aim_camera_at_robot,
    attach_video_capture,
    clip_for_stem,
    clip_prefix_for_label,
    follow_robot_camera,
    retained_clip,
)


def _install_fake_launch_compat(
    monkeypatch: pytest.MonkeyPatch, calls: list[dict]
) -> None:
    """Stand in for ``attach_video_recorder``, which needs Isaac Lab."""

    def attach_video_recorder(
        env_cfg, *, output_dir, video_length, video_interval, output_filename_prefix
    ):
        calls.append(
            {
                "env_cfg": env_cfg,
                "output_dir": Path(output_dir),
                "video_length": video_length,
                "video_interval": video_interval,
                "output_filename_prefix": output_filename_prefix,
            }
        )
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return Path(output_dir).resolve()

    monkeypatch.setattr(
        "imitationlab.runners.launch_compat.attach_video_recorder",
        attach_video_recorder,
    )


def test_attach_records_one_clip_from_the_first_step(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict] = []
    _install_fake_launch_compat(monkeypatch, calls)
    env_cfg = object()

    capture = attach_video_capture(
        env_cfg, output_dir=tmp_path / "videos", video_length=250
    )

    assert len(calls) == 1
    assert calls[0]["env_cfg"] is env_cfg
    assert calls[0]["video_length"] == 250
    assert calls[0]["video_interval"] == 0
    assert capture.directory == (tmp_path / "videos").resolve()


def test_attach_passes_the_clip_interval_for_periodic_clips(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict] = []
    _install_fake_launch_compat(monkeypatch, calls)

    attach_video_capture(
        object(), output_dir=tmp_path, video_length=200, video_interval=2000
    )

    assert (calls[0]["video_length"], calls[0]["video_interval"]) == (200, 2000)


def test_attach_passes_the_clip_prefix_and_report_lists_only_that_prefix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Two labels in one directory: each report names only its own clips."""
    calls: list[dict] = []
    _install_fake_launch_compat(monkeypatch, calls)

    ours = attach_video_capture(
        object(),
        output_dir=tmp_path,
        video_length=10,
        output_filename_prefix="kneel_ours",
    )
    sonic = attach_video_capture(
        object(),
        output_dir=tmp_path,
        video_length=10,
        output_filename_prefix="kneel_sonic",
    )
    assert [call["output_filename_prefix"] for call in calls] == [
        "kneel_ours",
        "kneel_sonic",
    ]
    (tmp_path / "kneel_ours_0000.mp4").write_bytes(b"a")
    (tmp_path / "kneel_sonic_0000.mp4").write_bytes(b"b")

    assert [clip.name for clip in ours.report()] == ["kneel_ours_0000.mp4"]
    assert [clip.name for clip in sonic.report()] == ["kneel_sonic_0000.mp4"]
    lines = capsys.readouterr().out.splitlines()
    assert lines == [
        f"[VIDEO] {tmp_path.resolve() / 'kneel_ours_0000.mp4'}",
        f"[VIDEO] {tmp_path.resolve() / 'kneel_sonic_0000.mp4'}",
    ]


def test_a_prefix_does_not_claim_the_clips_of_a_longer_prefix(
    tmp_path: Path,
) -> None:
    """Prefix `ours` must not take the clip of the run `ours_full`."""
    (tmp_path / "ours_0000.mp4").write_bytes(b"a")
    (tmp_path / "ours_full_0000.mp4").write_bytes(b"b")
    capture = VideoCapture(
        directory=tmp_path.resolve(), started_at=time.time() - 1.0, prefix="ours"
    )

    assert [clip.name for clip in capture.clips()] == ["ours_0000.mp4"]


@pytest.mark.parametrize(
    ("label", "prefix"),
    [
        ("kneeling_stop_r15135_ours", "kneeling_stop_r15135_ours"),
        ("run 1/ours:full", "run_1_ours_full"),
        ("..hidden", "hidden"),
        ("", "eval"),
        (None, "eval"),
    ],
)
def test_clip_prefix_for_label_gives_a_plain_file_name_part(
    label: str | None, prefix: str
) -> None:
    assert clip_prefix_for_label(label) == prefix


def _touch(path: Path, mtime: float) -> Path:
    path.write_bytes(b"clip")
    os.utime(path, (mtime, mtime))
    return path


def test_clip_for_stem_accepts_the_old_and_the_develop_name(tmp_path: Path) -> None:
    assert clip_for_stem(tmp_path, "walk") is None

    old = _touch(tmp_path / "walk.mp4", 1_000.0)
    assert clip_for_stem(tmp_path, "walk") == old

    new = _touch(tmp_path / "walk_0000.mp4", 2_000.0)
    assert clip_for_stem(tmp_path, "walk") == new


def test_clip_for_stem_takes_the_newest_match(tmp_path: Path) -> None:
    _touch(tmp_path / "walk_0000.mp4", 1_000.0)
    newer_old_name = _touch(tmp_path / "walk.mp4", 3_000.0)
    _touch(tmp_path / "walk_0001.mp4", 2_000.0)

    assert clip_for_stem(tmp_path, "walk") == newer_old_name


def test_clip_for_stem_ignores_superseded_and_other_stems(tmp_path: Path) -> None:
    _touch(tmp_path / "walk_0000.mp4.superseded-20261007_100000", 9_000.0)
    _touch(tmp_path / "walk_fast_0000.mp4", 9_000.0)
    _touch(tmp_path / "run_0000.mp4", 9_000.0)
    _touch(tmp_path / "walk.mp4.superseded-20261007_100000", 9_000.0)

    assert clip_for_stem(tmp_path, "walk") is None
    with pytest.raises(FileNotFoundError, match="walk.mp4 and walk_<index>.mp4"):
        clip_for_stem(tmp_path, "walk", required=True)


def test_clip_for_stem_treats_dots_in_a_stem_literally(tmp_path: Path) -> None:
    _touch(tmp_path / "alpha0x25_0000.mp4", 1_000.0)
    wanted = _touch(tmp_path / "alpha0.25_0000.mp4", 1_000.0)

    assert clip_for_stem(tmp_path, "alpha0.25") == wanted


@pytest.mark.parametrize("video_length", [0, -3])
def test_attach_rejects_a_non_positive_clip_length(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, video_length: int
) -> None:
    _install_fake_launch_compat(monkeypatch, [])

    with pytest.raises(ValueError, match="video_length must be positive"):
        attach_video_capture(object(), output_dir=tmp_path, video_length=video_length)


def test_report_prints_the_absolute_path_of_each_new_clip(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    capture = VideoCapture(directory=tmp_path.resolve(), started_at=time.time() - 1.0)
    (tmp_path / "clip_0001.mp4").write_bytes(b"b")
    (tmp_path / "clip_0000.mp4").write_bytes(b"a")

    clips = capture.report()

    assert [clip.name for clip in clips] == ["clip_0000.mp4", "clip_0001.mp4"]
    lines = capsys.readouterr().out.splitlines()
    expected = [
        tmp_path.resolve() / "clip_0000.mp4",
        tmp_path.resolve() / "clip_0001.mp4",
    ]
    assert lines == [f"[VIDEO] {path}" for path in expected]
    assert all(Path(line.split(" ", 1)[1]).is_absolute() for line in lines)


def test_report_fails_when_no_clip_was_written(tmp_path: Path) -> None:
    capture = VideoCapture(directory=tmp_path.resolve(), started_at=time.time())

    with pytest.raises(RuntimeError, match="no .mp4 clip was written"):
        capture.report()


def test_report_ignores_a_stale_clip_from_an_earlier_run(tmp_path: Path) -> None:
    stale = tmp_path / "clip_0000.mp4"
    stale.write_bytes(b"old")
    long_ago = time.time() - 3600.0
    os.utime(stale, (long_ago, long_ago))
    capture = VideoCapture(directory=tmp_path.resolve(), started_at=time.time())

    with pytest.raises(RuntimeError, match="no .mp4 clip was written"):
        capture.report()


def test_retained_clip_is_the_first_recorder_clip(
    tmp_path: Path,
) -> None:
    assert retained_clip(tmp_path) is None

    (tmp_path / "clip_0001.mp4").write_bytes(b"b")
    (tmp_path / "clip_0000.mp4").write_bytes(b"a")
    assert retained_clip(tmp_path) == tmp_path / "clip_0000.mp4"


def test_report_warns_instead_of_failing_when_a_clip_is_not_required(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    capture = VideoCapture(directory=tmp_path.resolve(), started_at=time.time())

    assert capture.report(require_clip=False) == []
    assert "[WARN] Video was requested, but no .mp4 clip" in capsys.readouterr().out


class _FakeProxyArray:
    """Stand-in for a Warp-backed ``ProxyArray``: the tensor sits behind ``.torch``."""

    def __init__(self, tensor) -> None:
        self.torch = tensor


def _fake_env(root_pos_w) -> SimpleNamespace:
    camera_views: list[tuple] = []
    return SimpleNamespace(
        scene={"robot": SimpleNamespace(data=SimpleNamespace(root_pos_w=root_pos_w))},
        sim=SimpleNamespace(
            set_camera_view=lambda eye, target: camera_views.append((eye, target))
        ),
        camera_views=camera_views,
    )


@pytest.mark.parametrize("wrap_in_proxy_array", [False, True])
def test_aim_camera_at_robot_offsets_the_followed_root(
    wrap_in_proxy_array: bool,
) -> None:
    torch = pytest.importorskip("torch")
    root_pos_w = torch.tensor([[0.0, 0.0, 0.8], [3.0, -2.0, 0.9]])
    env = _fake_env(_FakeProxyArray(root_pos_w) if wrap_in_proxy_array else root_pos_w)

    aim_camera_at_robot(env, 1)

    ((eye, target),) = env.camera_views
    assert eye == pytest.approx(
        tuple(p + o for p, o in zip((3.0, -2.0, 0.9), FOLLOW_EYE_OFFSET))
    )
    assert target == pytest.approx(
        tuple(p + o for p, o in zip((3.0, -2.0, 0.9), FOLLOW_LOOKAT_OFFSET))
    )


def test_aim_camera_at_robot_takes_custom_offsets_and_asset_name() -> None:
    torch = pytest.importorskip("torch")
    env = _fake_env(torch.tensor([[1.0, 2.0, 3.0]]))
    env.scene["humanoid"] = env.scene.pop("robot")

    aim_camera_at_robot(
        env,
        0,
        asset_name="humanoid",
        eye_offset=(1.0, 0.0, 0.0),
        lookat_offset=(0.0, 0.0, -1.0),
    )

    assert env.camera_views == [((2.0, 2.0, 3.0), (1.0, 2.0, 2.0))]


def test_follow_robot_camera_aims_before_every_step_and_never_on_render() -> None:
    """The develop recorder captures inside step; a render hook would be a silent no-op."""
    torch = pytest.importorskip("torch")
    gym = pytest.importorskip("gymnasium")
    order: list[str] = []
    fake = _fake_env(torch.tensor([[1.0, 2.0, 3.0]]))
    fake.sim.set_camera_view = lambda eye, target: order.append("aim")

    class _Inner(gym.Env):
        unwrapped_env = fake

        @property
        def unwrapped(self):
            return fake

        def step(self, action):
            order.append("step")
            return None

        def render(self):
            order.append("render")

    wrapped = follow_robot_camera(
        _Inner(), 0, eye_offset=(1.0, 0.0, 0.0), lookat_offset=(0.0, 0.0, 0.5)
    )
    wrapped.step(None)
    wrapped.step(None)
    wrapped.render()
    assert order == ["aim", "step", "aim", "step", "render"]
