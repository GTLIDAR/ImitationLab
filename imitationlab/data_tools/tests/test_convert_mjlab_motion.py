import importlib.util
from pathlib import Path

import numpy as np
import pytest

from imitationlab.envs.isaaclab.task_model.g1_kinematics import G1_ISAACLAB_JOINT_NAMES

# The converter is a standalone script; load it by path.
_SCRIPT = Path(__file__).resolve().parents[3] / "scripts/data/convert_mjlab_motion.py"
_SPEC = importlib.util.spec_from_file_location("convert_mjlab_motion", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
convert = _MODULE.convert


def _legacy_clip(path: Path, num_frames: int = 6) -> Path:
    """Write a synthetic clip in the legacy layout: 30 unnamed bodies, no joint names."""
    rng = np.random.default_rng(0)

    def values(*shape: int) -> np.ndarray:
        return rng.standard_normal((num_frames, *shape)).astype(np.float32)

    np.savez(
        path,
        fps=np.asarray([50]),
        joint_pos=values(29),
        joint_vel=values(29),
        body_pos_w=values(30, 3),
        body_quat_w=values(30, 4),
        body_lin_vel_w=values(30, 3),
        body_ang_vel_w=values(30, 3),
    )
    return path


def test_legacy_clip_uses_joint_names_and_half_open_range(tmp_path):
    names = G1_ISAACLAB_JOINT_NAMES[::-1]
    joints = "".join(f'<joint name="{name}"/>' for name in names)
    model = tmp_path / "g1.xml"
    model.write_text(f"<mujoco><worldbody><body>{joints}</body></worldbody></mujoco>")
    clip = _legacy_clip(tmp_path / "legacy.npz")
    target = tmp_path / "motion.npz"

    convert(clip, target, model, 1, 4)

    with np.load(clip) as source, np.load(target) as motion:
        assert motion["joint_names"].tolist() == list(names)
        np.testing.assert_array_equal(
            motion["joint_pos"], source["joint_pos"][1:4, ::-1]
        )
        np.testing.assert_array_equal(
            motion["root_quat_w"], source["body_quat_w"][1:4, 0]
        )
    with pytest.raises(FileExistsError):
        convert(clip, target, model, 1, 4)
