from pathlib import Path

import numpy as np
import pytest

from convert_mjlab_motion import convert
from imitationlab.envs.isaaclab.task_model.g1_kinematics import G1_ISAACLAB_JOINT_NAMES


DANCE102 = (
    Path(__file__).resolve().parents[2] / "data/unitree/npz/g1/G1_Take_102.bvh_60hz.npz"
)


def test_legacy_clip_uses_joint_names_and_half_open_range(tmp_path):
    names = G1_ISAACLAB_JOINT_NAMES[::-1]
    joints = "".join(f'<joint name="{name}"/>' for name in names)
    model = tmp_path / "g1.xml"
    model.write_text(f"<mujoco><worldbody><body>{joints}</body></worldbody></mujoco>")
    target = tmp_path / "motion.npz"

    convert(DANCE102, target, model, 1, 4)

    with np.load(DANCE102) as source, np.load(target) as motion:
        assert motion["joint_names"].tolist() == list(names)
        np.testing.assert_array_equal(
            motion["joint_pos"], source["joint_pos"][1:4, ::-1]
        )
        np.testing.assert_array_equal(
            motion["root_quat_w"], source["body_quat_w"][1:4, 0]
        )
    with pytest.raises(FileExistsError):
        convert(DANCE102, target, model, 1, 4)
