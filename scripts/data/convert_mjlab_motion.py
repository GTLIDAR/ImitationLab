#!/usr/bin/env python3
"""Convert one ImitationLab G1 reference clip to SuperMimic's mjlab format."""

import argparse
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from imitationlab.envs.isaaclab.task_model.g1_kinematics import G1_ISAACLAB_JOINT_NAMES
from imitationlab.paths import REPO_ROOT


G1_XML = (
    REPO_ROOT
    / "third_party/mjlab_research/src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml"
)


def convert(
    source: Path, target: Path, model: Path, start: int, stop: int | None
) -> None:
    joints = [
        joint.attrib["name"]
        for joint in ElementTree.parse(model).findall("./worldbody//joint")
    ]
    with np.load(source, allow_pickle=False) as motion:
        source_joints = (
            motion["joint_names"].tolist()
            if "joint_names" in motion
            else list(G1_ISAACLAB_JOINT_NAMES)
        )
        if (
            len(joints) != 29
            or len(source_joints) != 29
            or set(joints) != set(source_joints)
        ):
            raise ValueError(
                "The clip and mjlab model must have the same 29 G1 joints."
            )
        if np.asarray(motion["fps"]).item() != 50:
            raise ValueError("Expected a 50 Hz reference clip.")
        num_frames = len(motion["joint_pos"])
        if stop is None:
            stop = num_frames
        if not 0 <= start < stop <= num_frames or stop - start < 2:
            raise ValueError(
                "The selected range must contain at least two frames within the clip."
            )
        if "body_names" in motion:
            pelvis = motion["body_names"].tolist().index("pelvis")
        elif motion["body_pos_w"].shape[1:] == (30, 3):
            pelvis = 0  # ImitationLab's legacy G1 NPZ body order.
        else:
            raise ValueError(
                "The clip needs body names or the legacy 30-body G1 layout."
            )
        frame_slice = slice(start, stop)
        indices = [source_joints.index(name) for name in joints]
        result = {
            "joint_names": np.asarray(joints),
            "joint_pos": motion["joint_pos"][frame_slice, indices],
            "joint_vel": motion["joint_vel"][frame_slice, indices],
            "root_pos_w": motion["body_pos_w"][frame_slice, pelvis],
            "root_quat_w": motion["body_quat_w"][frame_slice, pelvis],
            "root_lin_vel_w": motion["body_lin_vel_w"][frame_slice, pelvis],
            "root_ang_vel_w": motion["body_ang_vel_w"][frame_slice, pelvis],
        }
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        np.savez_compressed(stream, **result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Input ImitationLab G1 NPZ clip.")
    parser.add_argument("target", type=Path, help="New mjlab NPZ clip.")
    parser.add_argument("--model", type=Path, default=G1_XML, help="mjlab G1 XML.")
    parser.add_argument("--start", type=int, default=0, help="First input frame.")
    parser.add_argument("--stop", type=int, help="Exclusive last input frame.")
    args = parser.parse_args()
    convert(args.source, args.target, args.model, args.start, args.stop)


if __name__ == "__main__":
    main()
