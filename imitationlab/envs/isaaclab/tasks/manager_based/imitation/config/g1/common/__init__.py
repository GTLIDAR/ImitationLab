"""Shared building blocks of the G1 task configuration: the action term and the
joint and body name tables."""

from .actions import G1ActionsCfg
from .constants import (
    G1_29DOF_DATASET_BODY_NAMES,
    G1_29DOF_ISAACLAB_JOINT_NAMES,
    G1_29DOF_JOINT_NAMES,
    G1_EE_BODY_NAMES,
    G1_KEYPOINT5_BODY_NAMES,
    G1_TRACKED_BODY_NAMES,
)

__all__ = [
    "G1ActionsCfg",
    "G1_29DOF_DATASET_BODY_NAMES",
    "G1_29DOF_ISAACLAB_JOINT_NAMES",
    "G1_29DOF_JOINT_NAMES",
    "G1_EE_BODY_NAMES",
    "G1_KEYPOINT5_BODY_NAMES",
    "G1_TRACKED_BODY_NAMES",
]
