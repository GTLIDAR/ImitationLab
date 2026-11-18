"""Batched torch forward kinematics of the 29-DoF G1 from its URDF.

Used to turn a ``root_qpos`` macro frame (29 joint positions plus the pelvis
pose) into the body poses a teleoperation device or a keypoint source would
observe: head/torso and wrists for VR three-point tracking, the 14 tracked
bodies for keypoint input. Everything is differentiable and runs on the
frames' device; the kinematic constants are buffers.

Conventions (verified against the reference arrays' ``body_pos_w`` in
``tests/test_teleop_views.py``):

* joint angles are given in ``joint_names`` order (default: the IsaacLab
  articulation order, which is the order of the macro state's joint block and
  of the reference arrays' ``qpos[:, 7:]``);
* the root is the ``pelvis`` link, given as a position and a rotation matrix
  in whatever frame the caller works in; every output pose is in that frame.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

G1_URDF_PATH = (
    Path(__file__).resolve().parents[1]
    / "assets/unitree/g1_description/g1_29dof_rev_1_0.urdf"
)

# IsaacLab articulation (level-order) joint order: the macro state's joint block.
# This matches G1_29DOF_ISAACLAB_JOINT_NAMES in the simulator asset.
# Keep this module importable without the Isaac stack.
G1_ISAACLAB_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)

# The 14 bodies the reference arrays retain (SONIC's tracked set).
G1_TRACKED_BODY_NAMES: tuple[str, ...] = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> Tensor:
    """URDF fixed-axis rpy: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return torch.tensor(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=torch.float64,
    )


def axis_angle_matrix(axis: Tensor, angle: Tensor) -> Tensor:
    """Rodrigues rotation for unit ``axis`` [3] and angles [...] -> [..., 3, 3]."""
    x, y, z = (axis[0], axis[1], axis[2])
    c, s = torch.cos(angle), torch.sin(angle)
    t = 1.0 - c
    rows = [
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ]
    return torch.stack([torch.stack(row, dim=-1) for row in rows], dim=-2)


# For a rotation about coordinate axis a, the plane it turns: (i, j) with (i, j, a) cyclic.
_CYCLIC_PAIR = {0: (1, 2), 1: (2, 0), 2: (0, 1)}


@dataclass(frozen=True)
class _Joint:
    name: str
    kind: str
    parent: str
    child: str
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    axis: tuple[float, float, float]


def _parse_urdf(path: Path) -> list[_Joint]:
    root = ET.parse(path).getroot()
    joints: list[_Joint] = []
    for element in root.findall("joint"):
        origin = element.find("origin")
        axis = element.find("axis")
        xyz = (
            tuple(float(v) for v in origin.get("xyz", "0 0 0").split())
            if origin is not None
            else (0.0, 0.0, 0.0)
        )
        rpy = (
            tuple(float(v) for v in origin.get("rpy", "0 0 0").split())
            if origin is not None
            else (0.0, 0.0, 0.0)
        )
        joints.append(
            _Joint(
                name=str(element.get("name")),
                kind=str(element.get("type")),
                parent=str(element.find("parent").get("link")),  # type: ignore[union-attr]
                child=str(element.find("child").get("link")),  # type: ignore[union-attr]
                xyz=xyz,  # type: ignore[arg-type]
                rpy=rpy,  # type: ignore[arg-type]
                axis=(
                    tuple(float(v) for v in axis.get("xyz", "1 0 0").split())  # type: ignore[arg-type]
                    if axis is not None
                    else (1.0, 0.0, 0.0)
                ),
            )
        )
    return joints


class G1Kinematics(nn.Module):
    """Forward kinematics from the pelvis to a chosen set of G1 links."""

    def __init__(
        self,
        *,
        urdf_path: Path | str = G1_URDF_PATH,
        joint_names: Sequence[str] = G1_ISAACLAB_JOINT_NAMES,
        root_link: str = "pelvis",
    ) -> None:
        super().__init__()
        joints = _parse_urdf(Path(urdf_path))
        by_child = {joint.child: joint for joint in joints}
        self.root_link = str(root_link)
        self.joint_names = tuple(str(name) for name in joint_names)
        index_of = {name: i for i, name in enumerate(self.joint_names)}
        revolute = [j.name for j in joints if j.kind in {"revolute", "continuous"}]
        missing = sorted(set(revolute) - set(index_of))
        if missing:
            msg = f"joint_names lacks URDF revolute joints {missing}."
            raise ValueError(msg)
        # Topological order: every joint after the joint producing its parent.
        ordered: list[_Joint] = []
        produced = {self.root_link}
        pending = list(joints)
        while pending:
            progressed = False
            for joint in list(pending):
                if joint.parent in produced:
                    ordered.append(joint)
                    produced.add(joint.child)
                    pending.remove(joint)
                    progressed = True
            if not progressed:
                break
        self._joints = ordered
        self.link_names = (self.root_link, *(j.child for j in ordered))
        self._link_index = {name: i for i, name in enumerate(self.link_names)}
        self._parent_index = [self._link_index[j.parent] for j in ordered]
        self._qpos_index = [
            index_of.get(j.name, -1) if j.kind != "fixed" else -1 for j in ordered
        ]
        self.register_buffer(
            "origin_pos", torch.tensor([j.xyz for j in ordered], dtype=torch.float32)
        )
        self.register_buffer(
            "origin_rot",
            torch.stack([_rpy_matrix(*j.rpy) for j in ordered]).to(torch.float32),
        )
        self.register_buffer(
            "axes", torch.tensor([j.axis for j in ordered], dtype=torch.float32)
        )
        self._origin_rotated = [any(abs(v) > 1.0e-12 for v in j.rpy) for j in ordered]
        # Coordinate index of each joint's axis when it is +x, +y or +z, else -1.
        unit = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        self._axis_slot = [
            unit.index(tuple(j.axis)) if tuple(j.axis) in unit else -1 for j in ordered
        ]
        del by_child

    def link_index(self, name: str) -> int:
        return self._link_index[name]

    def forward(
        self,
        root_pos: Tensor,
        root_rot: Tensor,
        joint_pos: Tensor,
        links: Sequence[str],
    ) -> tuple[Tensor, Tensor]:
        """Poses of ``links``: positions ``[..., L, 3]`` and rotations ``[..., L, 3, 3]``."""
        positions, rotations, _, _ = self._walk(root_pos, root_rot, joint_pos, links)
        wanted = [self._link_index[name] for name in links]
        return (
            torch.stack([positions[i] for i in wanted], dim=-2),
            torch.stack([rotations[i] for i in wanted], dim=-3),
        )

    def forward_with_velocity(
        self,
        root_pos: Tensor,
        root_rot: Tensor,
        joint_pos: Tensor,
        root_lin_vel: Tensor,
        root_ang_vel: Tensor,
        joint_vel: Tensor,
        links: Sequence[str],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Poses and exact link-origin twists of ``links``.

        Velocities are in the same frame as the root inputs. A revolute
        joint's anchor is its child link origin, so
        ``w_child = w_parent + R_joint @ axis * qdot`` and
        ``v_child = v_parent + w_parent x (p_child - p_parent)``.
        Returns positions, rotations, linear and angular velocities.
        """
        positions, rotations, lin, ang = self._walk(
            root_pos, root_rot, joint_pos, links, root_lin_vel, root_ang_vel, joint_vel
        )
        wanted = [self._link_index[name] for name in links]
        return (
            torch.stack([positions[i] for i in wanted], dim=-2),
            torch.stack([rotations[i] for i in wanted], dim=-3),
            torch.stack([lin[i] for i in wanted], dim=-2),
            torch.stack([ang[i] for i in wanted], dim=-2),
        )

    def _walk(
        self,
        root_pos: Tensor,
        root_rot: Tensor,
        joint_pos: Tensor,
        links: Sequence[str],
        root_lin_vel: Tensor | None = None,
        root_ang_vel: Tensor | None = None,
        joint_vel: Tensor | None = None,
    ) -> tuple[
        dict[int, Tensor], dict[int, Tensor], dict[int, Tensor], dict[int, Tensor]
    ]:
        wanted = [self._link_index[name] for name in links]
        needed = set(wanted)
        # Walk up the tree once to know which links must be computed.
        frontier = list(wanted)
        while frontier:
            link = frontier.pop()
            if link == 0:
                continue
            parent = self._parent_index[link - 1]
            if parent not in needed:
                needed.add(parent)
                frontier.append(parent)
        with_velocity = root_lin_vel is not None
        positions: dict[int, Tensor] = {0: root_pos}
        rotations: dict[int, Tensor] = {0: root_rot}
        lin: dict[int, Tensor] = {}
        ang: dict[int, Tensor] = {}
        if with_velocity:
            lin[0], ang[0] = root_lin_vel, root_ang_vel
        for k, joint in enumerate(self._joints):
            link = k + 1
            if link not in needed:
                continue
            parent = self._parent_index[k]
            parent_rot = rotations[parent]
            position = positions[parent] + (
                parent_rot * self.origin_pos[k].to(dtype=root_pos.dtype)
            ).sum(-1)
            rotation = parent_rot
            if self._origin_rotated[k]:
                # Constant right factor: one plain GEMM over all rows.
                local_rot = self.origin_rot[k].to(dtype=root_rot.dtype)
                rotation = (parent_rot.reshape(-1, 3) @ local_rot).reshape(
                    parent_rot.shape
                )
            q_index = self._qpos_index[k]
            if with_velocity:
                omega = ang[parent]
                lin[link] = lin[parent] + torch.linalg.cross(
                    omega, position - positions[parent], dim=-1
                )
                if q_index >= 0:
                    # Joint axis in the root frame: joint frame = parent * origin.
                    axis_w = (rotation * self.axes[k].to(dtype=rotation.dtype)).sum(-1)
                    omega = omega + axis_w * joint_vel[..., q_index].unsqueeze(-1)
                ang[link] = omega
            if q_index >= 0:
                if self._axis_slot[k] >= 0:
                    # R @ Rot(e_a, q) for a coordinate axis e_a: columns (i, j) after a
                    # rotate by the angle, column a is unchanged. Far cheaper than a
                    # batched 3x3 matmul with a Rodrigues matrix.
                    angle = joint_pos[..., q_index].unsqueeze(-1)
                    c, s_ = torch.cos(angle), torch.sin(angle)
                    i, j = _CYCLIC_PAIR[self._axis_slot[k]]
                    col_i, col_j = rotation[..., :, i], rotation[..., :, j]
                    cols = [
                        rotation[..., :, 0],
                        rotation[..., :, 1],
                        rotation[..., :, 2],
                    ]
                    cols[i] = c * col_i + s_ * col_j
                    cols[j] = c * col_j - s_ * col_i
                    rotation = torch.stack(cols, dim=-1)
                else:
                    rotation = rotation @ axis_angle_matrix(
                        self.axes[k].to(dtype=joint_pos.dtype), joint_pos[..., q_index]
                    )
            positions[link] = position
            rotations[link] = rotation
        return positions, rotations, lin, ang


# ---------------------------------------------------------------------------
# rot6d layouts
# ---------------------------------------------------------------------------
def rot6d_to_matrix(rot6d: Tensor, *, layout: str = "interleaved") -> Tensor:
    """6D rotation -> 3x3 (Gram-Schmidt) for a named memory layout.

    ``interleaved``: ``R[:, :2]`` flattened row-major,
    ``[r00, r01, r10, r11, r20, r21]`` -- what the data plane's
    ``quat_to_rot6d_flat`` writes into the macro state's anchor block.
    ``columns``: ``[r00, r10, r20, r01, r11, r21]`` (first column, then second).
    """
    if layout == "interleaved":
        a1 = rot6d[..., 0::2]
        a2 = rot6d[..., 1::2]
    elif layout == "columns":
        a1 = rot6d[..., 0:3]
        a2 = rot6d[..., 3:6]
    else:
        msg = f"unknown rot6d layout {layout!r}."
        raise ValueError(msg)
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = torch.nn.functional.normalize(
        a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1
    )
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def matrix_to_rot6d(rotation: Tensor, *, layout: str = "interleaved") -> Tensor:
    """Inverse of :func:`rot6d_to_matrix` for the same ``layout``."""
    two = rotation[..., :, :2]
    if layout == "interleaved":
        return two.reshape(*rotation.shape[:-2], 6)
    if layout == "columns":
        return torch.cat([two[..., :, 0], two[..., :, 1]], dim=-1)
    msg = f"unknown rot6d layout {layout!r}."
    raise ValueError(msg)


def quat_wxyz_to_matrix(quat: Tensor) -> Tensor:
    w, x, y, z = quat.unbind(-1)
    norm = torch.sqrt(w * w + x * x + y * y + z * z).clamp_min(1.0e-12)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    rows = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]
    return torch.stack([torch.stack(row, dim=-1) for row in rows], dim=-2)
