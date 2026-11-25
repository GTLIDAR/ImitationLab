"""G1 motion and MDP equations of ``imitation-g1-v1``, independent of the simulator runtime.
License notice: ``third_party/licenses/supermimic_vibed.txt``.

Reimplemented from ``zita-ch/supermimic_vibed`` at commit ``0525a23``
(MIT license), specifically ``motion.py``, ``mdp.py`` and ``kinematics.py``.
No code from the reference checkout is imported or executed. Joint coordinates
use this repository's canonical named order; the reference uses SDK order.

The input bank is this repository's 50 Hz qpos arrays. Velocities are derived
again within each clip by central differences (one-sided at its endpoints),
as in the replica. Body velocities are FK link-origin twists, never COM
velocities. By default only the 71-D physical states are resident; body FK is
evaluated on gathered rows, avoiding a second full BONES body-state bank. A run
on a small bank can ask for tables of goals and body states (``wm.goal_table``,
``wm.reference_body_table``), which hold the same numbers.

``G1MDP(reward_kernel_scale=k)`` multiplies the six tracking-kernel widths by
``k``, as NDM Gym's ``--reward_std_scale`` does (``training_reward_cfg`` in
``ndm_gym/train.py`` at ``05ea9f3``; MIT notices in
``third_party/licenses/ndm_gym.txt``). At ``k = 1`` the arithmetic is the
unscaled one bit for bit.

``G1MDP(reward_heading_velocity=True)`` turns the reference body velocities by
the same heading offset as the re-anchored pose targets before the two velocity
terms compare them, as upstream SuperMimic ``5fb9229`` does. Off by default.

``drift_weight``, ``semi_implicit_drift``, ``structured_targets`` and
``structured_integrate`` (``wm.wm_structured``) port the drift, ``targets`` and
``integrate`` of NDM Gym's ``SemiImplicitIncrementWorldModel``
(``ndm_gym/worldmodels/semi_implicit.py`` at ``05ea9f3``, itself from BlinkMimic
``d264658``; same MIT notices) to this module's 71-D state and WXYZ quaternions.
``mirror_joint_map`` and ``G1Mirror`` (``wm.wm_mirror``) port ``Mirror`` of
``ndm_gym/algos/blinkmimic.py`` (same notices), by joint name. ``G1MDP.reward``'s
``contact`` argument adds ``g1_contact``'s term as ``ndm_gym/tasks/tracking.py``'s
``reward(..., contact=...)`` does (same notices).
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from imitationlab.envs.isaaclab.task_model.g1_kinematics import (
    G1_ISAACLAB_JOINT_NAMES,
    G1_TRACKED_BODY_NAMES,
    G1_URDF_PATH,
    G1Kinematics,
    matrix_to_rot6d,
    quat_wxyz_to_matrix,
)
from imitationlab.envs.isaaclab.task_model.g1_contact import contact_term
from imitationlab.envs.isaaclab.task_model.quat import (
    quat_exp,
    quat_mul,
    quat_normalize,
)
from imitationlab.envs.isaaclab.task_model.rotations import mat_to_quat

STATE_DIM = 71
"""Physical state: pelvis position (env-local), quaternion (w, x, y, z), world linear
and angular velocity of the pelvis link origin, joint positions and velocities in
``G1_ISAACLAB_JOINT_NAMES`` order."""
ACTION_DIM = 29
INCREMENT_DIM = 70


def unpack(state: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    return (
        state[..., 0:3],
        state[..., 3:7],
        state[..., 7:10],
        state[..., 10:13],
        state[..., 13:42],
        state[..., 42:71],
    )


def pack(
    p: Tensor, quat: Tensor, v: Tensor, w: Tensor, q: Tensor, qd: Tensor
) -> Tensor:
    return torch.cat((p, quat, v, w, q, qd), dim=-1)


CONTROL_DT = 0.02
REFERENCE_TABLE_LIMIT_BYTES = 8 * 2**30
"""Largest table of one kind: LAFAN1 needs 1.2 GiB of goals, a BONES bank about 129."""
TABLE_CHUNK = 4096
"""Rows per call when a table is built. The computation it replaces rounds differently
at some batch shapes: on an RTX A4500, 34 of 98.7M goal entries of LAFAN1 differ between
batches of 65536 and 135168, by at most 1e-6 relative. Batches of 777, 4096 and 135168
agree with each other, so this size matches what training computes there."""
PROPRIO_DIM = 99
GOAL_FRAME_DIM = 73
GOAL_FRAME_JOINT_POS = 3 + 6 + 3 + 3
"""Where the reference joint positions start in one goal frame of ``G1ReferenceLibrary.goal``:
after the position, the rotation (6D) and the linear and angular velocities."""
CONTEXT_DIM = 33
OBS_NOISE_WIDTH = (
    (0.01,) * 29 + (0.5,) * 29 + (0.2,) * 3 + (0.05,) * 3 + (0.0,) * 29 + (0.05,) * 6
)
# OBS_SCALE in the replica's mdp.py is unused by its actual Policy/train path.
# The task deliberately emits the raw noisy observation; the actor owns normalization.
REWARD_WEIGHTS = {
    "body_pos": 1.0,
    "body_ori": 1.0,
    "body_lin_vel": 1.0,
    "body_ang_vel": 1.0,
    "root_pos": 0.5,
    "root_ori": 0.5,
    "action_rate": -0.1,
    "joint_limit": -2.0,
}


def rotate(rotation: Tensor, vector: Tensor) -> Tensor:
    return (rotation @ vector.unsqueeze(-1)).squeeze(-1)


def heading(rotation: Tensor) -> Tensor:
    """Replica heading, including its epsilon at a vertical projected axis."""
    x, y = rotation[..., 0, 0], rotation[..., 1, 0]
    norm = torch.sqrt(x.square() + y.square() + 1e-9)
    c, s = x / norm, y / norm
    zero, one = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack((c, -s, zero, s, c, zero, zero, zero, one), -1).reshape(
        *c.shape, 3, 3
    )


def rotation_vector(rotation: Tensor) -> Tensor:
    """Shortest SO(3) log, stable for the endpoint finite difference near pi."""
    quat = mat_to_quat(rotation)
    xyz = quat[..., 1:]
    sine = xyz.norm(dim=-1, keepdim=True)
    angle = 2 * torch.atan2(sine, quat[..., :1])
    scale = torch.where(
        sine < 1e-6, 2 / quat[..., :1].clamp_min(1e-6), angle / (sine + 1e-12)
    )
    return xyz * scale


def rotation_error_sq(a: Tensor, b: Tensor) -> Tensor:
    """The replica's clamped squared geodesic angle, including its zero floor."""
    cosine = ((a * b).sum((-1, -2)) - 1) / 2
    return cosine.clamp(-1 + 1e-6, 1 - 1e-6).acos().square()


def normalize_context(context: Tensor) -> Tensor:
    return torch.cat(
        (
            (context[..., :1] - 0.75) / 0.45,
            context[..., 1:30] / 0.01,
            context[..., 30:33] / 0.05,
        ),
        -1,
    )


def invariant_features(state: Tensor) -> Tensor:
    p, quat, v, w, q, qd = unpack(state)
    rotation = quat_wxyz_to_matrix(quat)
    transpose = rotation.transpose(-1, -2)
    return torch.cat(
        (
            p[..., 2:3],
            matrix_to_rot6d(heading(rotation).transpose(-1, -2) @ rotation),
            rotate(transpose, v),
            rotate(transpose, w),
            q,
            qd,
        ),
        -1,
    )


def increment(state: Tensor, next_state: Tensor) -> Tensor:
    """Replica local-coordinate state difference (71 physical, 70 tangent)."""
    p, quat, v, w, q, qd = unpack(state)
    pn, quatn, vn, wn, qn, qdn = unpack(next_state)
    rotation = quat_wxyz_to_matrix(quat)
    rt = rotation.transpose(-1, -2)
    return torch.cat(
        (
            rotate(heading(rotation).transpose(-1, -2), pn - p),
            rotation_vector(rt @ quat_wxyz_to_matrix(quatn)),
            rotate(rt, vn - v),
            rotate(rt, wn - w),
            qn - q,
            qdn - qd,
        ),
        -1,
    )


def integrate(state: Tensor, residual: Tensor) -> Tensor:
    p, quat, v, w, q, qd = unpack(state)
    rotation = quat_wxyz_to_matrix(quat)
    return pack(
        p + rotate(heading(rotation), residual[..., :3]),
        quat_normalize(quat_mul(quat, quat_exp(residual[..., 3:6]))),
        v + rotate(rotation, residual[..., 6:9]),
        w + rotate(rotation, residual[..., 9:12]),
        q + residual[..., 12:41],
        qd + residual[..., 41:70],
    )


def state_error(predicted: Tensor, truth: Tensor) -> Tensor:
    return increment(truth, predicted)


def drift_weight(substeps: int) -> float:
    """``wm.wm_structured``: the weight c of the new velocity in the position change of one
    control step of ``substeps`` semi-implicit substeps whose velocity changes linearly from
    v to v': sum_k k / n^2 = (n + 1) / (2 n), 0.625 for the task's four Newton substeps."""
    if substeps < 1:
        raise ValueError(f"A control step needs at least one substep, got {substeps}.")
    return (substeps + 1) / (2 * substeps)


def semi_implicit_drift(
    state: Tensor,
    v1: Tensor,
    w1: Tensor,
    qd1: Tensor,
    weight: float,
    dt: float = CONTROL_DT,
) -> tuple[Tensor, Tensor, Tensor]:
    """The position change dt ((1 - c) v + c v') that the old and new velocities imply, in
    the frames of ``increment``: the heading frame for the root position, the body frame
    for the rotation vector, joint coordinates for the joints."""
    _, quat, v, w, _, qd = unpack(state)
    rotation = quat_wxyz_to_matrix(quat)
    a, b = dt * (1 - weight), dt * weight
    return (
        rotate(heading(rotation).transpose(-1, -2), a * v + b * v1),
        rotate(rotation.transpose(-1, -2), a * w + b * w1),
        a * qd + b * qd1,
    )


def structured_targets(state: Tensor, next_state: Tensor, weight: float) -> Tensor:
    """The semi-implicit model's outputs (70) for ``state -> next_state``: the velocity
    increments of ``increment``, and for the positions the corrections on top of the drift."""
    inc = increment(state, next_state)
    _, _, v1, w1, _, qd1 = unpack(next_state)
    gp, gth, gq = semi_implicit_drift(state, v1, w1, qd1, weight)
    return torch.cat(
        (
            inc[..., :3] - gp,
            inc[..., 3:6] - gth,
            inc[..., 6:12],
            inc[..., 12:41] - gq,
            inc[..., 41:70],
        ),
        -1,
    )


def structured_integrate(state: Tensor, output: Tensor, weight: float) -> Tensor:
    """The next state from the semi-implicit model's denormalized outputs: new velocities
    first, then positions by the drift plus the corrections (``integrate``)."""
    _, quat, v, w, _, qd = unpack(state)
    rotation = quat_wxyz_to_matrix(quat)
    v1 = v + rotate(rotation, output[..., 6:9])
    w1 = w + rotate(rotation, output[..., 9:12])
    qd1 = qd + output[..., 41:70]
    gp, gth, gq = semi_implicit_drift(state, v1, w1, qd1, weight)
    return integrate(
        state,
        torch.cat(
            (
                output[..., :3] + gp,
                output[..., 3:6] + gth,
                output[..., 6:12],
                output[..., 12:41] + gq,
                output[..., 41:70],
            ),
            -1,
        ),
    )


def joint_limits(
    device: torch.device | str, dtype: torch.dtype = torch.float32
) -> Tensor:
    """Hard URDF joint limits in the bank's canonical named order."""
    nodes = {
        node.attrib["name"]: node.find("limit")
        for node in ET.parse(G1_URDF_PATH).getroot().findall("joint")
    }
    return torch.tensor(
        [
            [float(nodes[name].attrib[bound]) for bound in ("lower", "upper")]
            for name in G1_ISAACLAB_JOINT_NAMES
        ],
        device=device,
        dtype=dtype,
    )


def mirror_joint_map(
    names: tuple[str, ...] = G1_ISAACLAB_JOINT_NAMES,
) -> tuple[list[int], list[float]]:
    """``wm.wm_mirror``: each joint's left-right partner (its index in ``names``) and the
    sign of its mirrored angle, by name, since this repository's joint order is not
    NDM Gym's. Roll and yaw axes reverse under the reflection, the waist's too (-1);
    pitch, knee and elbow axes do not (+1)."""
    perm, sign = [], []
    for name in names:
        partner = (
            name.replace("left_", "@").replace("right_", "left_").replace("@", "right_")
        )
        perm.append(names.index(partner))
        sign.append(
            -1.0 if any(f"_{axis}_" in name for axis in ("roll", "yaw")) else 1.0
        )
    return perm, sign


class G1Mirror:
    """``wm.wm_mirror``: the reflection M = diag(1, -1, 1) of the world's y axis on the
    71-D state, the 29 joint coordinates (also the actions, normalized setpoints, when
    the default pose and the action scale are symmetric) and the 33-D context.

    Positions and linear velocities map to M x, rotations to M R M, which on a WXYZ
    quaternion is (w, -x, y, -z), angular velocities (pseudo-vectors) to -M w, joints
    to their partners (``mirror_joint_map``). In the context the friction stays, the
    joint offsets mirror as joints and the torso's lateral COM offset changes sign.
    Every map is a permutation with signs, so it is exact and its own inverse.
    """

    def __init__(self, device: torch.device | str) -> None:
        perm, sign = mirror_joint_map()
        self.perm = torch.tensor(perm, device=device)
        self.sign = torch.tensor(sign, device=device)
        self.vector = torch.tensor([1.0, -1.0, 1.0], device=device)
        self.quat = torch.tensor([1.0, -1.0, 1.0, -1.0], device=device)

    def joints(self, x: Tensor) -> Tensor:
        return x[..., self.perm] * self.sign

    def state(self, state: Tensor) -> Tensor:
        p, quat, v, w, q, qd = unpack(state)
        return pack(
            p * self.vector,
            quat * self.quat,
            v * self.vector,
            -w * self.vector,
            self.joints(q),
            self.joints(qd),
        )

    def context(self, context: Tensor) -> Tensor:
        return torch.cat(
            (
                context[..., :1],
                self.joints(context[..., 1:30]),
                context[..., 30:33] * self.vector,
            ),
            -1,
        )


@dataclass
class G1ReferenceFrames:
    root_pos: Tensor
    root_rot: Tensor
    root_lin_vel: Tensor
    root_ang_vel: Tensor
    joint_pos: Tensor
    joint_vel: Tensor
    body_pos: Tensor
    body_rot: Tensor
    body_lin_vel: Tensor
    body_ang_vel: Tensor
    valid: Tensor


class G1ReferenceLibrary:
    """Compact qpos-derived 50 Hz bank shared by the live env and task model."""

    def __init__(
        self,
        env: Any,
        device: torch.device | str,
        chunk: int = 65536,
        *,
        host_mapped: bool = False,
        managed: bool = False,
        host_shared: bool = False,
        shared_file: str = "",
    ) -> None:
        """``host_mapped`` (env ``library_host_mapped``): keep ``states`` in host
        memory registered with CUDA (``HostMappedTensor``) and expose its device alias,
        so every gather reads the rows over PCIe inside the kernel and the bank takes no
        GPU memory (47.5M rows of the full BONES-SEED bank are 13.5 GiB). ``managed``
        (env ``library_managed``): the rows on CUDA managed memory instead
        (``ManagedTensor``): pages the GPU reads migrate to it and stay as read-only
        duplicates, so the hot clips are served from device memory. Values and indices
        are the same either way, so reads equal the resident library bit for bit."""
        tm = env.trajectory_manager
        self.device = torch.device(device)
        self.host_mapped = bool(host_mapped)
        self.shared_file = str(shared_file)
        self._shared_rows: Any = None
        if (host_shared or shared_file) and not host_mapped:
            raise ValueError(
                "a shared reference library requires library_host_mapped=true."
            )
        self.managed = bool(managed)
        if self.host_mapped and self.managed:
            raise ValueError("library_host_mapped and library_managed are exclusive.")
        self._host_mapping: Any = None
        self.start = tm.start.to(device=self.device, dtype=torch.long)
        self.length = tm.length.to(device=self.device, dtype=torch.long)
        if chunk < 1 or self.start.numel() == 0 or bool((self.length < 2).any()):
            raise ValueError(
                "The reference library requires a nonempty bank with at least two frames per clip and a positive chunk."
            )
        expected = torch.cat((self.length.new_zeros(1), self.length.cumsum(0)[:-1]))
        if not torch.equal(self.start, expected):
            raise ValueError("Reference clips must form a contiguous, zero-based bank.")
        names = tuple(
            getattr(tm, "reference_joint_names", ()) or G1_ISAACLAB_JOINT_NAMES
        )
        if names != G1_ISAACLAB_JOINT_NAMES:
            raise ValueError("The qpos joint order must be G1_ISAACLAB_JOINT_NAMES.")
        self.body_names = G1_TRACKED_BODY_NAMES
        self.kin = G1Kinematics().to(self.device)
        self.goal_table: Tensor | None = None
        self.body_table: tuple[Tensor, Tensor, Tensor, Tensor] | None = None
        self.num_rows = int((self.start + self.length).max().item())
        if self.host_mapped:
            if self.device.type != "cuda":
                raise ValueError(
                    "library_host_mapped maps host rows into a CUDA device."
                )
            from imitationlab.envs.isaaclab.envs.host_mapped import HostMappedTensor

            if host_shared or shared_file:
                from imitationlab.envs.isaaclab.task_model.shared_host_rows import (
                    SharedHostRows,
                )

                self._shared_rows = SharedHostRows(self.num_rows, 71, path=shared_file)
                self.shared_file = self._shared_rows.path
                host = self._shared_rows.host
            else:
                host = torch.zeros(self.num_rows, 71)
            self._host_mapping = HostMappedTensor(host, self.device)
            self.states = self._host_mapping.device_view
        elif self.managed:
            if self.device.type != "cuda":
                raise ValueError("library_managed is read through a CUDA device.")
            from imitationlab.envs.isaaclab.envs.managed import ManagedTensor

            self._host_mapping = ManagedTensor(
                (self.num_rows, 71), torch.float32, self.device
            )
            self.states = self._host_mapping.tensor
            # The fill writes from the device: keep the pages home on the host for it,
            # so no 13 GiB migrates in; afterwards let the reads pull their pages.
            self._host_mapping.prefer_host(True)
        else:
            self.states = torch.zeros(self.num_rows, 71, device=self.device)
        if shared_file:
            # The trainer publishes this descriptor only after construction and
            # a device synchronization. Never rewrite the shared rows here.
            return
        for lo in range(0, self.num_rows, chunk):
            hi = min(lo + chunk, self.num_rows)
            rows = tm.rb[torch.arange(lo, hi, device=tm.storage_device)]["qpos"]
            self.states[lo:hi, :7].copy_(rows[:, :7])
            self.states[lo:hi, 13:42].copy_(rows[:, 7:36])
        # A second chunked pass needs only neighboring stored poses; no body
        # arrays, full-bank matrix rotations, or whole-bank temporary copy.
        for lo in range(0, self.num_rows, chunk):
            hi = min(lo + chunk, self.num_rows)
            index = torch.arange(lo, hi, device=self.device)
            rank = torch.searchsorted(self.start, index, right=True) - 1
            first = self.start[rank]
            last = first + self.length[rank] - 1
            prev = (index - 1).maximum(first)
            nxt = (index + 1).minimum(last)
            dt = ((nxt - prev).float() * CONTROL_DT).unsqueeze(-1)
            s0, s1 = self.states[prev], self.states[nxt]
            self.states[lo:hi, 7:10] = (s1[:, :3] - s0[:, :3]) / dt
            self.states[lo:hi, 10:13] = (
                rotation_vector(
                    quat_wxyz_to_matrix(s1[:, 3:7])
                    @ quat_wxyz_to_matrix(s0[:, 3:7]).transpose(-1, -2)
                )
                / dt
            )
            self.states[lo:hi, 42:71] = (s1[:, 13:42] - s0[:, 13:42]) / dt
        if self.managed:
            torch.cuda.synchronize(self.device)
            self._host_mapping.prefer_host(False)
            self._host_mapping.read_mostly(True)
        if host_shared:
            torch.cuda.synchronize(self.device)

    @classmethod
    def from_qpos(
        cls,
        qpos: Tensor,
        lengths: Tensor | list[int],
        *,
        host_mapped: bool = False,
        managed: bool = False,
        host_shared: bool = False,
        shared_file: str = "",
    ) -> G1ReferenceLibrary:
        """Build from already ordered 50 Hz rows (also useful for CPU tests)."""
        from types import SimpleNamespace

        lens = torch.as_tensor(lengths, device=qpos.device, dtype=torch.long)
        starts = torch.cat((lens.new_zeros(1), lens.cumsum(0)[:-1]))
        if int(lens.sum()) != len(qpos):
            raise ValueError("Clip lengths must cover every qpos row exactly.")

        class Rows:
            def __getitem__(self, index: Tensor) -> dict[str, Tensor]:
                return {"qpos": qpos[index]}

        tm = SimpleNamespace(
            start=starts, length=lens, rb=Rows(), storage_device=qpos.device
        )
        return cls(
            SimpleNamespace(trajectory_manager=tm),
            qpos.device,
            host_mapped=host_mapped,
            managed=managed,
            host_shared=host_shared,
            shared_file=shared_file,
        )

    def resident_bytes(self) -> int:
        """Bytes of the rows and tables on the device; a host-mapped or managed bank's
        rows live in host memory (managed: migrating as read) and count 0 here."""
        tables = (
            *(self.body_table or ()),
            *(() if self.goal_table is None else (self.goal_table,)),
        )
        rows = () if (self.host_mapped or self.managed) else (self.states,)
        return sum(t.numel() * t.element_size() for t in (*rows, *tables))

    def _bank_rows(self) -> tuple[Tensor, Tensor]:
        """Rank and frame of every bank row."""
        index = torch.arange(self.num_rows, device=self.device)
        rank = torch.searchsorted(self.start, index, right=True) - 1
        return rank, index - self.start[rank]

    def _check_table_size(self, name: str, row_floats: int) -> None:
        size = self.num_rows * row_floats * 4
        if size > REFERENCE_TABLE_LIMIT_BYTES:
            raise ValueError(
                f"wm.{name}: {self.num_rows} rows need {size / 2**30:.1f} GiB, above "
                f"the limit of {REFERENCE_TABLE_LIMIT_BYTES / 2**30:.0f} GiB."
            )

    @torch.no_grad()
    def build_goal_table(self, count: int = 10, chunk: int = TABLE_CHUNK) -> None:
        """``wm.goal_table``: ``goal`` of every row, computed once by the same code,
        which ``goal`` then reads. Frames past a clip clamp to its last row, whose
        goal they share."""
        if self.goal_table is not None:
            if self.goal_table.shape[1] != count * GOAL_FRAME_DIM:
                raise ValueError("The bank's goal table holds another goal length.")
            return
        self._check_table_size("goal_table", count * GOAL_FRAME_DIM)
        rank, frame = self._bank_rows()
        table = torch.empty(self.num_rows, count * GOAL_FRAME_DIM, device=self.device)
        for lo in range(0, self.num_rows, chunk):
            hi = min(lo + chunk, self.num_rows)
            table[lo:hi] = self.goal(rank[lo:hi], frame[lo:hi], count)
        self.goal_table = table

    @torch.no_grad()
    def build_body_table(self, chunk: int = TABLE_CHUNK) -> None:
        """``wm.reference_body_table``: the body poses and twists ``gather`` computes by
        FK, of every row, which ``gather`` then reads."""
        if self.body_table is not None:
            return
        self._check_table_size("reference_body_table", len(self.body_names) * 18)
        rank, frame = self._bank_rows()
        parts = []
        for lo in range(0, self.num_rows, chunk):
            ref = self.gather(rank[lo : lo + chunk], frame[lo : lo + chunk])
            parts.append(
                (ref.body_pos, ref.body_rot, ref.body_lin_vel, ref.body_ang_vel)
            )
        self.body_table = tuple(torch.cat(column) for column in zip(*parts))

    # -- rollout window (``wm.reference_window``) ----------------------------------
    #: While a window is open, ``rank`` arguments are window slots (0..B-1), not bank
    #: ranks, and every read comes from ``win_states``; see ``open_window``.
    win_active: bool = False
    win_states: Tensor | None = None
    win_rows: Tensor | None = None
    win_start: Tensor | None = None
    win_length: Tensor | None = None

    def open_window(self, rank: Tensor, frame: Tensor, length: int) -> Tensor:
        """Gather the rows of ``length`` consecutive frames from ``frame`` of each clip
        ``rank`` (both (B,)) once, and serve every read until ``close_window`` from
        them: ``wm.reference_window``.

        Afterwards a read's ``rank`` is the window slot b (``torch.arange(B)``, which
        this returns) and its ``frame`` is a bank frame of that clip, at or after
        ``frame[b]``. The row a read gets is the row the bank would give: a frame past
        the clip's end clamps to its last row, which the window holds when the clip
        ends inside it, and a read is never past the window's end, so ``_index``
        keeps its formula with ``win_start[b] = b * length - frame[b]`` and
        ``win_length[b] = length[rank[b]]``. ``valid`` flags and ``frames_left`` use
        the clip length as before. ``bank_rows`` maps a window index back to its bank
        row (for tables indexed by bank row). One gather of B * ``length`` rows
        replaces a gather per step and per goal frame, which on a host-mapped bank
        read over PCIe every time. Values are the bank's, bit for bit.
        """
        if self.win_active:
            raise RuntimeError("A reference window is already open.")
        if rank.ndim != 1 or frame.shape != rank.shape or length < 1:
            raise ValueError(
                "open_window needs (B,) ranks and frames and a positive length."
            )
        frames = frame.unsqueeze(1) + torch.arange(length, device=frame.device)
        rows, _ = self._bank_index(rank, frames)  # bank rows, whatever chunk is active
        self.win_rows = rows.reshape(-1)
        self.win_states = self.states[self.win_rows]
        slots = torch.arange(rank.shape[0], device=rank.device)
        self.win_start = slots * length - frame
        self.win_length = self.length[rank]
        self.win_active = True
        return slots

    def close_window(self) -> None:
        self.win_active = False
        self.win_states = self.win_rows = self.win_start = self.win_length = None

    # -- per-env chunk (env ``library_env_window``) -------------------------------
    #: Frames past the cursor an env step reads: the goal's 10 future frames (the
    #: latent task's skill frames f+1..f+10), plus one of margin.
    ENV_LOOKAHEAD = 12
    env_win_active: bool = False
    env_win_length: int = 0
    env_win_states: Tensor | None = None  # (N * K, 71)
    env_win_rows: Tensor | None = None  # (N * K,) bank rows held
    env_win_base: Tensor | None = None  # (N,) the first frame held per env
    env_win_rank: Tensor | None = None  # (N,) the clip held per env

    def env_window_open(self, num_envs: int, length: int) -> None:
        """Hold, per env, ``length`` consecutive rows of its clip on the device: the
        env's reads of the current frame and its look-ahead (``ENV_LOOKAHEAD``) come
        from this chunk instead of the bank while ``env_win_active`` (env
        ``library_env_window``). ``env_window_refill`` (re)fills chosen envs'
        chunks from a new (clip, frame), ``env_window_advance`` refills the envs whose
        cursor got within the look-ahead of their chunk's end, so every read at or
        after the cursor is held. Reads arrive with one row per env, in env order,
        with the clips' true ranks (``_index`` maps them by position); other shapes
        read the bank. Values are the bank's, bit for bit; only chunk refills cross
        PCIe on a host-mapped bank.
        """
        if length < self.ENV_LOOKAHEAD + 2:
            raise ValueError(
                f"library_env_window needs at least {self.ENV_LOOKAHEAD + 2} frames, got {length}."
            )
        n, k = int(num_envs), int(length)
        self.env_win_length = k
        self.env_win_states = torch.zeros(
            n * k, self.states.shape[1], device=self.device
        )
        self.env_win_rows = torch.zeros(n * k, dtype=torch.long, device=self.device)
        self.env_win_base = torch.zeros(n, dtype=torch.long, device=self.device)
        self.env_win_rank = torch.zeros(n, dtype=torch.long, device=self.device)
        self.env_window_refill(
            torch.ones(n, dtype=torch.bool, device=self.device),
            self.env_win_rank,
            self.env_win_base,
        )

    def env_window_refill(self, mask: Tensor, rank: Tensor, frame: Tensor) -> None:
        """Rows ``frame .. frame + K - 1`` of clip ``rank`` into the chunks of the envs
        where ``mask`` (N,) is true; the other chunks stay. All (N,), device-side, no
        host read: capturable."""
        k = self.env_win_length
        frames = frame.unsqueeze(1) + torch.arange(k, device=frame.device)
        rows, _ = self._bank_index(rank, frames)
        rows = rows.reshape(-1)
        rowmask = mask.unsqueeze(1).expand(-1, k).reshape(-1)
        _masked_rows_into(self.env_win_states, self.states, rows, rowmask)
        torch.where(rowmask, rows, self.env_win_rows, out=self.env_win_rows)
        torch.where(mask, frame, self.env_win_base, out=self.env_win_base)
        torch.where(mask, rank, self.env_win_rank, out=self.env_win_rank)

    def env_window_advance(self, rank: Tensor, frame: Tensor) -> None:
        """Refill the envs whose cursor (``rank``, ``frame``, both (N,)) left the held
        clip or got within the look-ahead of the chunk's end."""
        need = (
            frame - self.env_win_base > self.env_win_length - 1 - self.ENV_LOOKAHEAD
        ) | (rank != self.env_win_rank)
        self.env_window_refill(need, rank, frame)

    def env_window_activate(self, on: bool) -> None:
        if on and self.env_win_length == 0:
            raise RuntimeError("env_window_activate: open the env window first.")
        self.env_win_active = bool(on)

    def bank_rows(self, index: Tensor) -> Tensor:
        """The bank row of an ``_index`` result (identity without a window)."""
        if self.win_active:
            return self.win_rows[index]
        if self.env_win_active and self._env_indexed:
            return self.env_win_rows[index]
        return index

    _env_indexed: bool = False  # the last ``_index`` call used the env chunk

    def _rows(self, index: Tensor) -> Tensor:
        if self.win_active:
            return self.win_states[index]
        if self.env_win_active and self._env_indexed:
            return self.env_win_states[index]
        return self.states[index]

    def _bank_index(self, rank: Tensor, frame: Tensor) -> tuple[Tensor, Tensor]:
        """``_index`` on the bank itself, whatever window is open."""
        while rank.ndim < frame.ndim:
            rank = rank.unsqueeze(-1)
        length = self.length[rank]
        return self.start[rank] + frame.clamp_min(0).minimum(length - 1), (
            frame >= 0
        ) & (frame < length)

    def _index(self, rank: Tensor, frame: Tensor) -> tuple[Tensor, Tensor]:
        while rank.ndim < frame.ndim:
            rank = rank.unsqueeze(-1)
        if self.win_active:
            self._env_indexed = False
            length = self.win_length[rank]
            return self.win_start[rank] + frame.clamp_min(0).minimum(length - 1), (
                frame >= 0
            ) & (frame < length)
        n = 0 if self.env_win_base is None else self.env_win_base.shape[0]
        if self.env_win_active and rank.shape[0] == n and frame.shape[0] == n:
            # One row per env in env order: the chunk index ``slot * K + (frame -
            # base)``; a frame past the clip's end clamps to its last row, which the
            # chunk holds as the bank does (``env_window_refill`` gathered it clamped).
            self._env_indexed = True
            k = self.env_win_length
            slots = torch.arange(n, device=rank.device).view(
                n, *([1] * (rank.ndim - 1))
            )
            length = self.length[rank]
            held = frame.clamp_min(0).minimum(length - 1) - self.env_win_base.view_as(
                slots
            )
            return slots * k + held.clamp(0, k - 1), (frame >= 0) & (frame < length)
        self._env_indexed = False
        return self._bank_index(rank, frame)

    def ref_state(self, rank: Tensor, frame: Tensor) -> Tensor:
        index, _ = self._index(rank, frame)
        return self._rows(index)

    state = ref_state

    def gather(self, rank: Tensor, frame: Tensor) -> G1ReferenceFrames:
        index, valid = self._index(rank, frame)
        p, quat, v, w, q, qd = unpack(self._rows(index))
        rotation = quat_wxyz_to_matrix(quat)
        if self.body_table is None:
            bp, br, bv, bw = self.kin.forward_with_velocity(
                p, rotation, q, v, w, qd, self.body_names
            )
        else:
            rows = self.bank_rows(index)
            bp, br, bv, bw = (part[rows] for part in self.body_table)
        return G1ReferenceFrames(p, rotation, v, w, q, qd, bp, br, bv, bw, valid)

    def body_index(self, names: list[str] | tuple[str, ...]) -> Tensor:
        return torch.tensor(
            [self.body_names.index(name) for name in names], device=self.device
        )

    def frames_left(self, rank: Tensor, frame: Tensor) -> Tensor:
        while rank.ndim < frame.ndim:
            rank = rank.unsqueeze(-1)
        lengths = self.win_length if self.win_active else self.length
        return lengths[rank] - 1 - frame

    def goal(self, rank: Tensor, frame: Tensor, count: int = 10) -> Tensor:
        """73 features per future frame, in the current reference heading.

        ``frame`` is not negative. With a table, ``count`` must be the table's.
        """
        if self.goal_table is not None:
            if count * GOAL_FRAME_DIM != self.goal_table.shape[1]:
                raise ValueError(f"The goal table does not hold {count} frames.")
            return self.goal_table[self.bank_rows(self._index(rank, frame)[0])]
        future = frame.unsqueeze(-1) + torch.arange(1, count + 1, device=frame.device)
        current = self.ref_state(rank, frame)
        state = self.ref_state(rank, future)
        p, quat, v, w, q, qd = unpack(state)
        rotation = quat_wxyz_to_matrix(quat)
        current_rotation = quat_wxyz_to_matrix(current[..., 3:7])
        ht = heading(current_rotation).transpose(-1, -2).unsqueeze(-3)
        valid = (self.frames_left(rank, future) >= 0).unsqueeze(-1)
        feat = torch.cat(
            (
                rotate(ht, p - current[..., None, :3]),
                matrix_to_rot6d(ht @ rotation),
                torch.where(valid, rotate(rotation.transpose(-1, -2), v), 0.0),
                torch.where(valid, rotate(rotation.transpose(-1, -2), w), 0.0),
                q,
                torch.where(valid, qd * 0.1, 0.0),
            ),
            -1,
        )
        return feat.flatten(-2)


class G1MDP:
    """Shared scalar reward/termination equations and noiseless feature views."""

    def __init__(
        self,
        library: G1ReferenceLibrary,
        default_q: Tensor,
        limits: Tensor | None = None,
        reward_kernel_scale: float = 1.0,
        reward_heading_velocity: bool = False,
    ) -> None:
        self.library = library
        self.reward_kernel_scale = float(reward_kernel_scale)
        self.reward_heading_velocity = bool(reward_heading_velocity)
        self.kin = library.kin
        self.body_names = library.body_names
        self.default_q = default_q
        hard = (
            joint_limits(default_q.device, default_q.dtype)
            if limits is None
            else limits
        )
        middle, half = hard.mean(-1), (hard[:, 1] - hard[:, 0]) * 0.45
        self.soft_lo, self.soft_hi = middle - half, middle + half
        self.noise_width = default_q.new_tensor(OBS_NOISE_WIDTH)
        self._compiled_measured_reward = None

    def proprio(
        self,
        state: Tensor,
        action: Tensor,
        context: Tensor,
        rank: Tensor,
        frame: Tensor,
        noise: bool = False,
    ) -> Tensor:
        _, quat, _, omega, q, qd = unpack(state)
        rotation = quat_wxyz_to_matrix(quat)
        ref = self.library.ref_state(rank, frame)
        obs = torch.cat(
            (
                q - self.default_q - context[..., 1:30],
                qd,
                rotate(rotation.transpose(-1, -2), omega),
                -rotation[..., 2, :],
                action,
                matrix_to_rot6d(
                    rotation.transpose(-1, -2) @ quat_wxyz_to_matrix(ref[..., 3:7])
                ),
            ),
            -1,
        )
        return obs + (2 * torch.rand_like(obs) - 1) * self.noise_width if noise else obs

    obs = proprio

    def body_states(self, state: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        p, quat, v, w, q, qd = unpack(state)
        return self.kin.forward_with_velocity(
            p, quat_wxyz_to_matrix(quat), q, v, w, qd, self.body_names
        )

    @staticmethod
    def heading_offset(rotation: Tensor, ref: G1ReferenceFrames) -> Tensor:
        """Yaw from the reference root to the robot root, (..., 1, 3, 3)."""
        return heading(rotation @ ref.root_rot.transpose(-1, -2)).unsqueeze(-3)

    @classmethod
    def reanchor(
        cls, p: Tensor, rotation: Tensor, ref: G1ReferenceFrames
    ) -> tuple[Tensor, Tensor]:
        """Reference body poses moved to the robot's root xy and heading.

        The reference root keeps its height, roll and pitch.
        """
        delta = cls.heading_offset(rotation, ref)
        origin = torch.cat((p[..., :2], ref.root_pos[..., 2:3]), -1)
        return (
            origin.unsqueeze(-2)
            + rotate(delta, ref.body_pos - ref.root_pos.unsqueeze(-2)),
            delta @ ref.body_rot,
        )

    def compile_measured_reward(self) -> None:
        """``wm.compile_eager``: the simulator's reward as one compiled function.

        Only the call with measured ``body_state`` changes: that one runs eagerly
        in collection, at one batch shape. The model's own reward stays as it is,
        inside the compiled imagination step.
        """
        if self._compiled_measured_reward is None:
            self._compiled_measured_reward = torch.compile(self._reward, dynamic=False)

    def reward(
        self,
        state: Tensor,
        action: Tensor,
        previous_action: Tensor,
        rank: Tensor,
        frame: Tensor,
        body_state: tuple[Tensor, Tensor, Tensor, Tensor] | None = None,
        ref: G1ReferenceFrames | None = None,
        contact: tuple[float, Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compare physical state with explicit TARGET frame, return dt-scaled terms.

        The simulator supplies its measured body poses/link twists through
        ``body_state``; the differentiable model uses FK from physical state.
        ``contact`` (weight, foot body slots, reference contact at the target
        frame (..., feet)) adds the model-only foot-contact term
        (``wm.contact_reward``, ``g1_contact``) on the same FK body states.
        """
        if contact is not None and body_state is not None:
            raise ValueError(
                "The contact term belongs to the model's reward, not the simulator's."
            )
        compiled = self._compiled_measured_reward
        fn = self._reward if body_state is None or compiled is None else compiled
        if contact is None:
            return fn(state, action, previous_action, rank, frame, body_state, ref)
        return fn(state, action, previous_action, rank, frame, body_state, ref, contact)

    def _reward(
        self,
        state: Tensor,
        action: Tensor,
        previous_action: Tensor,
        rank: Tensor,
        frame: Tensor,
        body_state: tuple[Tensor, Tensor, Tensor, Tensor] | None,
        ref: G1ReferenceFrames | None,
        contact: tuple[float, Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        ref = self.library.gather(rank, frame) if ref is None else ref
        bp, br, bv, bw = self.body_states(state) if body_state is None else body_state
        p, quat, _, _, q, _ = unpack(state)
        rotation = quat_wxyz_to_matrix(quat)
        target_pos, target_rot = self.reanchor(p, rotation, ref)
        target_lin_vel, target_ang_vel = ref.body_lin_vel, ref.body_ang_vel
        if self.reward_heading_velocity:
            # Velocity targets turn with the same heading offset as the pose
            # targets, so a yaw offset is scored once, by root_ori, and not
            # again by the two velocity terms (upstream SuperMimic 5fb9229 and
            # mjlab#1274).
            delta = self.heading_offset(rotation, ref)
            target_lin_vel = rotate(delta, target_lin_vel)
            target_ang_vel = rotate(delta, target_ang_vel)
        k = self.reward_kernel_scale
        terms = {
            "body_pos": torch.exp(
                -(target_pos - bp).square().sum(-1).mean(-1) / (0.3 * k) ** 2
            ),
            "body_ori": torch.exp(
                -rotation_error_sq(target_rot, br).mean(-1) / (0.4 * k) ** 2
            ),
            "body_lin_vel": torch.exp(
                -(target_lin_vel - bv).square().sum(-1).mean(-1) / k**2
            ),
            "body_ang_vel": torch.exp(
                -(target_ang_vel - bw).square().sum(-1).mean(-1) / (3.14 * k) ** 2
            ),
            "root_pos": torch.exp(
                -(ref.root_pos - p).square().sum(-1) / (0.3 * k) ** 2
            ),
            "root_ori": torch.exp(
                -rotation_error_sq(ref.root_rot, rotation) / (0.4 * k) ** 2
            ),
            "action_rate": (action - previous_action).square().sum(-1),
            "joint_limit": (
                torch.relu(q - self.soft_hi) + torch.relu(self.soft_lo - q)
            ).sum(-1),
        }
        parts = {
            name: value * REWARD_WEIGHTS[name] * CONTROL_DT
            for name, value in terms.items()
        }
        total = torch.stack(tuple(parts.values()), -1).sum(-1)
        if contact is not None:
            weight, feet, reference = contact
            parts["contact"] = (
                weight
                * contact_term(bp[..., feet, :], bv[..., feet, :], reference)
                * CONTROL_DT
            )
            total = total + parts["contact"]
        return total, parts

    def termination(
        self, state: Tensor, rank: Tensor, frame: Tensor, hard: bool = False
    ) -> Tensor:
        ref = self.library.ref_state(rank, frame)
        ez = (ref[..., 2] - state[..., 2]).abs()
        er = (
            quat_wxyz_to_matrix(ref[..., 3:7])[..., 2, 2]
            - quat_wxyz_to_matrix(state[..., 3:7])[..., 2, 2]
        ).abs()
        if hard:
            return (ez > 0.25) | (er > 0.8)
        return 1 - (1 - torch.sigmoid((ez - 0.25) / 0.02)) * (
            1 - torch.sigmoid((er - 0.8) / 0.04)
        )

    def critic_features(
        self, state: Tensor, context: Tensor, rank: Tensor, frame: Tensor
    ) -> Tensor:
        ref = self.library.gather(rank, frame)
        p, quat, _, _, q, _ = unpack(state)
        rotation = quat_wxyz_to_matrix(quat)
        bp = self.kin(p, rotation, q, self.body_names)[0]
        target, _ = self.reanchor(p, rotation, ref)
        ht = heading(rotation).transpose(-1, -2)
        return torch.cat(
            (
                invariant_features(state),
                ref.joint_pos - q,
                rotate(ht.unsqueeze(-3), target - bp).flatten(-2),
                rotate(ht, ref.root_pos - p),
                matrix_to_rot6d(rotation.transpose(-1, -2) @ ref.root_rot),
                normalize_context(context),
            ),
            -1,
        )


def _masked_rows_into(dst: Tensor, src: Tensor, rows: Tensor, mask: Tensor) -> None:
    """``dst[i] = src[rows[i]]`` where ``mask[i]``, other rows untouched. On CUDA the
    masked Triton gather (only masked rows are read, so only they cross PCIe from a
    host-mapped ``src``); elsewhere a full gather and a select."""
    if dst.is_cuda:
        try:
            from imitationlab.envs.isaaclab.envs.graph_step.reference_frame import (
                masked_row_gather_,
            )
        except ImportError:  # the pure-Python environment
            masked_row_gather_ = None
        if masked_row_gather_ is not None:
            masked_row_gather_(dst, src, rows, mask)
            return
    torch.where(mask.unsqueeze(-1), src[rows], dst, out=dst)


def pack_reference_rows(
    library: G1ReferenceLibrary,
    rank: Tensor,
    frame: Tensor,
    *,
    field_order: tuple[str, ...],
    field_shapes: dict[str, tuple[int, ...]],
    out: Tensor,
) -> Tensor:
    """The reference arrays' runtime row (``qpos``, ``qvel``, ``body_pos_w``,
    ``body_quat_w``, ``body_lin_vel_w``, ``body_ang_vel_w`` in ``field_order``, packed
    as the data plane packs them) of frame ``frame`` of clip ``rank`` (both (N,)),
    derived from the library's 71-float rows: the root pose, velocities and joints are
    the row's, the tracked bodies come from the library's forward kinematics (the
    reward's reference bodies), their quaternions scalar-first as in the arrays.
    Written into ``out`` (N, width) in place and returned. Velocities are the
    library's central differences, not the dataset's own, so the row equals the
    arrays' only in ``qpos``.
    """
    index, _ = library._index(rank, frame)
    p, quat, v, w, q, qd = unpack(library._rows(index))
    if library.body_table is None:
        bp, br, bv, bw = library.kin.forward_with_velocity(
            p, quat_wxyz_to_matrix(quat), q, v, w, qd, library.body_names
        )
    else:
        rows = library.bank_rows(index)
        bp, br, bv, bw = (part[rows] for part in library.body_table)
    parts = {
        "qpos": torch.cat((p, quat, q), -1),
        "qvel": torch.cat((v, w, qd), -1),
        "body_pos_w": bp,
        "body_quat_w": mat_to_quat(br),
        "body_lin_vel_w": bv,
        "body_ang_vel_w": bw,
    }
    n = int(rank.shape[0])
    width = sum(math.prod(tuple(field_shapes[name])) for name in field_order)
    if tuple(out.shape) != (n, width):
        raise ValueError(
            f"pack_reference_rows: the layout has width {width} for {n} rows, out is {tuple(out.shape)}."
        )
    for name in field_order:
        shape = tuple(field_shapes[name])
        if tuple(parts[name].shape[1:]) != shape:
            raise ValueError(
                f"pack_reference_rows: {name} is {tuple(parts[name].shape[1:])}, the layout wants {shape}."
            )
    offset = 0
    for name in field_order:
        size = math.prod(tuple(field_shapes[name]))
        out[:, offset : offset + size].copy_(parts[name].reshape(n, size))
        offset += size
    return out
