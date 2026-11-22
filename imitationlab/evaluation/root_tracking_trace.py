"""Per-step, every-environment root tracking traces for a scored board.

A board JSON keeps one number per episode. Global (world-frame) tracking needs
the error as a function of time and the root paths themselves, for our tracker
and for SONIC under one definition, so both evaluators can write this trace
beside their JSON (``--root-trace``).

Definitions, identical to the evaluators' per-environment metrics:

* robot root: the articulation root link pose in the world; reference root:
  ``_get_reference_root_state_w_fast`` -- the pair that
  ``root_pos_xyz_error_m`` is computed from;
* MPJPE-L / MPJPE-G: ``mpjpe_local_global`` over the given tracked links.

Positions are stored relative to each environment's origin, quaternions in the
simulator's order (xyzw in Isaac Lab 3).

Row convention: row 0 is the reset state before the first action and row
``t >= 1`` is the state returned by step ``t``. ``active[t, e]`` is True where
row ``t`` belongs to environment ``e``'s first, still-running episode: the
evaluators' metric mask, because a finishing environment's returned state is
already its next reset. Row 0 carries no error metric in the evaluators, so
``summaries`` skips it. Every per-step tensor stays on the device until
``write``; recording adds no host synchronisation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROW_FIELDS = (
    "robot_x",
    "robot_y",
    "robot_z",
    "robot_qx",
    "robot_qy",
    "robot_qz",
    "robot_qw",
    "ref_x",
    "ref_y",
    "ref_z",
    "ref_qx",
    "ref_qy",
    "ref_qz",
    "ref_qw",
    "mpjpe_l_m",
    "mpjpe_g_m",
)
_FIELD = {name: index for index, name in enumerate(ROW_FIELDS)}


def _tensor(value: Any) -> torch.Tensor:
    value = value.torch if hasattr(value, "torch") else value
    return torch.as_tensor(value)


class RootTrackingTrace:
    """Accumulate root poses and MPJPE for every environment at every step."""

    def __init__(
        self,
        *,
        env_origins: torch.Tensor,
        trajectory_ranks: list[int],
        motion_names: list[str],
        step_dt: float,
        provenance: dict[str, Any],
        joint_names: list[str] | None = None,
    ) -> None:
        self._origins = _tensor(env_origins).detach().float().clone()
        num_envs = int(self._origins.shape[0])
        if len(trajectory_ranks) != num_envs or len(motion_names) != num_envs:
            raise ValueError(
                "Root trace needs one rank and one motion name per environment, "
                f"got {len(trajectory_ranks)} ranks and {len(motion_names)} names "
                f"for {num_envs} environments."
            )
        self.num_envs = num_envs
        self.trajectory_ranks = [int(rank) for rank in trajectory_ranks]
        self.motion_names = [str(name) for name in motion_names]
        self.step_dt = float(step_dt)
        self.provenance = dict(provenance)
        self.joint_names = None if joint_names is None else list(joint_names)
        self._rows: list[torch.Tensor] = []
        self._active: list[torch.Tensor] = []
        self._local_step: list[torch.Tensor] = []
        self._joints: list[torch.Tensor] = []

    def append(
        self,
        *,
        robot_pose_w: torch.Tensor,
        reference_pose_w: torch.Tensor,
        mpjpe_local_m: torch.Tensor,
        mpjpe_global_m: torch.Tensor,
        active: torch.Tensor,
        local_step: torch.Tensor,
        joint_pos: torch.Tensor | None = None,
        reference_joint_pos: torch.Tensor | None = None,
    ) -> None:
        """Store one row. Poses are ``[N, 7]`` world xyz + quaternion."""

        origins = self._origins.to(robot_pose_w.device)
        robot = robot_pose_w.detach().float().clone()
        reference = reference_pose_w.detach().float().clone()
        robot[:, :3] -= origins
        reference[:, :3] -= origins
        row = torch.cat(
            [
                robot,
                reference,
                mpjpe_local_m.detach().float().reshape(-1, 1),
                mpjpe_global_m.detach().float().reshape(-1, 1),
            ],
            dim=-1,
        )
        if tuple(row.shape) != (self.num_envs, len(ROW_FIELDS)):
            raise ValueError(f"Root trace row has shape {tuple(row.shape)}.")
        self._rows.append(row)
        self._active.append(active.detach().to(row.device).bool().reshape(-1).clone())
        self._local_step.append(
            local_step.detach().to(row.device).to(torch.int32).reshape(-1).clone()
        )
        if (joint_pos is None) != (reference_joint_pos is None):
            raise ValueError("Pass both robot and reference joints, or neither.")
        if joint_pos is not None:
            if self.joint_names is None:
                raise ValueError("Joint rows need joint_names at construction.")
            self._joints.append(
                torch.stack(
                    [
                        joint_pos.detach().float(),
                        reference_joint_pos.detach().to(joint_pos.device).float(),
                    ],
                    dim=1,
                ).clone()
            )
        elif self._joints:
            raise ValueError("Joint rows must be recorded on every step or none.")

    def capture(
        self,
        base_env: Any,
        *,
        active: torch.Tensor,
        tracked_body_names: list[str],
        record_joints: bool = False,
    ) -> None:
        """Read the environment's current state and store it as one row."""

        from imitationlab.envs.isaaclab.contracts.tracking_metrics import (
            mpjpe_local_global,
        )

        from imitationlab.evaluation.closed_loop_metrics import (
            _body_tracking_tensors,
        )

        robot = base_env.robot.data
        root_pose = _tensor(robot.root_link_pose_w)
        ref_pos, ref_quat, _, _ = base_env._get_reference_root_state_w_fast()
        bodies = _body_tracking_tensors(base_env, list(tracked_body_names))
        if bodies is None:
            raise ValueError("Root trace needs at least one tracked body.")
        local, global_ = mpjpe_local_global(
            bodies["actual_pos"],
            _tensor(robot.root_pos_w),
            bodies["ref_pos"],
            ref_pos,
        )
        joints = reference_joints = None
        if record_joints:
            joints = _tensor(robot.joint_pos)
            defaults = _tensor(robot.default_joint_pos)
            reference = base_env.current_expert_frame["joint_pos"]
            reference_joints = torch.where(torch.isnan(reference), defaults, reference)
        self.append(
            robot_pose_w=root_pose,
            reference_pose_w=torch.cat([ref_pos, ref_quat], dim=-1),
            mpjpe_local_m=local,
            mpjpe_global_m=global_,
            active=active,
            local_step=base_env._current_reference_local_step,
            joint_pos=joints,
            reference_joint_pos=reference_joints,
        )

    def arrays(self) -> dict[str, np.ndarray]:
        if not self._rows:
            raise ValueError("Root trace has no rows.")
        out = {
            "rows": torch.stack(self._rows).cpu().numpy(),
            "active": torch.stack(self._active).cpu().numpy(),
            "local_step": torch.stack(self._local_step).cpu().numpy(),
            "row_fields": np.asarray(ROW_FIELDS),
            "trajectory_ranks": np.asarray(self.trajectory_ranks, dtype=np.int64),
            "motion_names": np.asarray(self.motion_names),
            "step_dt": np.asarray(self.step_dt),
            "provenance": np.asarray(json.dumps(self.provenance)),
        }
        if self._joints:
            out["joints"] = torch.stack(self._joints).cpu().numpy()
            out["joint_names"] = np.asarray(self.joint_names)
        return out

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = self.arrays()
        np.savez_compressed(path, **arrays)
        rows = arrays["rows"]
        print(
            f"[ROOT TRACE] {path}: {rows.shape[0]} rows x {rows.shape[1]} envs"
            + (" + joints" if "joints" in arrays else ""),
            flush=True,
        )
        return path


def load_root_trace(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        trace = {key: data[key] for key in data.files}
    trace["provenance"] = json.loads(str(trace["provenance"]))
    trace["step_dt"] = float(trace["step_dt"])
    return trace


def field(trace: dict[str, Any], name: str) -> np.ndarray:
    """One ``[rows, envs]`` column of the trace by its ROW_FIELDS name."""

    return trace["rows"][..., _FIELD[name]]


def root_error(trace: dict[str, Any], *, planar: bool = False) -> np.ndarray:
    """World-frame root position error ``[rows, envs]`` in metres."""

    dims = ("x", "y") if planar else ("x", "y", "z")
    delta = np.stack(
        [field(trace, f"robot_{axis}") - field(trace, f"ref_{axis}") for axis in dims],
        axis=-1,
    )
    return np.linalg.norm(delta, axis=-1)


def episode_rows(trace: dict[str, Any], env: int) -> np.ndarray:
    """Row indices of environment ``env``'s scored episode, reset row included.

    The reset row 0 always belongs to the first episode; scored rows follow
    while ``active`` holds. The indices are consecutive by construction.
    """

    active = trace["active"][:, env]
    rows = [0]
    for index in range(1, active.shape[0]):
        if not active[index]:
            break
        rows.append(index)
    return np.asarray(rows, dtype=np.int64)


def summaries(trace: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-environment means over scored rows (row 0 excluded), in mm."""

    drift = root_error(trace)
    planar = root_error(trace, planar=True)
    local = field(trace, "mpjpe_l_m")
    global_ = field(trace, "mpjpe_g_m")
    active = trace["active"].copy()
    active[0] = False
    out = []
    for env in range(active.shape[1]):
        mask = active[:, env]
        count = int(mask.sum())
        rows = episode_rows(trace, env)

        def _mean(values: np.ndarray, mask: np.ndarray = mask) -> float:
            return float(values[mask, env].mean() * 1000.0) if count else float("nan")

        out.append(
            {
                "env": env,
                "trajectory_rank": int(trace["trajectory_ranks"][env]),
                "motion_name": str(trace["motion_names"][env]),
                "scored_rows": count,
                "mpjpe_l_mm": _mean(local),
                "mpjpe_g_mm": _mean(global_),
                "root_xyz_error_mm": _mean(drift),
                "root_xy_error_mm": _mean(planar),
                "root_xyz_error_final_mm": float(drift[rows[-1], env] * 1000.0)
                if count
                else float("nan"),
            }
        )
    return out


def take_arrays(
    trace: dict[str, Any], env: int, *, reference: bool = False
) -> dict[str, Any]:
    """One environment's episode as a renderer take (``--takes``) payload.

    The same ``root_pose``/``joint_pos``/``joint_names`` layout TrackingTake
    writes, in environment-local coordinates (origin already removed), so a
    single-environment replay puts the robot where the rollout had it.
    """

    if "joints" not in trace:
        raise ValueError("This root trace was recorded without joints.")
    rows = episode_rows(trace, env)
    prefix = "ref" if reference else "robot"
    names = ("x", "y", "z", "qx", "qy", "qz", "qw")
    root_pose = np.stack(
        [field(trace, f"{prefix}_{name}")[rows, env] for name in names], axis=-1
    )
    joint_pos = trace["joints"][rows, env, 1 if reference else 0]
    steps = trace["local_step"][rows, env].astype(np.int64)
    if np.any(np.diff(steps) != 1):
        raise ValueError(f"Environment {env} reference frames are not consecutive.")
    provenance = {
        **trace["provenance"],
        "trajectory_rank": int(trace["trajectory_ranks"][env]),
        "motion_name": str(trace["motion_names"][env]),
        "reference": bool(reference),
        "step_dt": trace["step_dt"],
        "env_origin": [0.0, 0.0, 0.0],
        "coordinates": "environment-local simulation world",
        "source": "root_tracking_trace",
    }
    return {
        "root_pose": root_pose.astype(np.float32),
        "joint_pos": joint_pos.astype(np.float32),
        "joint_names": np.asarray(trace["joint_names"]),
        "local_step": steps,
        "time_s": steps * trace["step_dt"],
        "provenance": json.dumps(provenance),
    }


__all__ = [
    "ROW_FIELDS",
    "RootTrackingTrace",
    "episode_rows",
    "field",
    "load_root_trace",
    "root_error",
    "summaries",
    "take_arrays",
]
