"""ImitationRLEnv: the Isaac Lab manager-based base class of the G1 tracking env.

:class:`~imitationlab.envs.isaaclab.envs.g1_env.G1RLEnv` (``imitation-g1-v1``)
subclasses it and owns the control step and the episode resets. This class
composes the shared components the step builds on:

- :class:`~imitationlab.envs.isaaclab.envs.expert_data_plane.ExpertDataPlane`:
  dataset load, reference metadata / joint alignment, the MDP fast paths,
  frame refresh, alignment transforms, expert window / goal building, expert
  batch / macro-transition sampling (encoder pretraining), and the MPJPE
  metric computation.
- the two command terms of the declared command interface
  (``tasks/manager_based/imitation/command_interface.py``): the ``reference``
  channel (selection, tracking metrics) and the explicit ``actor`` channel,
  which own their own buffers.

The env keeps every public name the mdp funcs, the command terms, the TorchRL
wrapper and the evaluation tools call (``trajectory_manager``,
``current_expert_frame``, the ``_get_*_fast`` accessors, the expert samplers,
the reference replay targets, ...) as thin delegators.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from imitationlab.envs.isaaclab.contracts.command_channels import (
    ACTOR_TERM_NAME,
    REFERENCE_TERM_NAME,
)
from imitationlab.envs.isaaclab.contracts.training_log import (
    REQUIRED_ENV_LOG_ENTRIES,
)
from imitationlab.envs.isaaclab.contracts.causal_planner_observation import (
    CAUSAL_PLANNER_FRAME_DIM,
    CausalPlannerHistory,
    build_causal_planner_frame,
    causal_planner_observation_spec,
)
from tensordict import TensorDict

from .collect_fastpath import fastpath_enabled
from .expert_data_plane import ExpertDataPlane
from .imitation_interface import ImitationEnvInterface

logger = logging.getLogger(__name__)

_CAUSAL_PLANNER_HISTORY_STEPS = 9
_FAST_TERMINAL_GATHER = fastpath_enabled("terminal_gather")


def _apply_robot_self_collisions(cfg: Any) -> None:
    """Write ``cfg.robot_self_collisions`` into the robot spawn's articulation root.

    Runs before the scene is built (CLI overrides land after the config's own
    ``__post_init__``). ``None`` leaves the asset's authored value.
    """
    value = getattr(cfg, "robot_self_collisions", None)
    if value is None:
        return
    from isaaclab.sim.schemas import ArticulationRootPropertiesCfg  # noqa: PLC0415

    spawn = cfg.scene.robot.spawn
    props = getattr(spawn, "articulation_props", None)
    props = (
        ArticulationRootPropertiesCfg(enabled_self_collisions=bool(value))
        if props is None
        else props.replace(enabled_self_collisions=bool(value))
    )
    cfg.scene.robot.spawn = spawn.replace(articulation_props=props)


def _apply_robot_collision_filter_pairs(cfg: Any) -> None:
    """Filter ``cfg.robot_collision_filter_pairs`` (robot body-name pairs) at spawn.

    Wraps the robot's spawn function: after the robot prim exists, each pair is
    written as a UsdPhysics ``physics:filteredPairs`` relationship between the two
    body prims, which Newton's importer turns into shape collision filter pairs of
    the prototype and so of every replicated world.
    """
    pairs = getattr(cfg, "robot_collision_filter_pairs", None)
    if not pairs:
        return
    pairs = [_body_pair(pair) for pair in pairs]
    spawn = cfg.scene.robot.spawn
    spawn_robot = spawn.func

    def spawn_with_filtered_pairs(prim_path, spawn_cfg, *args, **kwargs):
        prim = spawn_robot(prim_path, spawn_cfg, *args, **kwargs)
        author_filtered_body_pairs(prim, pairs)
        return prim

    cfg.scene.robot.spawn = spawn.replace(func=spawn_with_filtered_pairs)


def _body_pair(pair: Any) -> tuple[str, str]:
    """A body pair from ``["a", "b"]`` or the CLI-friendly ``"a:b"``."""
    parts = pair.split(":") if isinstance(pair, str) else list(pair)
    if len(parts) != 2:
        raise ValueError(
            f"robot_collision_filter_pairs: expected a body pair, got {pair!r}"
        )
    return str(parts[0]).strip(), str(parts[1]).strip()


def author_filtered_body_pairs(root: Any, pairs: list[tuple[str, str]]) -> None:
    """Write ``physics:filteredPairs`` between named rigid bodies under ``root``."""
    from pxr import Usd, UsdPhysics  # noqa: PLC0415

    names = {name for pair in pairs for name in pair}
    bodies = {}
    for prim in Usd.PrimRange(root):
        if prim.GetName() in names and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            bodies.setdefault(prim.GetName(), prim)
    missing = sorted(names - bodies.keys())
    if missing:
        raise ValueError(
            f"robot_collision_filter_pairs: no rigid body named {missing} under {root.GetPath()}"
        )
    for first, second in pairs:
        rel = UsdPhysics.FilteredPairsAPI.Apply(bodies[first]).CreateFilteredPairsRel()
        rel.AddTarget(bodies[second].GetPath())


def _apply_robot_collision_primitives(cfg: Any) -> None:
    """With ``cfg.robot_collision_primitives``, swap mesh colliders for capsules at spawn."""
    if not getattr(cfg, "robot_collision_primitives", False):
        return
    spawn = cfg.scene.robot.spawn
    spawn_robot = spawn.func

    def spawn_with_capsules(prim_path, spawn_cfg, *args, **kwargs):
        prim = spawn_robot(prim_path, spawn_cfg, *args, **kwargs)
        replace_mesh_colliders_with_capsules(prim)
        return prim

    cfg.scene.robot.spawn = spawn.replace(func=spawn_with_capsules)


def replace_mesh_colliders_with_capsules(root: Any) -> int:
    """Give every collision ``Mesh`` under ``root`` a bounding-capsule twin.

    The capsule is fitted in the mesh's parent frame: its axis is the principal
    axis of the mesh points, its radius the largest distance of a point from that
    axis, and its length the points' extent along it, so the capsule contains the
    mesh. The capsule copies the mesh's collision and physics-material bindings;
    the mesh keeps rendering with collision disabled. Instanced collision prims
    are de-instanced first (instance proxies cannot be edited). Returns the
    number of meshes replaced.
    """
    import numpy as np  # noqa: PLC0415
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: PLC0415

    pred = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    meshes = [
        prim
        for prim in Usd.PrimRange(root, pred)
        if prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    stage = root.GetStage()
    for mesh in meshes:
        node = mesh
        while node and node != root:
            if node.IsInstanceable():
                node.SetInstanceable(False)
            node = node.GetParent()
    replaced = 0
    for mesh_proxy in meshes:
        mesh = stage.GetPrimAtPath(mesh_proxy.GetPath())
        points = np.asarray(UsdGeom.Mesh(mesh).GetPointsAttr().Get(), dtype=np.float64)
        if points.size == 0:
            continue
        local = UsdGeom.Xformable(mesh).GetLocalTransformation()
        m = np.array(local, dtype=np.float64)  # row-vector convention
        pts = points @ m[:3, :3] + m[3, :3]
        center = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - center, full_matrices=False)
        axis = vt[0] / np.linalg.norm(vt[0])
        along = (pts - center) @ axis
        radial = np.linalg.norm((pts - center) - np.outer(along, axis), axis=1)
        radius = float(radial.max())
        lo, hi = float(along.min()), float(along.max())
        mid = center + axis * (lo + hi) / 2.0
        height = max(0.0, (hi - lo) - 2.0 * radius)
        capsule = UsdGeom.Capsule.Define(
            stage, mesh.GetParent().GetPath().AppendChild(f"{mesh.GetName()}_capsule")
        )
        capsule.CreateAxisAttr("Z")
        capsule.CreateRadiusAttr(radius)
        capsule.CreateHeightAttr(height)
        rotation = Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), Gf.Vec3d(*axis.tolist()))
        xform = UsdGeom.Xformable(capsule)
        xform.AddTranslateOp().Set(Gf.Vec3d(*mid.tolist()))
        xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Quatd(rotation.GetQuat())
        )
        UsdGeom.Imageable(capsule).CreatePurposeAttr(UsdGeom.Tokens.guide)
        UsdPhysics.CollisionAPI.Apply(capsule.GetPrim())
        binding = (
            UsdShade.MaterialBindingAPI(mesh).GetDirectBinding("physics").GetMaterial()
        )
        if binding:
            UsdShade.MaterialBindingAPI.Apply(capsule.GetPrim()).Bind(
                binding, UsdShade.Tokens.weakerThanDescendants, "physics"
            )
        UsdPhysics.CollisionAPI(mesh).CreateCollisionEnabledAttr(False)
        replaced += 1
    return replaced


class ImitationRLEnv(ManagerBasedRLEnv):
    """Reference-tracking env composing the ExpertDataPlane and command planes.

    Its configuration is :class:`~imitationlab.envs.isaaclab.tasks.manager_based.imitation.config.g1.imitation_g1_env.ImitationG1EnvCfg`.
    """

    # The entries every recipe of this env publishes in ``extras["log"]``.
    required_log_entries: tuple[str, ...] = REQUIRED_ENV_LOG_ENTRIES
    # Which termination terms are failures; resolved at the first reset.
    _tracking_failure_terms: torch.Tensor | None = None
    # Control-step implementation (`cfg.step_mode`, set in `__init__`).
    _step_mode: str = "eager"

    def __init__(self, cfg: Any, render_mode: str | None = None, **kwargs: Any) -> None:
        """Initialize the env: planes first, then the base managers."""
        # Get device
        device = torch.device(cfg.sim.device)
        num_envs = int(cfg.scene.num_envs)

        # Isaac Lab 3.0 applies `env.*` CLI overrides and preset selections
        # after the config is constructed, so this is the first moment the
        # config's field values are final. The config owns exactly one
        # resolution step, and it needs nothing from here: every derived value
        # is a function of the fields it can already see.
        cfg.resolve()

        # The declared command interface is the whole command surface. The
        # environment derives only what its data plane and reset path need;
        # the terms read the interface themselves.
        command_interface = cfg.command_interface
        reference_channel = command_interface.reference
        self._reference_start_frame = int(reference_channel.selection.start_frame)
        self._command_ee_body_names = tuple(
            str(name) for name in reference_channel.ee_body_names
        )
        self._command_keypoint_body_names = tuple(
            str(name) for name in reference_channel.keypoint_body_names
        )
        # Window the offline expert-batch mapper serves policy command keys at:
        # the skill encoder's view when there is one, else the actor's own.
        past_steps, future_steps, frame_stride = command_interface.expert_batch_window()
        self._latent_patch_past_steps = int(past_steps)
        self._latent_patch_future_steps = int(future_steps)
        self._latent_patch_frame_stride = int(frame_stride)
        self._latent_goal_steps = int(getattr(cfg, "latent_goal_steps", 0))
        if self._latent_goal_steps < 0:
            raise ValueError("latent_goal_steps must be >= 0.")

        # Phase 1 of the data plane: dataset load, trajectory manager, first
        # reference frame. Runs before `super().__init__` (which builds the
        # managers).
        self.expert_data_plane = ExpertDataPlane(cfg, self)

        self._reference_replay_targets_enabled = False
        self._reference_replay_source_env_ids: torch.Tensor | None = None
        self._reference_replay_target_env_ids: torch.Tensor | None = None
        self._last_tracked_root_pos_w = torch.zeros((num_envs, 3), device=device)
        self._last_tracked_root_pos_valid = torch.zeros(
            (num_envs,), device=device, dtype=torch.bool
        )
        step_mode = str(getattr(cfg, "step_mode", "eager"))
        if step_mode not in ("eager", "masked", "graph"):
            raise ValueError(
                f"step_mode must be 'eager', 'masked' or 'graph', got {step_mode!r}."
            )
        self._step_mode = step_mode
        self._masked_stepper = None
        _apply_robot_self_collisions(cfg)
        _apply_robot_collision_filter_pairs(cfg)
        _apply_robot_collision_primitives(cfg)

        # Initialize parent class (builds the managers; the `motion` term
        # constructs its owned reset-start samplers inside `load_managers`).
        super().__init__(cfg, render_mode, **kwargs)

        self.robot: Articulation = self.scene["robot"]
        # Phase 2 of the data plane: retarget the trajectory manager, build
        # the fast paths against the live scene, resolve the metric bodies.
        self.expert_data_plane.finalize(self.scene, self.robot)
        self._initialize_causal_planner_history()

        # Reference selection, reset-start sampling, and the adaptive-failure
        # bookkeeping live on the reference channel; the environment only calls
        # them at the timing-critical points below.
        self._reference_term = self.reference_command

    # ------------------------------------------------------------------
    # ExpertDataPlane delegators.
    # ------------------------------------------------------------------

    @property
    def trajectory_manager(self):
        return self.expert_data_plane.trajectory_manager

    @property
    def current_expert_frame(self) -> TensorDict | None:
        return self.expert_data_plane.current_expert_frame

    @property
    def _current_reference_local_step(self) -> torch.Tensor:
        return self.expert_data_plane._current_reference_local_step

    @property
    def reference_joint_names(self) -> list[str]:
        return self.expert_data_plane.reference_joint_names

    @property
    def reference_body_names(self) -> list[str]:
        return self.expert_data_plane.reference_body_names

    @property
    def reference_site_names(self) -> list[str]:
        return self.expert_data_plane.reference_site_names

    @property
    def _mpjpe_metric_body_names(self) -> list[str]:
        return self.expert_data_plane._mpjpe_metric_body_names

    @property
    def _mdp_reset_pose_bounds(self) -> torch.Tensor | None:
        return self.expert_data_plane._mdp_reset_pose_bounds

    @_mdp_reset_pose_bounds.setter
    def _mdp_reset_pose_bounds(self, value: torch.Tensor | None) -> None:
        self.expert_data_plane._mdp_reset_pose_bounds = value

    @property
    def _mdp_reset_velocity_bounds(self) -> torch.Tensor | None:
        return self.expert_data_plane._mdp_reset_velocity_bounds

    @_mdp_reset_velocity_bounds.setter
    def _mdp_reset_velocity_bounds(self, value: torch.Tensor | None) -> None:
        self.expert_data_plane._mdp_reset_velocity_bounds = value

    @property
    def _mdp_reset_root_pose_source(self) -> str:
        return self.expert_data_plane._mdp_reset_root_pose_source

    @property
    def _mdp_reset_root_velocity_source(self) -> str:
        return self.expert_data_plane._mdp_reset_root_velocity_source

    def _invalidate_mdp_cache(self) -> None:
        self.expert_data_plane._invalidate_mdp_cache()

    def _get_reference_alignment_fast(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.expert_data_plane._get_reference_alignment_fast()

    def _get_body_ids_tensor_fast(
        self, body_ids: Sequence[int] | torch.Tensor | slice
    ) -> torch.Tensor | slice:
        return self.expert_data_plane._get_body_ids_tensor_fast(body_ids)

    def _get_joint_ids_tensor_fast(
        self, joint_ids: Sequence[int] | torch.Tensor | slice
    ) -> torch.Tensor | slice:
        return self.expert_data_plane._get_joint_ids_tensor_fast(joint_ids)

    def _get_reference_root_state_w_fast(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.expert_data_plane._get_reference_root_state_w_fast()

    def _get_reference_cvel_fast(self) -> torch.Tensor:
        return self.expert_data_plane._get_reference_cvel_fast()

    def _get_reference_body_ids_fast(
        self, reference_body_names: Sequence[str]
    ) -> torch.Tensor:
        return self.expert_data_plane._get_reference_body_ids_fast(reference_body_names)

    def _get_robot_body_ids_by_name_fast(
        self, body_names: Sequence[str]
    ) -> torch.Tensor:
        return self.expert_data_plane._get_robot_body_ids_by_name_fast(body_names)

    def _get_reference_body_pose_w_fast(
        self, reference_body_names: Sequence[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.expert_data_plane._get_reference_body_pose_w_fast(
            reference_body_names
        )

    def _get_reference_body_velocity_w_fast(
        self, reference_body_names: Sequence[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.expert_data_plane._get_reference_body_velocity_w_fast(
            reference_body_names
        )

    def _get_robot_anchor_body_id_fast(self, anchor_body_name: str) -> int:
        return self.expert_data_plane._get_robot_anchor_body_id_fast(anchor_body_name)

    def _get_robot_anchor_state_w_fast(
        self, anchor_body_name: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.expert_data_plane._get_robot_anchor_state_w_fast(anchor_body_name)

    def _get_robot_body_pose_w_fast(
        self, body_ids: Sequence[int] | torch.Tensor | slice
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.expert_data_plane._get_robot_body_pose_w_fast(body_ids)

    def _get_robot_body_velocity_w_fast(
        self, body_ids: Sequence[int] | torch.Tensor | slice
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.expert_data_plane._get_robot_body_velocity_w_fast(body_ids)

    def _get_robot_body_state_in_anchor_frame_fast(
        self,
        body_ids: Sequence[int] | torch.Tensor | slice,
        anchor_body_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.expert_data_plane._get_robot_body_state_in_anchor_frame_fast(
            body_ids, anchor_body_name
        )

    def get_expert_motion_qpos_command(
        self, joint_ids: Sequence[int] | slice = slice(None)
    ) -> torch.Tensor:
        return self.expert_data_plane.get_expert_motion_qpos_command(joint_ids)

    def _get_expert_motion_command_fast(
        self, joint_ids: Sequence[int] | torch.Tensor | slice
    ) -> torch.Tensor:
        return self.expert_data_plane._get_expert_motion_command_fast(joint_ids)

    def _get_reference_alignment_transform(
        self, env_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.expert_data_plane._get_reference_alignment_transform(env_ids)

    def _transform_reference_pose_to_world(
        self,
        ref_pos: torch.Tensor,
        ref_quat: torch.Tensor | None = None,
        env_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.expert_data_plane._transform_reference_pose_to_world(
            ref_pos, ref_quat, env_ids=env_ids
        )

    def _transform_reference_body_pose_to_init_alignment(
        self,
        ref_pos: torch.Tensor,
        ref_quat: torch.Tensor | None = None,
        env_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.expert_data_plane._transform_reference_body_pose_to_init_alignment(
            ref_pos, ref_quat, env_ids=env_ids
        )

    def _index_copy_reference_rows_(
        self, dst: TensorDict, src: TensorDict, env_ids: torch.Tensor
    ) -> None:
        self.expert_data_plane._index_copy_reference_rows_(dst, src, env_ids)

    def _refresh_current_expert_frame(
        self, env_ids: torch.Tensor | None = None, *, advance: bool = False
    ) -> None:
        self.expert_data_plane._refresh_current_expert_frame(env_ids, advance=advance)

    def current_reference_is_final_frame(self) -> torch.Tensor:
        return self.expert_data_plane.current_reference_is_final_frame()

    def _current_local_steps(self, env_ids: torch.Tensor) -> torch.Tensor:
        return self.expert_data_plane._current_local_steps(env_ids)

    def _compute_mpjpe_metric(self) -> torch.Tensor | None:
        return self.expert_data_plane._compute_mpjpe_metric()

    def _compute_mpjpe_metrics(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        return self.expert_data_plane._compute_mpjpe_metrics()

    def _compute_rollout_reference_state_log(self) -> dict[str, float]:
        return self.expert_data_plane._compute_rollout_reference_state_log()

    def get_expert_trajectory_data(
        self, key: str | None = None, joint_indices: Sequence[int] | None = None
    ) -> TensorDict | torch.Tensor:
        """Get the current reference data.

        Args:
            key: Specific key to extract. If None, returns full TensorDict.

        Returns:
            Reference data for all environments
        """
        if self.current_expert_frame is None:
            raise RuntimeError("No reference data available. Call reset() first.")

        if key is None:
            return self.current_expert_frame

        data: torch.Tensor | TensorDict | None = None
        if key in self.current_expert_frame:
            data = self.current_expert_frame[key]
        elif key == "xpos" and "body_pos_w" in self.current_expert_frame:
            data = self.current_expert_frame["body_pos_w"]
        elif key == "xquat" and "body_quat_w" in self.current_expert_frame:
            data = self.current_expert_frame["body_quat_w"]
        elif key == "cvel":
            data = self._get_reference_cvel_fast()

        if data is None:
            available_keys = [str(k) for k in self.current_expert_frame.keys()]
            raise KeyError(f"Key '{key}' not found. Available keys: {available_keys}")

        if joint_indices is not None:
            return data[..., joint_indices]
        else:
            return data  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Expert sampling delegators (reached through ``imitation_interface``).
    # ------------------------------------------------------------------

    def sample_achieved_chunk_windows(
        self, batch_size: int, horizon_steps: int
    ) -> TensorDict | None:
        """Achieved macro windows from the raw-pose ring; None until filled."""
        return self.expert_data_plane.sample_achieved_chunk_windows(
            int(batch_size), int(horizon_steps)
        )

    def expert_macro_frame_stride(self) -> int:
        """Reference frames between consecutive DiffSR macro-window slots.

        Published so a consumer that pairs a pretrained skill encoder with this
        environment can refuse a stride the encoder was not trained on. The
        macro state's width is identical at every stride, so this is the only
        way that mismatch can be detected.
        """
        return self.expert_data_plane._expert_macro_frame_stride()

    def expert_macro_anchor_mode(self) -> str:
        """Frame convention of the DiffSR macro window.

        Published so a consumer that pairs a pretrained skill encoder with this
        environment can refuse a frame convention the encoder was not trained
        on. The macro state's width is identical in every mode, so this is the
        only way that mismatch can be detected.
        """
        return self.expert_data_plane._expert_macro_anchor_mode()

    def sample_expert_macro_transition_batch(
        self,
        batch_size: int,
        horizon_steps: int,
        split: str | None = None,
        eval_fraction: float = 0.1,
        split_seed: int = 0,
        trajectory_ranks: Sequence[int] | torch.Tensor | None = None,
        state_history_steps: int = 0,
    ) -> TensorDict:
        return self.expert_data_plane.sample_expert_macro_transition_batch(
            batch_size,
            horizon_steps,
            split=split,
            eval_fraction=eval_fraction,
            split_seed=split_seed,
            trajectory_ranks=trajectory_ranks,
            state_history_steps=state_history_steps,
        )

    def current_expert_macro_transition_batch(
        self,
        horizon_steps: int,
        env_ids: torch.Tensor | Sequence[int] | None = None,
        state_history_steps: int = 0,
    ) -> TensorDict:
        return self.expert_data_plane.current_expert_macro_transition_batch(
            horizon_steps,
            env_ids=env_ids,
            state_history_steps=state_history_steps,
        )

    def current_achieved_macro_transition_batch(
        self,
        horizon_steps: int,
        env_ids: torch.Tensor | Sequence[int] | None = None,
        state_history_steps: int = 0,
    ) -> TensorDict:
        return self.expert_data_plane.current_achieved_macro_transition_batch(
            horizon_steps,
            env_ids=env_ids,
            state_history_steps=state_history_steps,
        )

    def expert_macro_feature_slices(
        self,
        horizon_steps: int,
    ) -> dict[str, tuple[int, int]]:
        return self.expert_data_plane.expert_macro_feature_slices(horizon_steps)

    def expert_trajectory_motion_names(self) -> list[str]:
        return self.expert_data_plane.expert_trajectory_motion_names()

    def get_current_expert_window_term(
        self,
        term_name: str,
        *,
        past_steps: int,
        future_steps: int,
        frame_stride: int = 1,
        joint_ids: torch.Tensor | Sequence[int] | slice = slice(None),
        anchor_body_name: str = "torso_link",
        reference_body_names: Sequence[str] = (),
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.expert_data_plane.get_current_expert_window_term(
            term_name,
            past_steps=past_steps,
            future_steps=future_steps,
            frame_stride=frame_stride,
            joint_ids=joint_ids,
            anchor_body_name=anchor_body_name,
            reference_body_names=reference_body_names,
            env_ids=env_ids,
        )

    def get_current_expert_goal_term(
        self,
        term_name: str,
        *,
        goal_steps: int,
        joint_ids: torch.Tensor | Sequence[int] | slice = slice(None),
        anchor_body_name: str = "torso_link",
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.expert_data_plane.get_current_expert_goal_term(
            term_name,
            goal_steps=goal_steps,
            joint_ids=joint_ids,
            anchor_body_name=anchor_body_name,
            env_ids=env_ids,
        )

    # ------------------------------------------------------------------
    # Command-channel access (the two terms own their state).
    # ------------------------------------------------------------------

    @property
    def imitation_interface(self) -> ImitationEnvInterface:
        """The capability surface the learner resolves (expert data + publication)."""
        interface = getattr(self, "_imitation_interface", None)
        if interface is None:
            interface = ImitationEnvInterface(self)
            self._imitation_interface = interface
        return interface

    @property
    def reference_command(self):
        """The always-present reference channel term."""
        return self.command_manager.get_term(REFERENCE_TERM_NAME)

    @property
    def actor_command(self):
        """The single actor command term (latent, explicit, or chunk)."""
        return self.command_manager.get_term(ACTOR_TERM_NAME)

    def set_adaptive_reset_weight_fn(self, weight_fn: Any) -> None:
        """Install a custom adaptive start-frame weight function.

        Delegates to the reference channel, which owns the ``StartFrameSampler``.
        The callable takes ``(trajectory_ranks, frame_steps)`` and returns one
        non-negative weight per pair, as expected by
        ``imitationlab.data_tools.common.reset_sampling.StartFrameSampler``.
        """
        self.reference_command.set_weight_fn(weight_fn)

    # ------------------------------------------------------------------
    # Causal planner observation surface (robot-only, no expert reads).
    # ------------------------------------------------------------------

    def _pinned_joint_ids(self) -> torch.Tensor:
        """Live articulation indices in the action term's pinned joint order.

        Physics backends enumerate the articulation differently, so any joint
        vector that a policy or a recorded dataset consumes must be expressed
        in a fixed order. The action term already pins one via
        ``preserve_order=True``; reusing its mapping keeps joint state, the
        action, and the expert command mutually pairable on every backend.
        """
        cached = getattr(self, "_pinned_joint_ids_cache", None)
        if cached is not None:
            return cached
        term_joint_ids = self.action_manager.get_term("joint_pos")._joint_ids
        if isinstance(term_joint_ids, slice):
            term_joint_ids = range(self.robot.num_joints)[term_joint_ids]
        if not isinstance(term_joint_ids, torch.Tensor):
            term_joint_ids = list(term_joint_ids)
        pinned = torch.as_tensor(term_joint_ids, dtype=torch.long, device=self.device)
        self._pinned_joint_ids_cache = pinned
        return pinned

    def _current_causal_planner_frame(
        self, env_ids: torch.Tensor | Sequence[int] | None = None
    ) -> torch.Tensor:
        """Build one planner frame exclusively from the live robot and action state."""
        if env_ids is None:
            env_ids_t = torch.arange(
                self.num_envs, device=self.device, dtype=torch.long
            )
        else:
            env_ids_t = torch.as_tensor(
                env_ids, device=self.device, dtype=torch.long
            ).reshape(-1)
        action = self.action_manager.action.index_select(0, env_ids_t)
        joint_ids = self._pinned_joint_ids()
        return build_causal_planner_frame(
            {
                "joint_pos_rel": (
                    self.robot.data.joint_pos.index_select(0, env_ids_t)
                    - self.robot.data.default_joint_pos.index_select(0, env_ids_t)
                ).index_select(1, joint_ids),
                "joint_vel_rel": (
                    self.robot.data.joint_vel.index_select(0, env_ids_t)
                    - self.robot.data.default_joint_vel.index_select(0, env_ids_t)
                ).index_select(1, joint_ids),
                "base_ang_vel": self.robot.data.root_ang_vel_b.index_select(
                    0, env_ids_t
                ),
                "projected_gravity": self.robot.data.projected_gravity_b.index_select(
                    0, env_ids_t
                ),
                "last_action": action,
            }
        )

    def _initialize_causal_planner_history(self) -> None:
        """Initialize ten 50 Hz frames using repeat-first reset padding."""
        frame = self._current_causal_planner_frame()
        if tuple(frame.shape) != (self.num_envs, CAUSAL_PLANNER_FRAME_DIM):
            raise ValueError(
                "Unexpected causal planner frame shape during initialization: "
                f"{tuple(frame.shape)}."
            )
        self._causal_planner_history = CausalPlannerHistory(
            frame, history_steps=_CAUSAL_PLANNER_HISTORY_STEPS
        )

    def _reset_causal_planner_history(self, env_ids: torch.Tensor) -> None:
        history = getattr(self, "_causal_planner_history", None)
        if history is None:
            return
        env_ids_t = torch.as_tensor(
            env_ids, device=self.device, dtype=torch.long
        ).reshape(-1)
        if int(env_ids_t.numel()) == 0:
            return
        frame = self._current_causal_planner_frame(env_ids_t)
        history.reset(env_ids_t, frame)

    def _append_causal_planner_history(self) -> None:
        history = getattr(self, "_causal_planner_history", None)
        if history is None:
            return
        history.append(self._current_causal_planner_frame())
        # Same cadence, same call sites: the achieved-pose ring records the
        # raw robot pose whenever the causal history records a frame, so both
        # views of "what the robot did" stay step-aligned by construction.
        self.expert_data_plane.append_achieved_pose()

    def causal_planner_observation_spec(
        self, history_steps: int = _CAUSAL_PLANNER_HISTORY_STEPS
    ) -> dict[str, Any]:
        """Return the exact feature and history contract used by the planner."""
        history_steps = int(history_steps)
        if history_steps > _CAUSAL_PLANNER_HISTORY_STEPS:
            raise ValueError(
                "history_steps exceeds the live causal history capacity: "
                f"{history_steps} > {_CAUSAL_PLANNER_HISTORY_STEPS}."
            )
        return causal_planner_observation_spec(history_steps=history_steps)

    def current_causal_planner_observation(
        self,
        env_ids: torch.Tensor | Sequence[int] | None = None,
        history_steps: int = _CAUSAL_PLANNER_HISTORY_STEPS,
    ) -> TensorDict:
        """Return robot-only planner state without reading any expert reference."""
        history_steps = int(history_steps)
        spec = self.causal_planner_observation_spec(history_steps)
        if env_ids is None:
            env_ids_t = torch.arange(
                self.num_envs, device=self.device, dtype=torch.long
            )
        else:
            env_ids_t = torch.as_tensor(
                env_ids, device=self.device, dtype=torch.long
            ).reshape(-1)
        if int(env_ids_t.numel()) == 0:
            raise ValueError("env_ids must select at least one environment.")
        history = self._causal_planner_history.select(
            env_ids_t, history_steps=history_steps
        )
        expected_history = (
            int(env_ids_t.numel()),
            int(spec["history_frames"]),
            int(spec["frame_dim"]),
        )
        if tuple(history.shape) != expected_history:
            raise RuntimeError(
                "Causal planner history shape mismatch: expected "
                f"{expected_history}, got {tuple(history.shape)}."
            )
        planner = TensorDict(
            {"state": history[:, -1], "state_history": history},
            batch_size=[int(env_ids_t.numel())],
            device=self.device,
        )
        return TensorDict(
            {"planner": planner},
            batch_size=[int(env_ids_t.numel())],
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Reference replay targets (side-by-side comparison tools).
    # ------------------------------------------------------------------

    def configure_reference_replay_targets(
        self,
        *,
        source_env_ids: Sequence[int] | torch.Tensor,
        target_env_ids: Sequence[int] | torch.Tensor,
    ) -> None:
        """Configure target envs to replay the reference cursor of source envs."""

        source_env_ids_t = torch.as_tensor(
            source_env_ids, dtype=torch.long, device=self.device
        ).reshape(-1)
        target_env_ids_t = torch.as_tensor(
            target_env_ids, dtype=torch.long, device=self.device
        ).reshape(-1)
        if source_env_ids_t.shape != target_env_ids_t.shape:
            raise ValueError(
                "source_env_ids and target_env_ids must have the same shape."
            )

        self._reference_replay_source_env_ids = source_env_ids_t
        self._reference_replay_target_env_ids = target_env_ids_t
        self._reference_replay_targets_enabled = True

    def apply_reference_replay_targets(self) -> None:
        """Public hook to synchronize and replay configured reference target envs."""

        self._apply_reference_replay_targets()

    def _apply_reference_replay_targets(self) -> None:
        """Replay target envs from their paired source env trajectory cursors."""

        if not self._reference_replay_targets_enabled:
            return
        if (
            self._reference_replay_source_env_ids is None
            or self._reference_replay_target_env_ids is None
        ):
            return

        self.sync_reference_cursor_from_source_envs(
            source_env_ids=self._reference_replay_source_env_ids,
            target_env_ids=self._reference_replay_target_env_ids,
        )
        self._replay_reference(env_ids=self._reference_replay_target_env_ids)

    def sync_reference_cursor_from_source_envs(
        self,
        *,
        source_env_ids: Sequence[int] | torch.Tensor,
        target_env_ids: Sequence[int] | torch.Tensor,
    ) -> None:
        """Copy trajectory cursor state from source envs to target envs."""

        tm = self.trajectory_manager
        source_env_ids_tm = torch.as_tensor(
            source_env_ids, dtype=torch.long, device=tm._state_device
        ).reshape(-1)
        target_env_ids_tm = torch.as_tensor(
            target_env_ids, dtype=torch.long, device=tm._state_device
        ).reshape(-1)
        if source_env_ids_tm.shape != target_env_ids_tm.shape:
            raise ValueError(
                "source_env_ids and target_env_ids must have the same shape."
            )
        if source_env_ids_tm.numel() == 0:
            return

        source_ranks = tm.env_traj_rank.index_select(0, source_env_ids_tm)
        source_steps = tm.env_step.index_select(0, source_env_ids_tm)
        tm.set_env_cursor(
            env_ids=target_env_ids_tm,
            ranks=source_ranks,
            steps=source_steps,
        )

        source_env_ids = source_env_ids_tm.to(device=self.device)
        target_env_ids = target_env_ids_tm.to(device=self.device)

        self._refresh_current_expert_frame(target_env_ids, advance=False)

        tracked_root_pos_w = self._get_tracked_reference_root_pos_w()
        if tracked_root_pos_w is not None:
            self._last_tracked_root_pos_w.index_copy_(
                0,
                target_env_ids,
                tracked_root_pos_w.index_select(0, target_env_ids),
            )
            self._last_tracked_root_pos_valid.index_fill_(0, target_env_ids, True)

    def _replay_reference(
        self, env_ids: torch.Tensor | None = None, reference: TensorDict | None = None
    ):
        """Replay the reference data.

        If env_ids is provided, only replay the reference data for the given
        environments. If env_ids is not provided, replay the reference data
        for all environments.
        """

        if env_ids is None:
            ref = self.current_expert_frame if reference is None else reference
            defaults_pos = self.robot.data.default_joint_pos.torch
            defaults_vel = self.robot.data.default_joint_vel.torch
        else:
            env_ids_tensor = env_ids
            full_reference = (
                self.current_expert_frame if reference is None else reference
            )
            ref = full_reference[env_ids_tensor]
            defaults_pos = self.robot.data.default_joint_pos.torch[env_ids_tensor]
            defaults_vel = self.robot.data.default_joint_vel.torch[env_ids_tensor]

        root_pos, root_quat_opt = self._transform_reference_pose_to_world(
            ref["root_pos"], ref["root_quat"], env_ids=env_ids
        )
        if root_quat_opt is None:
            raise RuntimeError(
                "Failed to transform reference root quaternion for replay."
            )
        root_quat = root_quat_opt
        align_quat, _ = self._get_reference_alignment_transform(env_ids)
        root_lin_vel = self._estimate_reference_root_lin_vel_w_from_pos(
            ref["root_pos"], env_ids=env_ids
        )
        root_ang_vel = math_utils.quat_apply(align_quat, ref["root_ang_vel"])
        root_pose = torch.cat([root_pos, root_quat], dim=-1)
        root_vel = torch.cat([root_lin_vel, root_ang_vel], dim=-1)
        # Extract joint data from reference TensorDict
        # ref is a TensorDict, so accessing keys returns tensors
        joint_pos_raw = ref["joint_pos"]  # type: ignore[assignment]
        joint_vel_raw = ref["joint_vel"]  # type: ignore[assignment]
        joint_pos = joint_pos_raw.clone()
        joint_vel = joint_vel_raw.clone()

        # Replace NaN positions with default values
        joint_pos = torch.where(torch.isnan(joint_pos), defaults_pos, joint_pos)
        joint_vel = torch.where(torch.isnan(joint_vel), defaults_vel, joint_vel)
        # Use link/com-specific writers so all articulation data buffers stay
        # coherent. `base_lin_vel` uses root_com_vel_w + root_link_quat_w
        # internally.
        self.robot.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)
        self.robot.write_root_com_velocity_to_sim(root_vel, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.write_data_to_sim()
        # Refresh cached kinematics buffers (e.g. root_lin_vel_b) after direct
        # state writes.
        self.scene.update(dt=0.0)
        self.robot.update(dt=0.0)
        self._invalidate_mdp_cache()

    def _get_tracked_reference_root_pos_w(self) -> torch.Tensor | None:
        """Return tracked reference root positions in world frame for all environments."""
        if self.current_expert_frame is None:
            return None

        reference_root_pos = self.current_expert_frame.get("root_pos")
        if reference_root_pos is None:
            return None

        # Apply the full per-episode rigid transform (R, t) from reset frame to
        # world frame.
        tracked_root_pos_w, _ = self._transform_reference_pose_to_world(
            reference_root_pos
        )
        return tracked_root_pos_w

    def _estimate_reference_root_lin_vel_w_from_pos(
        self,
        reference_root_pos: torch.Tensor,
        env_ids: torch.Tensor | None = None,
        update_cache: bool = False,
    ) -> torch.Tensor:
        """Estimate reference root linear velocity in world frame from finite differences of root position."""
        if env_ids is None:
            tracked_root_pos_w, _ = self._transform_reference_pose_to_world(
                reference_root_pos
            )
            previous_pos_w = self._last_tracked_root_pos_w
            previous_valid = self._last_tracked_root_pos_valid
        else:
            env_ids_tensor = env_ids.to(dtype=torch.int64)
            tracked_root_pos_w, _ = self._transform_reference_pose_to_world(
                reference_root_pos, env_ids=env_ids_tensor
            )
            previous_pos_w = self._last_tracked_root_pos_w[env_ids_tensor]
            previous_valid = self._last_tracked_root_pos_valid[env_ids_tensor]

        reference_root_lin_vel_w = torch.zeros_like(tracked_root_pos_w)
        dt = float(self.step_dt)
        if dt > 0.0:
            reference_root_lin_vel_w[previous_valid] = (
                tracked_root_pos_w[previous_valid] - previous_pos_w[previous_valid]
            ) / dt

        if update_cache:
            if env_ids is None:
                self._last_tracked_root_pos_w.copy_(tracked_root_pos_w)
                self._last_tracked_root_pos_valid.fill_(True)
            else:
                env_ids_tensor = env_ids.to(dtype=torch.int64)
                self._last_tracked_root_pos_w[env_ids_tensor] = tracked_root_pos_w
                self._last_tracked_root_pos_valid[env_ids_tensor] = True

        return reference_root_lin_vel_w

    # ------------------------------------------------------------------
    # Lifecycle: step / reset.
    # ------------------------------------------------------------------

    def _publish_terminal_obs(self, reset_env_ids: torch.Tensor) -> None:
        """Publish batched terminal observations for the envs being reset.

        ``extras["final_obs"]`` is ``{"_env_ids": LongTensor[k], "obs": nested
        dict of [k, ...] tensors}`` gathered with one advanced index per
        observation key. This replaces the Gymnasium-style object array whose
        per-env ``_slice_obs`` clone loop issued thousands of tiny device
        copies per step at training scale; ``IsaacLabTerminalObsReader`` and
        ``replay_reference.py`` consume both formats.
        """
        ids = reset_env_ids.to(dtype=torch.long)

        def _slice_obs_batch(obs: dict | torch.Tensor):
            if isinstance(obs, dict):
                return {k: _slice_obs_batch(v) for k, v in obs.items()}
            if _FAST_TERMINAL_GATHER:
                # Indexing with a long tensor (a tensor or a TensorDict
                # group) already gathers into new storage; the extra clone
                # copied every terminal row a second time.
                return obs[ids]
            return obs[ids].clone()

        self.extras["final_obs"] = {
            "_env_ids": ids,
            "obs": _slice_obs_batch(self.obs_buf),
        }

    def close(self) -> None:
        """Drain the reference worker before releasing the simulator."""
        try:
            self.expert_data_plane.close()
        finally:
            super().close()
