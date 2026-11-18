"""Configuration of ``imitation-g1-v1``, the G1 full-body tracking task.

The MDP is reimplemented from zita-ch/supermimic_vibed 0525a23 (MIT):
rewards and terminations score the next reference frame, the explicit goal
contains velocities, and resets use the return-normalized distribution of
:class:`~imitationlab.envs.isaaclab.contracts.g1_reset_sampling.G1ResetSampler`.

Fields are selections: a class default, a Hydra ``env.*`` override, or a
launcher argument. ``__post_init__`` declares the fixed protocol and performs no
IO. :meth:`ImitationG1EnvCfg.resolve` derives everything that follows from the
final field values (the dataset layout, the command terms, the observation
features) and is called once by the environment constructor. Isaac Lab applies
``env.*`` overrides after ``__post_init__``, so nothing there may depend on a
value a user can still replace.

A frozen command encoder (``skill_encoder_path``) switches the policy and
critic ``features`` to the latent command; the environment class then loads the
encoder for live observations and imagined rollouts alike. ``step_mode``
selects the eager control step or its masked/CUDA-graph form
(``envs/graph_step/g1.py``), which draws reset randomness per environment
and so is a different random stream.
"""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import (
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import ArticulationRootPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_tasks.utils import PresetCfg

from imitationlab.envs.isaaclab.assets.robots.unitree import (
    UNITREE_G1_29DOF_MIMIC_CFG,
    unitree_g1_29dof_usd_articulation_cfg,
)
from imitationlab.envs.isaaclab.tasks.manager_based.imitation._resolve import (
    coerce_declared_types,
    resolve_remaining_presets,
    stamp_anchor_body,
)
from imitationlab.envs.isaaclab.tasks.manager_based.imitation.command_interface import (
    CommandInterfaceCfg,
    ExplicitCommandCfg,
    ReferenceChannelCfg,
    ReferenceSelectionPreset,
)
from imitationlab.envs.isaaclab.tasks.manager_based.imitation.config.g1.common.actions import (
    G1ActionsCfg,
)
from imitationlab.envs.isaaclab.tasks.manager_based.imitation.config.g1.common.constants import (
    G1_29DOF_DATASET_BODY_NAMES,
    G1_29DOF_ISAACLAB_JOINT_NAMES,
    G1_EE_BODY_NAMES,
    G1_KEYPOINT5_BODY_NAMES,
    G1_TRACKED_BODY_NAMES,
)
from imitationlab.envs.isaaclab.tasks.manager_based.imitation.mdp import g1
from imitationlab.envs.isaaclab.tasks.manager_based.imitation.mdp.commands import (
    ACTOR_TERM_NAME,
    REFERENCE_TERM_NAME,
)
from imitationlab.envs.isaaclab.tasks.manager_based.imitation.motion_data import (
    MotionDataCfg,
    ResolvedMotionData,
)


def _replica_physics():
    return NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=500,
            nconmax=128,
            cone="pyramidal",
            integrator="implicitfast",
            use_mujoco_contacts=True,
        ),
        num_substeps=4,
    )


@configclass
class G1PhysicsCfg(PresetCfg):
    """The familiar Newton selector preserves this recipe's fixed physics."""

    default = _replica_physics()
    newton_mjwarp = _replica_physics()


@configclass
class G1SceneCfg(InteractiveSceneCfg):
    """A flat plane, the robot and a dome light; no height scanner or contact sensor."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = MISSING
    height_scanner = None
    contact_forces = None
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class G1ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        features = ObservationTermCfg(func=g1.policy_features)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class CriticCfg(ObservationGroupCfg):
        features = ObservationTermCfg(func=g1.critic_features)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class WorldModelCfg(ObservationGroupCfg):
        root_state = ObservationTermCfg(func=g1.root_history)
        joint_state = ObservationTermCfg(func=g1.joint_history)
        last_action = ObservationTermCfg(func=g1.action_history)
        reference_cursor = ObservationTermCfg(func=g1.reference_cursor)
        sim_context = ObservationTermCfg(func=g1.sim_context)
        push_flag = ObservationTermCfg(func=g1.push_flag)
        proprio_history = ObservationTermCfg(func=g1.proprio_history)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy = PolicyCfg()
    critic = CriticCfg()
    wm = WorldModelCfg()
    reward_input = None


@configclass
class G1LatentObservationsCfg(G1ObservationsCfg):
    @configclass
    class PolicyCfg(G1ObservationsCfg.PolicyCfg):
        features = ObservationTermCfg(func=g1.latent_policy_features)

    @configclass
    class CriticCfg(G1ObservationsCfg.CriticCfg):
        features = ObservationTermCfg(func=g1.latent_critic_features)

    policy = PolicyCfg()
    critic = CriticCfg()


@configclass
class G1RewardsCfg:
    replica = RewardTermCfg(func=g1.replica_reward, weight=1.0)


@configclass
class G1TerminationsCfg:
    time_out = TerminationTermCfg(func=isaac_mdp.time_out, time_out=True)
    reference_finished = TerminationTermCfg(func=g1.reference_finished, time_out=True)
    tracking_failed = TerminationTermCfg(func=g1.tracking_failed)


@configclass
class G1EventsCfg:
    """The environment owns coupled context randomization and reference resets."""


@configclass
class G1CommandsCfg:
    """The two built command terms, derived from the declared interface.

    Both entries are the objects on ``ImitationG1EnvCfg.command_interface``
    (assigned by :meth:`ImitationG1EnvCfg.resolve`), so a command is configured
    in one place and the manager and the interface cannot disagree.
    """

    reference = None
    actor = None


# The DiffSR macro-state frame: qpos + root pose, no joint velocity.
_ROOT_QPOS_MACRO_STATE_TERMS: list[str] = [
    "expert_motion_qpos",
    "expert_anchor_pos_b",
    "expert_anchor_ori_b",
]


@configclass
class ImitationG1EnvCfg(ManagerBasedRLEnvCfg):
    """``imitation-g1-v1``: explicit or frozen-encoder commands on one MDP."""

    scene: G1SceneCfg = G1SceneCfg(num_envs=4096, env_spacing=2.5)
    observations = G1ObservationsCfg()
    actions = G1ActionsCfg()
    commands: G1CommandsCfg = G1CommandsCfg()
    rewards = G1RewardsCfg()
    terminations = G1TerminationsCfg()
    events = G1EventsCfg()
    curriculum = None

    # -- the declared command interface (the whole command surface) --
    # The dataset-backed reference channel feeds the tracking metrics and the
    # reset path; the explicit actor channel is the reference itself.
    command_interface: CommandInterfaceCfg = CommandInterfaceCfg(
        reference=ReferenceChannelCfg(
            anchor_body_name="pelvis",
            joint_names=G1_29DOF_ISAACLAB_JOINT_NAMES.copy(),
            mpjpe_body_names=G1_TRACKED_BODY_NAMES.copy(),
            ee_body_names=G1_EE_BODY_NAMES.copy(),
            keypoint_body_names=G1_KEYPOINT5_BODY_NAMES.copy(),
            selection=ReferenceSelectionPreset(),
        ),
        actor=ExplicitCommandCfg(),
        encoder=None,
        critic_channels=("reference",),
    )

    # -- the motion data this environment tracks --
    data: MotionDataCfg = MotionDataCfg()

    # -- body / joint name tables --
    # Inputs: resolution copies the joint order and the metric bodies onto the
    # reference channel, which makes the channel's tables outputs.
    reference_joint_names: list[str] = G1_29DOF_ISAACLAB_JOINT_NAMES.copy()
    target_joint_names: list[str] = G1_29DOF_ISAACLAB_JOINT_NAMES.copy()
    reference_body_names: list[str] = G1_29DOF_DATASET_BODY_NAMES.copy()
    mpjpe_metric_body_names: list[str] = G1_TRACKED_BODY_NAMES.copy()

    # Expert-window terms making up one DiffSR macro-state frame (consumed by
    # the env API `sample_expert_macro_transition_batch`, not an obs group):
    # 29 joint qpos + 3 root pos + 6 root ori = 38 per frame. A pretrained
    # encoder is built for one input width and must match.
    expert_macro_state_terms: list[str] | None = _ROOT_QPOS_MACRO_STATE_TERMS.copy()

    # Macro-window frame, anchor jitter and achieved-pose ring. The data plane
    # reads these with `getattr`, and `train.py` refuses undeclared `env.*` keys,
    # so they are declared here.
    expert_macro_frame_stride: int = 1
    expert_macro_anchor_mode: str = "robot"
    expert_macro_rollout_anchor_mode: str = "inherit"
    expert_macro_anchor_walk_std: float = 0.0
    expert_macro_anchor_walk_max: float = 0.1
    expert_macro_anchor_jitter: float = 0.0
    expert_macro_anchor_jump_prob: float = 0.0
    expert_macro_anchor_jump: float = 0.0
    achieved_ring_capacity: int = 0

    # All ranks pool reference visit/failure deltas before reset sampling.
    distributed_reference_sampling: bool = False
    # Data-parallel ranks whose environments together make the global batch
    # (set by the distributed entrypoint). Frame-based env schedules count
    # local frames times this, i.e. global frames, as the trainer's budget does.
    data_parallel_world_size: int = 1

    # -- control-step implementation --
    # "eager": the reference step (reset ids from `reset_buf.nonzero()`).
    # "masked": every reset consumer runs over all envs and is selected by
    #   the reset mask, with per-env counter-based reset randomness: a
    #   different random stream from "eager".
    # "graph": "masked" replayed as CUDA graphs; bit-identical to "masked".
    step_mode: str = "eager"
    # Robot self-collision override. None keeps the asset's own setting; False
    # removes the robot's self-contact pairs. A contact-physics change.
    robot_self_collisions: bool | None = None
    # Robot body pairs whose collisions are filtered (UsdPhysics FilteredPairsAPI
    # on the spawned bodies; Newton imports it per world), each "bodyA:bodyB".
    robot_collision_filter_pairs: list[str] | None = None
    # Replace every convex-mesh collider of the robot with its bounding capsule.
    # EXPERIMENTAL, NOT PHYSICAL AS FITTED: the bounding capsules overlap at rest.
    robot_collision_primitives: bool = False

    # -- derived, written only by `resolve` --
    resolved_data: ResolvedMotionData | None = None
    """The concrete dataset layout. An OUTPUT: never configure this."""

    # Reset-sampler return normalization must use the actor-critic discount.
    reset_discount: float = 0.99
    reset_noise: bool = True
    observation_noise: bool = True
    # Per-env friction, torso COM and per-episode joint offsets. Off, every env
    # gets `clean_friction` and no offsets.
    randomize_context: bool = True
    clean_friction: float = 1.0
    # Episodes start where `command_interface.reference.selection` says, not
    # where the return-based sampler draws them.
    pinned_reset: bool = False
    # Compute only the `wm` observation group, at both points of the step, and
    # publish `final_obs` for that group alone. WMBPTT reads nothing else. The
    # `wm` tensors, rewards, dones and random streams stay bit-identical; PPO
    # and the qualification tools need the other groups and refuse this.
    wm_only_observations: bool = False
    # Share of the resets one bin before recent failure points, from NDM Gym's
    # `--fail_reset`. Zero keeps the base distribution bit for bit. Exact under
    # `reference_prefetch_mode` next or off, the two modes this task accepts.
    reset_fail_share: float = 0.0
    # Text file of motion names, one per line (`#` starts a comment), that the
    # sampled resets never start in. Absolute, or relative to the repository
    # root. The bank and its ranks are unchanged, so boards and checkpoints keep
    # their clip indices. Empty keeps every draw bit for bit.
    reset_exclude_names_file: str = ""
    # Keep the shared reference library's 71-D rows (`G1ReferenceLibrary.states`,
    # 284 bytes per bank frame) in CUDA-registered host memory and gather them over
    # PCIe, instead of a GPU-resident copy. Values and indices are unchanged, so
    # every read is bit-identical; only the gathers' cost changes. For banks that do
    # not fit the GPU beside the simulator (the 129,785-clip BONES-SEED bank is
    # 13.5 GiB). `wm.goal_table` and `wm.reference_body_table` stay GPU tables.
    library_host_mapped: bool = False
    # Build the host-mapped library in a run-local, RAM-backed shared
    # mapping so an evaluation-probe worker does not allocate a second 12.6 GiB
    # host copy on BONES. The trainer owns the anonymous memfd's lifetime.
    library_host_shared: bool = False
    # Internal probe-worker attachment, passed by its run-local descriptor.
    # Empty builds the library normally. Never reuse across runs/datasets.
    library_shared_file: str = ""
    # The same bank on CUDA managed memory instead: the rows the GPU reads migrate
    # to it on first touch and stay as read-only duplicates (dropped under memory
    # pressure), so the hot clips are served from device memory while the bank as a
    # whole need not fit. Reads are bit-identical. Exclusive with the host mapping.
    library_managed: bool = False
    # Where the masked/graphed step reads the current reference frame that feeds
    # the command term's MPJPE metric and the reset refresh: "arrays" gathers the
    # data plane's packed runtime row from the host-mapped resident reference
    # arrays (needs `env.data.reference_arrays_resident=true`, the whole array set
    # pinned in host memory: 46 GiB on the BONES-SEED bank); "library" derives the
    # row from the shared 71-D library (`pack_reference_rows`: the row's root
    # pose, velocities and joints, the tracked bodies by the reward's forward
    # kinematics), so the step needs no array store on the device at all.
    # Rewards, terminations and observations never read this frame; only the
    # logged MPJPE and the command term's values change (library velocities are
    # central differences, body states are FK). The eager step ignores it.
    graph_reference_source: str = "arrays"
    # Frames of its clip each env holds on the device (`G1ReferenceLibrary.env_window_*`):
    # the env step's reads of the current frame and the goal's look-ahead come from this
    # per-env chunk instead of the shared bank, and only the chunk refills (every
    # `K - 13` steps per env, and at resets) cross PCIe when the bank is host-mapped.
    # 0 is off (every read gathers from the bank). Values are bit-identical. Eager,
    # masked and graphed steps alike; the agent's own reads keep gathering from the bank.
    library_env_window: int = 0
    # Factor on the six tracking-kernel widths, from NDM Gym's
    # `--reward_std_scale`. It is an env field because the env builds its MDP
    # first and the agent's task reuses it, so env and imagined rewards move
    # together. Evaluation scores at 1.
    reward_kernel_scale: float = 1.0
    # Keep the reset sampler's bin statistics in the checkpoints and restore them
    # on resume, so a walltime segment does not repeat their start-up transient.
    persist_reset_stats: bool = False

    # -- frozen command encoder (fsq and spectral interfaces) --
    # Empty selects the explicit command. Set, the policy and critic read the
    # encoder's command over the window starting at the next reference frame.
    skill_encoder_path: str = ""
    skill_encoder_sha256: str = ""
    skill_encoder_mode: str = "deterministic"

    def __post_init__(self):
        self.scene.robot = unitree_g1_29dof_usd_articulation_cfg(
            UNITREE_G1_29DOF_MIMIC_CFG
        )
        self.scene.robot.prim_path = "{ENV_REGEX_NS}/Robot"
        self.scene.robot.spawn.activate_contact_sensors = False
        self.scene.robot.spawn.articulation_props = ArticulationRootPropertiesCfg(
            enabled_self_collisions=False
        )
        self.scene.robot.spawn.rigid_props = RigidBodyPropertiesCfg(
            disable_gravity=False,
            linear_damping=0.0,
            angular_damping=0.0,
        )
        self.scene.env_spacing = 0.0
        self.scene.terrain.physics_material.static_friction = 0.3
        self.scene.terrain.physics_material.dynamic_friction = 0.3
        self.scene.terrain.physics_material.restitution = 0.0
        self.sim.physics_material = self.scene.terrain.physics_material
        # 50 Hz control on one 20 ms physics step of four Newton substeps.
        self.sim.dt = 0.02
        self.decimation = 1
        self.sim.render_interval = self.decimation
        self.sim.physics = G1PhysicsCfg()
        self.episode_length_s = 10.0
        self.is_finite_horizon = False
        self.data.reference_prefetch_mode = "next"

    def resolve(self) -> None:
        """Derive every override-dependent field; called by the env constructor.

        Fails loudly on a missing manifest source, an incoherent command
        selection, or a setting this task's fixed protocol does not accept.
        """
        if self.skill_encoder_path:
            latent = G1LatentObservationsCfg()
            self.observations.policy.features = latent.policy.features
            self.observations.critic.features = latent.critic.features
        # Presets first: everything below reads the selected alternatives, and
        # a config built without Hydra still has its presets standing here.
        resolve_remaining_presets(self)
        coerce_declared_types(self)
        self._apply_timing()
        self.resolved_data = self.data.resolve(
            sim_dt=float(self.sim.dt),
            decimation=int(self.decimation),
            joint_names=list(self.reference_joint_names),
            canonical_joint_names=list(self.target_joint_names),
        )

        # The config tables are the input, the reference channel the output.
        interface = self.command_interface
        interface.reference.joint_names = list(self.target_joint_names)
        interface.reference.mpjpe_body_names = list(self.mpjpe_metric_body_names)
        # The interface decides which declared command terms survive, which
        # channel each reads, and at what window; then the two built terms are
        # published to the manager's command group.
        interface.resolve()
        interface.apply_to_observations(self.observations)
        setattr(self.commands, REFERENCE_TERM_NAME, interface.reference)
        setattr(self.commands, ACTOR_TERM_NAME, interface.actor)
        stamp_anchor_body(
            interface.reference.anchor_body_name, self.observations, self.rewards
        )

        self._check_protocol()
        if self.skill_encoder_path:
            self._bind_skill_encoder()
        if self.step_mode != "eager":
            validate_graph_config(self)

    def _apply_timing(self) -> None:
        """Propagate the declared physics step and decimation to the render interval."""
        self.sim.render_interval = int(self.decimation)
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

    def _check_protocol(self) -> None:
        if not isinstance(self.sim.physics, NewtonCfg):
            raise ValueError(
                "imitation-g1-v1 requires the Newton/MJWarp physics protocol."
            )
        if abs(self.sim.dt - 0.02) > 1e-9 or self.decimation != 1:
            raise ValueError("imitation-g1-v1 requires sim.dt=0.02 and decimation=1.")
        expected = _replica_physics()
        if self.sim.physics.num_substeps != expected.num_substeps:
            raise ValueError("imitation-g1-v1 requires four Newton physics substeps.")
        for name in ("njmax", "nconmax", "cone", "integrator", "use_mujoco_contacts"):
            actual = getattr(self.sim.physics.solver_cfg, name)
            required = getattr(expected.solver_cfg, name)
            if actual != required:
                raise ValueError(
                    f"imitation-g1-v1 requires solver {name}={required!r}, got {actual!r}."
                )
        if self.data.resolved_reference_prefetch_mode() == "next_and_reset":
            raise ValueError(
                "imitation-g1-v1 reset sampling must use current returns; use next prefetch."
            )
        if self.wm_only_observations and self.observations.wm is None:
            raise ValueError("wm_only_observations needs the `wm` observation group.")
        if not 0.0 <= self.reset_fail_share < 1.0:
            raise ValueError("reset_fail_share must lie in [0, 1).")
        if self.pinned_reset and self.reset_fail_share > 0.0:
            raise ValueError(
                "reset_fail_share changes the sampled resets, and pinned_reset "
                "never samples; the option would do nothing."
            )
        if self.pinned_reset and self.reset_exclude_names_file:
            raise ValueError(
                "reset_exclude_names_file changes the sampled resets, and "
                "pinned_reset never samples; the option would do nothing."
            )
        if self.pinned_reset and self.persist_reset_stats:
            raise ValueError(
                "persist_reset_stats keeps sampler statistics, and pinned_reset "
                "never samples; the option would do nothing."
            )
        if not self.reward_kernel_scale > 0.0:
            raise ValueError("reward_kernel_scale must be positive.")
        if self.pinned_reset and self.reward_kernel_scale != 1.0:
            raise ValueError(
                "A pinned_reset readout scores the task's own kernels; keep "
                "reward_kernel_scale 1."
            )

    def _bind_skill_encoder(self) -> None:
        """Pin the frozen encoder's weights hash; refuse a mismatched pin."""
        import torch

        from imitationlab.envs.isaaclab.task_model.g1_task import FrozenSkillEncoder

        skill = FrozenSkillEncoder(
            self.skill_encoder_path,
            torch.device("cpu"),
            latent_mode=self.skill_encoder_mode,
        )
        if (
            self.skill_encoder_sha256
            and self.skill_encoder_sha256 != skill.weights_sha256
        ):
            raise ValueError(
                "The skill encoder SHA-256 does not match the pinned weights."
            )
        self.skill_encoder_sha256 = skill.weights_sha256


def validate_graph_config(cfg):
    if cfg.step_mode not in ("masked", "graph"):
        raise ValueError("The captured G1 step requires step_mode 'masked' or 'graph'.")
    if cfg.data.resolved_reference_prefetch_mode() != "next":
        raise ValueError(
            "The masked imitation-g1-v1 step needs reference_prefetch_mode=next."
        )
    if cfg.pinned_reset:
        raise ValueError("pinned_reset readouts require step_mode='eager'.")
    if cfg.graph_reference_source not in ("arrays", "library"):
        raise ValueError(
            "graph_reference_source is 'arrays' (the host-mapped resident reference "
            "arrays) or 'library' (rows derived from the shared 71-D library)."
        )


__all__ = [
    "G1CommandsCfg",
    "G1SceneCfg",
    "G1EventsCfg",
    "G1LatentObservationsCfg",
    "G1ObservationsCfg",
    "G1PhysicsCfg",
    "G1RewardsCfg",
    "G1TerminationsCfg",
    "ImitationG1EnvCfg",
    "validate_graph_config",
]
