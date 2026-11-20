"""Differentiable task model of the ``imitation-g1-v1`` MDP.
License notice: ``third_party/licenses/supermimic_vibed.txt``.

The equations follow ``zita-ch/supermimic_vibed`` commit ``0525a23`` (MIT).
The environment and the imagined model share ``g1_math``; the live reward
uses measured body states while the model obtains them by differentiable FK.

``bind_structured`` (``wm.wm_structured``) takes the semi-implicit drift
weight from the env's physics settings. ``mirror_window`` (``wm.wm_mirror``)
mirrors replay windows left-right (``G1Mirror``).
``reference_setpoint`` (``wm.actor_residual``) follows NDM Gym's
``--sm_residual`` at ``05ea9f3`` (``reference_setpoints`` in
``ndm_gym/algos/common.py``, ``SuperMimic.setpoint``; MIT notices in
``third_party/licenses/ndm_gym.txt``), reading the reference from the goal.
``perturb_start`` (``wm.imag_start_perturb``) follows ``perturb_start`` of
``ndm_gym/algos/common.py`` (same notices). ``contact_reward`` and
``ref_contact_fix`` (``wm.contact_reward``, ``wm.ref_contact_fix``) add
``g1_contact``'s term to ``reward_and_termination``, the model's reward, as
``SuperMimic._reward_term`` of ``ndm_gym/algos/supermimic.py`` does.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from imitationlab.envs.isaaclab.task_model.g1_kinematics import (
    G1_ISAACLAB_JOINT_NAMES,
    matrix_to_rot6d,
    quat_wxyz_to_matrix,
)
from imitationlab.envs.isaaclab.task_model.g1_contact import FEET, reference_contact
from imitationlab.envs.isaaclab.task_model.g1_math import (
    ACTION_DIM,
    INCREMENT_DIM,
    STATE_DIM,
    pack,
    unpack,
    CONTEXT_DIM,
    CONTROL_DT,
    GOAL_FRAME_DIM,
    GOAL_FRAME_JOINT_POS,
    PROPRIO_DIM,
    REWARD_WEIGHTS,
    G1MDP,
    G1Mirror,
    G1ReferenceLibrary,
    drift_weight,
    increment,
    integrate,
    invariant_features,
    normalize_context,
    state_error,
    structured_integrate,
    structured_targets,
)
from imitationlab.envs.isaaclab.task_model.rotations import twist_mat

# Per-dimension error floors of the k-step loss, in increment units: position
# 3 mm, rotation 5 mrad, linear velocity 2 cm/s, angular velocity 0.05 rad/s,
# joint position 3 mrad, joint velocity 0.05 rad/s (supermimic_vibed values).
INCREMENT_FLOOR = (
    (3e-3,) * 3 + (5e-3,) * 3 + (2e-2,) * 3 + (5e-2,) * 3 + (3e-3,) * 29 + (5e-2,) * 29
)
_M32 = 0xFFFFFFFF
# One skill-encoder input frame: joint positions (29), anchor-body position (3)
# and rot6d (6) in a heading frame (the env's root_qpos macro frame). The
# full_body frame adds the joint velocities (29) after the joint positions.
SKILL_FRAME_DIM = 38
SKILL_FRAME_DIMS = (38, 67, 44, 73)
# Columns of one explicit goal frame (``G1ReferenceLibrary.goal``: position 3,
# rot6d 6, root linear and angular velocity 6, joint positions 29, scaled joint
# velocities 29) in the encoder's order: joints, root velocities, anchor pose.
_EXPLICIT_GOAL_COLUMNS = {
    38: (*range(15, 44), *range(0, 9)),
    44: (*range(15, 44), *range(9, 15), *range(0, 9)),
    73: (*range(15, 73), *range(9, 15), *range(0, 9)),
}
# The sin/cos command phase the hl_skill source appends at a one-step
# hold (latent_steps 1, period 1): always phase 0.
SKILL_PHASE = (0.0, 1.0)


def _hash32(x: Tensor) -> Tensor:
    """32-bit integer mix (int64 tensors; multipliers below 2**31 keep every
    product below 2**63)."""
    x = x & _M32
    x = x ^ (x >> 16)
    x = (x * 0x7FEB352D) & _M32
    x = x ^ (x >> 15)
    x = (x * 0x2C1B3C6D) & _M32
    return x ^ (x >> 16)


def keyed_uniform(seed: int, env: Tensor, step: Tensor, n: int) -> Tensor:
    """Uniform draws in [-1, 1), (..., n), that are a pure function of
    (seed, env, step, index): the collector and a later re-simulation of the
    same replay row see the same sensor noise."""
    base = _hash32(_hash32(step.long() ^ (seed & _M32)) ^ env.long())
    idx = _hash32(torch.arange(n, device=env.device, dtype=torch.long) + 0x3C6EF372)
    u = _hash32(base.unsqueeze(-1) ^ idx)
    return u.float() * (2.0 / 4294967296.0) - 1.0


def _rot_t(R: Tensor, x: Tensor) -> Tensor:
    """R^T x for (..., 3, 3) and (..., 3)."""
    return (R.transpose(-1, -2) @ x.unsqueeze(-1)).squeeze(-1)


class FrozenSkillEncoder:
    """A skill-encoder checkpoint's encoder, frozen, as a differentiable goal.

    Rebuilt as the ``hl_skill`` command source rebuilds it. ``__call__``
    maps its input frames (..., F, 38), current frame first, to the command the
    low level reads: z followed by the hold-1 phase ``SKILL_PHASE``. Gradients
    flow to the frames (and so to the imagined state), not into the weights.
    """

    _REQUIRED = {
        "macro_frame_stride": 1,
        "source_anchor": "current",
    }

    def __init__(
        self,
        path: str | Path,
        device: torch.device,
        *,
        latent_mode: str = "deterministic",
    ) -> None:
        from imitationlab.encoders.diffsr.pretrain import HighLevelSkillDiffSRConfig
        from imitationlab.encoders.skill_encoder import build_skill_encoder

        ckpt = torch.load(
            Path(path).expanduser(), map_location="cpu", weights_only=False
        )
        cfg = HighLevelSkillDiffSRConfig.from_dict(ckpt["config"])
        if latent_mode not in {"deterministic", "sonic_fsq"}:
            raise ValueError(f"Unsupported tracking encoder mode: {latent_mode!r}.")
        for key, want in {**self._REQUIRED, "latent_mode": latent_mode}.items():
            if getattr(cfg, key) != want:
                msg = f"skill encoder {path}: {key}={getattr(cfg, key)!r}, the z goal needs {want!r}."
                raise ValueError(msg)
        if cfg.macro_anchor_mode not in ("robot_pos_heading", "reference_heading"):
            msg = (
                f"skill encoder {path}: macro_anchor_mode={cfg.macro_anchor_mode!r}, "
                "the z goal needs 'robot_pos_heading' or 'reference_heading'."
            )
            raise ValueError(msg)
        self.anchor_mode = str(cfg.macro_anchor_mode)
        horizon = int(cfg.horizon_steps)
        window = {"full": horizon, "intermediate": horizon - 1}.get(
            str(cfg.encoder_window_mode)
        )
        if window is None:
            msg = f"skill encoder {path}: encoder_window_mode={cfg.encoder_window_mode!r} is not supported."
            raise ValueError(msg)
        state_dict = ckpt["skill_encoder_state_dict"]
        inputs = int(state_dict["net.0.weight"].shape[1])
        if inputs not in {dim * (window + 1) for dim in SKILL_FRAME_DIMS}:
            msg = (
                f"skill encoder {path} reads {inputs} inputs; the z goal "
                f"builds {window + 1} macro frames of {SKILL_FRAME_DIMS}."
            )
            raise ValueError(msg)
        self.frame_dim = inputs // (window + 1)
        # Where the live window is anchored, and whether height is relative to
        # that anchor: both were fixed when the encoder was pretrained.
        self.rollout_anchor = str(cfg.rollout_anchor)
        self.origin = str(cfg.macro_origin)
        self.qvel_scale = float(cfg.macro_qvel_scale)
        self.heading = str(cfg.macro_heading)
        explicit = (self.rollout_anchor, self.origin) == ("previous_reference", "xyz")
        if self.heading == "forward" and not (
            explicit and self.frame_dim in _EXPLICIT_GOAL_COLUMNS
        ):
            msg = (
                f"skill encoder {path}: the forward heading is the explicit goal's "
                "and needs its anchor, origin and frame content."
            )
            raise ValueError(msg)
        if self.heading != "forward" and self.frame_dim in (44, 73):
            msg = (
                f"skill encoder {path}: root velocities need the explicit goal's frame."
            )
            raise ValueError(msg)
        if self.frame_dim == 73 and self.qvel_scale != 0.1:
            msg = (
                f"skill encoder {path}: the explicit goal scales joint velocity by 0.1."
            )
            raise ValueError(msg)
        if self.rollout_anchor == "robot" and self.origin != "xy":
            msg = f"skill encoder {path}: the robot anchor keeps absolute height."
            raise ValueError(msg)
        self.encoder = build_skill_encoder(
            state_dim=self.frame_dim,
            window_steps=window,
            z_dim=int(cfg.z_dim),
            hidden_dims=tuple(cfg.encoder_hidden_dims),
            spec=cfg.latent_spec(),
            activation=cfg.encoder_activation,
            layer_norm=cfg.encoder_layer_norm,
        )
        self.encoder.load_state_dict(state_dict)
        self.encoder.to(device).eval().requires_grad_(False)
        self.frames = window + 1
        self.command_dim = int(cfg.z_dim) + len(SKILL_PHASE)
        self.latent_mode = latent_mode
        digest = hashlib.sha256()
        for key in sorted(state_dict):
            digest.update(key.encode())
            digest.update(
                state_dict[key].detach().float().contiguous().numpy().tobytes()
            )
        self.weights_sha256 = digest.hexdigest()
        self.path = str(path)

    def __call__(self, frames: Tensor) -> Tensor:
        lead = frames.shape[:-2]
        flat = frames.reshape(-1, self.frames, self.frame_dim)
        z = self.encoder(flat[:, 0], flat[:, 1:]).float()
        phase = z.new_tensor(SKILL_PHASE).expand(z.shape[0], len(SKILL_PHASE))
        return torch.cat((z, phase), dim=-1).reshape(*lead, self.command_dim)


def unwrap_isaac_env(env: Any) -> Any:
    """The Isaac ``ImitationRLEnv`` under TorchRL / gym wrappers."""
    stack, seen = [env], set()
    while stack:
        cur = stack.pop()
        if id(cur) in seen:
            continue
        seen.add(id(cur))
        if hasattr(cur, "trajectory_manager") and hasattr(cur, "reward_manager"):
            return cur
        for attr in ("base_env", "_env", "env", "unwrapped"):
            nxt = getattr(cur, attr, None)
            if nxt is not None:
                stack.append(nxt)
    raise RuntimeError("No Isaac ImitationRLEnv found under the given env.")


class G1TrackingTask:
    """Explicit 10-frame full-body reference, 99-D causal observation history.

    ``reward`` and ``termination`` receive the observation's pre-step cursor
    and compare against frame + 1. ``proprio_at`` takes the cursor belonging
    to the observation itself, including each history frame's own cursor.
    """

    state_dim = STATE_DIM
    action_dim = ACTION_DIM
    context_dim = CONTEXT_DIM
    increment_dim = INCREMENT_DIM
    history = 10
    obs_group = "wm"
    goal_depends_on_state = False  # the goal is a function of (rank, frame) alone
    final_obs_is_post_step = True  # `G1RLEnv` publishes it before the reset
    policy_in_keys: tuple[tuple[str, str], ...] = (
        ("wm", "root_state"),
        ("wm", "joint_state"),
        ("wm", "last_action"),
        ("wm", "reference_cursor"),
        ("wm", "sim_context"),
        ("wm", "proprio_history"),
    )
    policy_obs_dim = 10 * PROPRIO_DIM + 10 * GOAL_FRAME_DIM
    critic_obs_dim = 71 + 29 + 14 * 3 + 3 + 6 + CONTEXT_DIM + 10 * GOAL_FRAME_DIM
    wm_input_dim = 71 + 29 + CONTEXT_DIM
    # The env's dt-scaled step reward at perfect tracking; normalizes returns.
    max_step_reward = sum(w for w in REWARD_WEIGHTS.values() if w > 0) * CONTROL_DT
    increment = staticmethod(increment)
    integrate = staticmethod(integrate)
    state_error = staticmethod(state_error)

    def __init__(
        self,
        env: Any,
        device: torch.device | str,
        *,
        sensor_noise: bool = True,
        goal_frames: int = 10,
        goal_stride: int = 1,
        goal_mode: str = "explicit",
        skill_encoder: str | None = None,
        termination_temperature: float = 0.02,
        contact_reward: float = 0.0,
        ref_contact_fix: bool = False,
    ) -> None:
        if contact_reward < 0:
            raise ValueError(
                f"contact_reward is a weight, at least 0; got {contact_reward}."
            )
        if ref_contact_fix and contact_reward == 0:
            raise ValueError(
                "ref_contact_fix needs contact_reward: it calibrates that term's reference."
            )
        if (
            goal_frames != 10
            or goal_stride != 1
            or goal_mode != "explicit"
            or skill_encoder
        ):
            raise ValueError(
                "The task fixes the replica's explicit 10-frame, stride-one goal; no skill encoder."
            )
        if termination_temperature != 0.02:
            raise ValueError(
                "The task fixes soft height/tilt temperatures at 0.02/0.04."
            )
        self.device = torch.device(device)
        self.isaac_env = unwrap_isaac_env(env)
        self.dt = float(self.isaac_env.step_dt)
        if abs(self.dt - CONTROL_DT) > 1e-9:
            raise ValueError(
                f"imitation-g1-v1 requires 50 Hz control, got step_dt={self.dt}."
            )
        self.sensor_noise = bool(sensor_noise)
        self.goal_frames, self.goal_stride, self.goal_mode = 10, 1, "explicit"
        self.skill = None
        self.skill_anchor = "pelvis"
        self.library = getattr(self.isaac_env, "g1_library", None)
        if self.library is None:
            env_cfg = getattr(self.isaac_env, "cfg", None)
            self.library = G1ReferenceLibrary(
                self.isaac_env,
                self.device,
                host_mapped=bool(getattr(env_cfg, "library_host_mapped", False)),
                managed=bool(getattr(env_cfg, "library_managed", False)),
                host_shared=bool(getattr(env_cfg, "library_host_shared", False)),
                shared_file=str(getattr(env_cfg, "library_shared_file", "")),
            )
            self.isaac_env.g1_library = self.library
        elif self.library.device != self.device:
            raise ValueError(
                "The env and the task must share the same reference-bank device."
            )
        self.kin = self.library.kin
        self.body_names = self.library.body_names
        robot = self.isaac_env.scene["robot"]
        ids, _ = robot.find_joints(list(G1_ISAACLAB_JOINT_NAMES), preserve_order=True)
        q0 = robot.data.default_joint_pos
        q0 = q0.torch if hasattr(q0, "torch") else q0
        self.default_q = q0[0, ids].to(device=self.device, dtype=torch.float32).clone()
        self.mdp = getattr(self.isaac_env, "g1_mdp", None)
        if self.mdp is None:
            self.mdp = G1MDP(
                self.library,
                self.default_q,
                reward_kernel_scale=getattr(
                    getattr(self.isaac_env, "cfg", None), "reward_kernel_scale", 1.0
                ),
            )
            self.isaac_env.g1_mdp = self.mdp
        self._soft_limits = torch.stack((self.mdp.soft_lo, self.mdp.soft_hi), -1)
        self.increment_floor = torch.tensor(INCREMENT_FLOOR, device=self.device)
        self.unsupported_rewards: list[str] = []
        self.unsupported_terminations: list[str] = []
        # ``wm.contact_reward``: this instance's own term. The env's task instance is
        # built without it, and the shared MDP and reward stay the task's.
        self.contact_reward = float(contact_reward)
        self.ref_contact_fix = bool(ref_contact_fix)
        if self.contact_reward > 0:
            self._feet = self.library.body_index(FEET)
            self._reference_contact = reference_contact(
                self.library, self.ref_contact_fix
            )

    @property
    def policy_goal_dim(self) -> int:
        """``wm.actor_arch=enc_dec``: the goal's width. The policy observation is the
        proprio history, then the goal (``policy_obs_from_parts``), so the goal is its
        last ``policy_goal_dim`` columns: 730 explicit, the command's width latent."""
        return self.policy_obs_dim - self.history * PROPRIO_DIM

    def refresh(self) -> None:
        """The task has no curriculum or mutable reward thresholds."""

    def read_obs(self, wm: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """Split the env's ``wm`` group into named tensors (current + history)."""
        n = wm["root_state"].shape[0]
        root = wm["root_state"].reshape(n, self.history, 13)
        joint = wm["joint_state"].reshape(n, self.history, 58)
        hist_states = torch.cat((root, joint), dim=-1)
        cursor = wm["reference_cursor"]
        return {
            "state": hist_states[:, -1],
            "hist_states": hist_states,
            "hist_actions": wm["last_action"].reshape(n, self.history, ACTION_DIM),
            "rank": cursor[:, 0].round().long(),
            "frame": cursor[:, 1].round().long(),
            "episode_step": cursor[:, 2].round().long(),
            "context": wm["sim_context"],
            "push": wm["push_flag"].reshape(n) > 0.5
            if "push_flag" in wm
            else torch.zeros(n, dtype=torch.bool, device=cursor.device),
        }

    def clip_end(self, rank: Tensor, frame: Tensor) -> Tensor:
        """True where ``frame`` is the clip's last frame or past it."""
        return self.library.frames_left(rank, frame) <= 0

    def valid_imagination_start(
        self, rank: Tensor, frame: Tensor, horizon: int
    ) -> Tensor:
        """Keep every transition inside its recorded clip, including frame + H.

        Ending exactly at the final frame remains eligible for critic
        bootstrap, as in the replica's start sampler.
        """
        return self.library.frames_left(rank, frame) >= horizon

    def wm_input(self, state: Tensor, action: Tensor, context: Tensor) -> Tensor:
        return torch.cat(
            (invariant_features(state), action, normalize_context(context)), -1
        )

    def proprio(
        self, states: Tensor, actions: Tensor, context: Tensor, noise: bool
    ) -> Tensor:
        raise ValueError(
            "The proprio frame needs the reference cursor; use proprio_at(state, action, context, rank, frame, noise)."
        )

    def proprio_at(
        self,
        states: Tensor,
        actions: Tensor,
        context: Tensor,
        rank: Tensor,
        frame: Tensor,
        noise: bool,
    ) -> Tensor:
        return self.mdp.proprio(states, actions, context, rank, frame, noise)

    def keyed_proprio_noise(
        self, seed: int, env: Tensor, step: Tensor, episode_step: Tensor | None = None
    ) -> Tensor:
        """One persistent sensor draw per actual frame, repeated at reset fill.

        The replay key is absolute collection step, so regeneration in SVG
        exactly matches collection even after asynchronous resets.
        """
        back = torch.arange(self.history - 1, -1, -1, device=env.device)
        if episode_step is not None:
            back = torch.minimum(back, episode_step.clamp_min(0).unsqueeze(-1))
        shape = (*env.shape, self.history, PROPRIO_DIM)
        if not self.sensor_noise:
            return torch.zeros(shape, device=env.device)
        return (
            keyed_uniform(
                seed, env.unsqueeze(-1), step.unsqueeze(-1) - back, PROPRIO_DIM
            )
            * self.mdp.noise_width
        )

    def keyed_goal_noise(self, seed: int, env: Tensor, step: Tensor) -> None:
        return None

    def policy_obs(
        self,
        hist_states: Tensor,
        hist_actions: Tensor,
        context: Tensor,
        rank: Tensor,
        frame: Tensor,
        noise: bool | None = None,
        noise_key: tuple[int, Tensor, Tensor] | None = None,
        episode_step: Tensor | None = None,
    ) -> Tensor:
        use_noise = self.sensor_noise if noise is None else noise
        back = torch.arange(self.history - 1, -1, -1, device=frame.device)
        if episode_step is not None:
            back = torch.minimum(back, episode_step.clamp_min(0).unsqueeze(-1))
        history_frame = (frame.unsqueeze(-1) - back).clamp_min(0)
        prop = self.proprio_at(
            hist_states,
            hist_actions,
            context.unsqueeze(-2),
            rank.unsqueeze(-1),
            history_frame,
            False,
        )
        if use_noise:
            if noise_key is not None:
                prop = prop + self.keyed_proprio_noise(
                    *noise_key, episode_step=episode_step
                )
            else:
                # Repeat the first observed draw over reset-filled slots, as
                # the replica's collector does when initializing history.
                draw = (2 * torch.rand_like(prop) - 1) * self.mdp.noise_width
                if episode_step is not None:
                    slots = (self.history - 1 - back).expand(*frame.shape, self.history)
                    draw = draw.gather(-2, slots.unsqueeze(-1).expand_as(draw))
                prop = prop + draw
        return self.policy_obs_from_parts(prop, hist_states[..., -1, :], rank, frame)

    def policy_obs_from_env(self, *tensors: Tensor) -> Tensor:
        wm = dict(zip((key[1] for key in self.policy_in_keys), tensors))
        rows = self.read_obs(wm)
        if self.sensor_noise:
            # The deployed actor receives the actual persistent noisy history
            # kept by the env, rather than re-noising old observations.
            return self.policy_obs_from_parts(
                wm["proprio_history"].reshape(-1, self.history, PROPRIO_DIM),
                rows["state"],
                rows["rank"],
                rows["frame"],
            )
        return self.policy_obs(
            rows["hist_states"],
            rows["hist_actions"],
            rows["context"],
            rows["rank"],
            rows["frame"],
            episode_step=rows["episode_step"],
        )

    def goal(
        self,
        state: Tensor,
        rank: Tensor,
        frame: Tensor,
        noise: bool = False,
        noise_values: Tensor | None = None,
    ) -> Tensor:
        return self.library.goal(rank, frame, self.goal_frames)

    explicit_goal = goal

    def policy_obs_from_parts(
        self,
        proprio_hist: Tensor,
        state: Tensor,
        rank: Tensor,
        frame: Tensor,
        goal_noise: Tensor | None = None,
    ) -> Tensor:
        return torch.cat(
            (
                proprio_hist.flatten(-2),
                self.goal(state, rank, frame),
            ),
            -1,
        )

    _setpoint_scale: Tensor | None = None
    _setpoint_columns: slice | None = None
    _drift_weight: float | None = None
    contact_reward: float = 0.0
    ref_contact_fix: bool = False

    def bind_structured(self) -> float:
        """``wm.wm_structured``: the drift weight c = (n + 1) / (2 n) of the env's n
        physics substeps per control step (``sim.physics.num_substeps`` times
        ``decimation``). The task fixes one control step of 0.02 s with four Newton
        substeps, so c = 0.625, NDM Gym's value. The state's velocities are the root
        link's and the joints', the rates of the positions the drift moves."""
        cfg = getattr(self.isaac_env, "cfg", None)
        physics = getattr(getattr(cfg, "sim", None), "physics", None)
        substeps, decimation = (
            getattr(physics, "num_substeps", None),
            getattr(cfg, "decimation", None),
        )
        if substeps is None or decimation is None:
            raise ValueError(
                "wm.wm_structured derives its drift from the env's sim.physics.num_substeps and decimation, which this env does not expose."
            )
        self._drift_weight = drift_weight(int(substeps) * int(decimation))
        return self._drift_weight

    def structured_targets(self, state: Tensor, next_state: Tensor) -> Tensor:
        """``wm.wm_structured``: the model's targets for ``state -> next_state``."""
        return structured_targets(state, next_state, self._drift_weight)

    def structured_integrate(self, state: Tensor, output: Tensor) -> Tensor:
        """``wm.wm_structured``: the next state from the model's outputs."""
        return structured_integrate(state, output, self._drift_weight)

    def bind_reference_setpoint(self) -> None:
        """``wm.actor_residual``: bind ``reference_setpoint`` to the env and check it.

        The action scale comes from the env's ``joint_pos`` action term, in
        ``G1_ISAACLAB_JOINT_NAMES`` order, so the setpoint is the one the env
        executes. The columns come from this task's layout: the first goal
        frame's joint positions, after the proprio history. On bank rows
        (first, middle and the two last frames of up to 1024 clips, so the
        clamped clip end too), those columns of the raw policy observation
        must equal ``ref_state(rank, frame + 1)``'s joint positions exactly.
        A changed goal layout then fails here, not in training. Runs once,
        outside the training loop (it reads the device).
        """
        names = G1_ISAACLAB_JOINT_NAMES
        scale = self._action_scale("wm.actor_residual")
        lo = self.history * PROPRIO_DIM + GOAL_FRAME_JOINT_POS
        if (
            self.policy_obs_dim
            != self.history * PROPRIO_DIM + self.goal_frames * GOAL_FRAME_DIM
        ):
            raise ValueError(
                "wm.actor_residual: the policy observation is not proprio history then goal frames."
            )
        columns = slice(lo, lo + len(names))
        lengths = self.library.length[:1024]
        rank = torch.arange(lengths.numel(), device=self.device).repeat(4)
        frame = torch.cat((lengths * 0, lengths // 2, lengths - 2, lengths - 1))
        obs = self.policy_obs_from_parts(
            torch.zeros(rank.numel(), self.history, PROPRIO_DIM, device=self.device),
            self.library.ref_state(rank, frame),
            rank,
            frame,
        )
        expected = self.library.ref_state(rank, frame + 1)[..., 13:42]
        if obs.shape[-1] != self.policy_obs_dim or not torch.equal(
            obs[..., columns], expected
        ):
            raise ValueError(
                f"wm.actor_residual: policy observation columns {lo}:{lo + len(names)} are not the reference joint positions of frame f + 1."
            )
        self._setpoint_scale = scale.clone()
        self._setpoint_columns = columns

    def _action_scale(self, option: str) -> Tensor:
        """The env's ``joint_pos`` action scale (29,) in ``G1_ISAACLAB_JOINT_NAMES``
        order: one positive value per joint, the same in every env."""
        manager = getattr(self.isaac_env, "action_manager", None)
        if manager is None:
            raise ValueError(
                f"{option} reads the action scale from the env's joint_pos action term; this env has no action manager."
            )
        term = manager.get_term("joint_pos")
        names = tuple(getattr(term, "_joint_names", ()))
        if names != G1_ISAACLAB_JOINT_NAMES:
            raise ValueError(
                f"{option} needs the joint_pos action in G1_ISAACLAB_JOINT_NAMES order, got {names}."
            )
        scale = torch.as_tensor(term._scale, dtype=torch.float32).to(self.device)
        if scale.ndim == 0:
            scale = scale.expand(len(names))
        else:
            scale = scale.reshape(-1, len(names))
            if not bool((scale == scale[:1]).all()):
                raise ValueError(f"{option} needs one action scale for every env.")
            scale = scale[0]
        if not bool((scale > 0).all()):
            raise ValueError(f"{option} needs a positive action scale on every joint.")
        return scale

    _mirror: G1Mirror | None = None
    MIRRORED_KEYS = (
        ("state", "state"),
        ("next_state", "state"),
        ("action", "joints"),
        ("policy_action", "joints"),
        ("prev_action", "joints"),
        ("context", "context"),
    )
    """The window keys ``mirror_window`` maps, and how: every physical one. Masks,
    cursors and outcomes (``valid``, ``ended``, rank, frame, rewards) stay."""

    def bind_mirror(self) -> None:
        """``wm.wm_mirror``: actions mirror as joints only if the default pose and the
        env's action scale are left-right symmetric (an action is a setpoint
        (target - default - offset) / scale); refuse an asset where they are not.
        Runs once, outside the training loop (it reads the device)."""
        mirror = G1Mirror(self.device)
        if not torch.allclose(
            mirror.joints(self.default_q), self.default_q, atol=1e-6, rtol=0
        ):
            raise ValueError(
                "wm.wm_mirror needs a left-right symmetric default pose (roll and yaw joints opposite)."
            )
        scale = self._action_scale("wm.wm_mirror")
        if not torch.allclose(scale[mirror.perm], scale, atol=0, rtol=1e-6):
            raise ValueError(
                "wm.wm_mirror needs the same action scale on each left and right joint."
            )
        self._mirror = mirror

    def mirror_window(self, window: dict[str, Tensor], n: int) -> dict[str, Tensor]:
        """``wm.wm_mirror``: replay ``window`` with its first ``n`` rows replaced by
        their mirror images on every physical key (``MIRRORED_KEYS``), a terminal
        transition's true state (``next_state``) included. Needs ``bind_mirror``."""
        out = dict(window)
        for key, kind in self.MIRRORED_KEYS:
            x = window[key]
            out[key] = torch.cat((getattr(self._mirror, kind)(x[:n]), x[n:]))
        return out

    def perturb_start(
        self, state: Tensor, std: float, generator: torch.Generator | None = None
    ) -> Tensor:
        """``wm.imag_start_perturb``: ``state`` with N(0, ``std``^2) m/s added to the
        root's horizontal linear velocity, world x and y; its z and every other
        column stay (NDM Gym's ``perturb_start``). ``state`` itself is not changed."""
        p, quat, v, w, q, qd = unpack(state)
        kick = std * torch.randn(
            (*v.shape[:-1], 2), device=v.device, dtype=v.dtype, generator=generator
        )
        return pack(p, quat, torch.cat((v[..., :2] + kick, v[..., 2:]), -1), w, q, qd)

    def reference_setpoint(self, obs: Tensor) -> Tensor:
        """The reference pose of frame f + 1 as an action, from the raw policy
        observation ``obs`` (before the actor's normalizer):
        ``(q_ref - default_q) / scale``. The env adds its per-env joint offset,
        as for any action. Needs ``bind_reference_setpoint`` first."""
        q = obs[..., self._setpoint_columns]
        return (q - self.default_q) / self._setpoint_scale

    def critic_obs(
        self, state: Tensor, context: Tensor, rank: Tensor, frame: Tensor
    ) -> Tensor:
        return torch.cat(
            (
                self.mdp.critic_features(state, context, rank, frame),
                self.goal(state, rank, frame),
            ),
            -1,
        )

    def body_states(
        self, state: Tensor, context: Tensor | None = None
    ) -> dict[str, Tensor]:
        pos, rot, lin, ang = self.mdp.body_states(state)
        return {
            "pos": pos,
            "rot": rot,
            "lin_vel": lin,
            "ang_vel": ang,
            "root_pos": state[..., :3],
            "root_rot": rot[..., 0, :, :],
        }

    def reward(
        self,
        next_state: Tensor,
        action: Tensor,
        prev_action: Tensor,
        context: Tensor,
        rank: Tensor,
        frame: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        return self.mdp.reward(next_state, action, prev_action, rank, frame + 1)

    def build_reference_tables(self, *, goal: bool, bodies: bool) -> None:
        """``wm.goal_table``, ``wm.reference_body_table``. The env and the model share
        the library, so both read the tables."""
        if goal:
            self.library.build_goal_table(self.goal_frames)
        if bodies:
            self.library.build_body_table()

    def compile_collection_reward(self) -> None:
        """``wm.compile_eager``: the env computes its reward with the shared MDP."""
        self.mdp.compile_measured_reward()

    def termination(
        self,
        next_state: Tensor,
        context: Tensor,
        rank: Tensor,
        frame: Tensor,
        hard: bool = False,
    ) -> Tensor:
        return self.mdp.termination(next_state, rank, frame + 1, hard=hard)

    def reward_and_termination(
        self,
        next_state: Tensor,
        action: Tensor,
        prev_action: Tensor,
        context: Tensor,
        rank: Tensor,
        frame: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """The model's reward: ``reward``, plus with ``contact_reward`` the foot-contact
        term at the same target frame (``g1_contact``). ``reward`` itself stays the
        simulator's, which the parity checks compare."""
        if self.contact_reward > 0:
            reward, _ = self.mdp.reward(
                next_state,
                action,
                prev_action,
                rank,
                frame + 1,
                contact=self._contact(rank, frame + 1),
            )
        else:
            reward, _ = self.reward(
                next_state, action, prev_action, context, rank, frame
            )
        return reward, self.termination(next_state, context, rank, frame)

    def _contact(self, rank: Tensor, frame: Tensor) -> tuple[float, Tensor, Tensor]:
        """``G1MDP.reward``'s ``contact`` argument at target ``frame``."""
        row, _ = self.library._index(rank, frame)
        return (
            self.contact_reward,
            self._feet,
            self._reference_contact[self.library.bank_rows(row)].float(),
        )

    def goal_binding(self) -> dict[str, Any]:
        return {
            "goal_mode": "explicit",
            "goal_frames": 10,
            "goal_stride": 1,
            "goal_frame_dim": GOAL_FRAME_DIM,
            "task": "g1-v3-supermimic-0525a23",
            "reference_velocity": "qpos-central-difference-50hz-link-origin-fk",
        }

    def describe(self) -> dict[str, Any]:
        return (
            {
                "task": "g1-v3",
                "reward_terms": dict(REWARD_WEIGHTS),
                "unsupported_rewards": [],
                "termination_terms": ["root_height", "root_tilt"],
                "unsupported_terminations": [],
                "goal": self.goal_binding(),
                "policy_obs_dim": self.policy_obs_dim,
                "critic_obs_dim": self.critic_obs_dim,
                "context_dim": self.context_dim,
                "reward_reference_offset": 1,
                "sensor_noise": self.sensor_noise,
                "reference_resident_bytes": self.library.resident_bytes(),
            }
            | (
                {}
                if self._drift_weight is None
                else {"wm_structured_drift_weight": self._drift_weight}
            )
            | (
                {}
                if self.contact_reward == 0
                else {
                    "contact_reward": self.contact_reward,
                    "ref_contact_fix": self.ref_contact_fix,
                    "reference_contact_fraction": float(
                        self._reference_contact.float().mean()
                    ),
                }
            )
        )


class G1LatentTrackingTask(G1TrackingTask):
    """The same dynamics with a frozen root_qpos skill-encoder goal starting at f+1.

    Reuses the existing encoder's validated robot-heading, stride-one input
    contract. The live robot state remains at f; only reference indices shift.
    """

    goal_depends_on_state = True  # the encoder reads the robot's heading
    # No ``wm.actor_residual``: the deployed input holds the command z, not the
    # reference pose, and adding the raw reference would break its binding.
    reference_setpoint = None
    bind_reference_setpoint = None

    def __init__(
        self,
        env,
        device,
        *,
        goal_mode="z",
        skill_encoder=None,
        skill_bf16=False,
        **kwargs,
    ):
        if goal_mode != "z":
            raise ValueError("The latent task requires goal_mode='z'.")
        super().__init__(env, device, **kwargs)
        # ``wm.skill_bf16``: the frozen encoder's matmuls under bf16 autocast (its
        # LayerNorms stay fp32). The actor's own bf16 autocast already rounds z at
        # its first matmul; the env's deployed encoder stays fp32.
        self.skill_bf16 = bool(skill_bf16)
        shared = getattr(self.isaac_env, "g1_skill", None)
        if shared is None:
            if not skill_encoder:
                raise ValueError(
                    "The latent task requires env.skill_encoder_path or skill_encoder."
                )
            shared = FrozenSkillEncoder(skill_encoder, self.device)
        elif skill_encoder and str(skill_encoder) != shared.path:
            candidate = FrozenSkillEncoder(
                skill_encoder, self.device, latent_mode=shared.latent_mode
            )
            if (candidate.weights_sha256, candidate.frames) != (
                shared.weights_sha256,
                shared.frames,
            ):
                raise ValueError(
                    "Environment and WMBPTT skill encoder bindings differ."
                )
        self.skill = shared
        self.goal_mode = "z"
        plane = getattr(self.isaac_env, "expert_data_plane", None)
        self.skill_anchor = str(getattr(plane, "_expert_anchor_body_name", "pelvis"))
        self._skill_anchor_slot = self.body_names.index(self.skill_anchor)
        if self.skill.heading == "forward":
            self._explicit_columns = torch.tensor(
                _EXPLICIT_GOAL_COLUMNS[self.skill.frame_dim], device=self.device
            )
        legacy = (self.skill.rollout_anchor, self.skill.origin, self.skill.frame_dim)
        if self.skill_anchor != "pelvis" and legacy != ("robot", "xy", SKILL_FRAME_DIM):
            raise ValueError("Only the pelvis anchor builds this encoder's window.")
        self.policy_obs_dim = self.history * PROPRIO_DIM + self.skill.command_dim
        self.critic_obs_dim = (
            G1TrackingTask.critic_obs_dim - 10 * GOAL_FRAME_DIM + self.skill.command_dim
        )

    def skill_frames(self, state, rank, frame):
        """The encoder's input frames (..., F, 38) for reference frames f+1 .. f+F,
        as ``_skill_frames_fk`` builds them. With the pelvis anchor they
        come from the reference rows alone: the pelvis is the FK root, so the
        gather's forward kinematics of every tracked body (which only the anchor
        slot of was read) is skipped. The frames are bit for bit the same."""
        if self.skill_anchor != "pelvis":
            return self._skill_frames_fk(state, rank, frame + 1)
        if self.skill.heading == "forward":
            # The explicit goal's own numbers, regrouped into the encoder's order.
            goal = self.library.goal(rank, frame, self.skill.frames)
            goal = goal.reshape(*goal.shape[:-1], self.skill.frames, GOAL_FRAME_DIM)
            return goal[..., self._explicit_columns]
        offsets = torch.arange(1, self.skill.frames + 1, device=frame.device)
        rows = self.library.ref_state(rank, frame.unsqueeze(-1) + offsets)
        ref_p, ref_quat, _, _, ref_q, ref_qd = unpack(rows)
        if self.skill.rollout_anchor == "previous_reference":
            # The reference frame the robot is on, f: the yaw frame and origin
            # the explicit goal uses. Nothing is read from the robot.
            p, quat, *_ = unpack(self.library.ref_state(rank, frame))
        else:
            p, quat, *_ = unpack(state)
        o = torch.zeros_like(quat[..., 0])
        twist = torch.stack((quat[..., 0], o, o, quat[..., 3]), dim=-1)
        H = quat_wxyz_to_matrix(
            twist / twist.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        )
        H = H.unsqueeze(-3)
        height = (
            p[..., 2:3] if self.skill.origin == "xyz" else torch.zeros_like(p[..., 2:3])
        )
        origin = torch.cat((p[..., :2], height), dim=-1)
        rel_pos = _rot_t(H, ref_p - origin.unsqueeze(-2))
        rel_rot = matrix_to_rot6d(H.transpose(-1, -2) @ quat_wxyz_to_matrix(ref_quat))
        joints = (
            ref_q
            if self.skill.frame_dim == SKILL_FRAME_DIM
            else torch.cat((ref_q, ref_qd * self.skill.qvel_scale), dim=-1)
        )
        return torch.cat((joints, rel_pos, rel_rot), dim=-1)

    def _skill_frames_fk(self, state: Tensor, rank: Tensor, frame: Tensor) -> Tensor:
        """The skill encoder's input frames (..., F, 38) for reference frames
        ``frame`` .. ``frame + F - 1``, as the env builds its macro window under
        ``expert_macro_state_terms=[expert_motion_qpos, expert_anchor_pos_b,
        expert_anchor_ori_b]`` and ``expert_macro_anchor_mode=robot_pos_heading``:
        reference joint positions, and the reference anchor-body pose in the
        live robot anchor's heading frame (twist about z, xy-only origin),
        rot6d."""
        offsets = torch.arange(self.skill.frames, device=frame.device)
        ref = self.library.gather(rank, frame.unsqueeze(-1) + offsets)
        p, quat, *_ = unpack(state)
        if self.skill_anchor == "pelvis":
            # The root: the env's own arithmetic (``heading_quat`` zeroes the
            # quaternion's x/y), which also rounds the same way where the
            # heading is ill-conditioned (pelvis near inverted).
            o = torch.zeros_like(quat[..., 0])
            twist = torch.stack((quat[..., 0], o, o, quat[..., 3]), dim=-1)
            H = quat_wxyz_to_matrix(
                twist / twist.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            )
            anchor = p
        else:
            pos, rot = self.kin(
                p, quat_wxyz_to_matrix(quat), state[..., 13:42], (self.skill_anchor,)
            )
            H, anchor = twist_mat(rot[..., 0, :, :]), pos[..., 0, :]
        H = H.unsqueeze(-3)
        origin = torch.cat(
            (anchor[..., :2], torch.zeros_like(anchor[..., 2:3])), dim=-1
        )
        i = self._skill_anchor_slot
        rel_pos = _rot_t(H, ref.body_pos[..., i, :] - origin.unsqueeze(-2))
        rel_rot = matrix_to_rot6d(H.transpose(-1, -2) @ ref.body_rot[..., i, :, :])
        return torch.cat((ref.joint_pos, rel_pos, rel_rot), dim=-1)

    def goal(self, state, rank, frame, noise=False, noise_values=None):
        frames = self.skill_frames(state, rank, frame)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=self.skill_bf16 and frames.is_cuda
        ):
            return self.skill(frames)

    def goal_binding(self):
        skill = self.skill
        # Keys added after the first release appear only when they differ from
        # it, so earlier checkpoints keep their binding.
        added = {
            "encoder_rollout_anchor": (skill.rollout_anchor, "robot"),
            "encoder_origin": (skill.origin, "xy"),
            "encoder_heading": (skill.heading, "twist"),
            "encoder_qvel_scale": (skill.qvel_scale, 1.0),
        }
        return {k: v for k, (v, first) in added.items() if v != first} | {
            "task": "g1-v3-latent",
            "goal_mode": "z",
            "skill_encoder_sha256": self.skill.weights_sha256,
            "encoder_frames": self.skill.frames,
            "encoder_frame_dim": self.skill.frame_dim,
            "encoder_frame_stride": 1,
            "encoder_anchor_body": self.skill_anchor,
            # The first release's name for ``robot_pos_heading``, kept so its
            # checkpoints' bindings still match.
            "encoder_anchor_mode": {"robot_pos_heading": "robot_heading"}.get(
                skill.anchor_mode, skill.anchor_mode
            ),
            "reference_start_offset": 1,
            "command_dim": self.skill.command_dim,
        }

    def describe(self):
        # ``skill_bf16`` is a training precision, not part of the goal binding: a
        # policy trained with it is deployed and evaluated with the fp32 encoder.
        return super().describe() | {"skill_bf16": self.skill_bf16}
