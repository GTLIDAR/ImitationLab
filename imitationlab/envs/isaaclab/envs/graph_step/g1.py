"""Masked and CUDA-graph control steps of the G1 env (``env.step_mode``).

:class:`G1MaskedStepper` runs ``G1RLEnv.step`` with every reset consumer
computed for all ``N`` environments and selected by the reset mask, so a step
has static shapes and no host synchronization:

- the next reference frame is installed for all envs before scoring, as the
  eager step's ``finish_next_reference`` does, so rewards and terminations see
  ``f + 1``;
- the history rings shift in place (the eager step rebinds them with
  ``torch.cat``, which a graph cannot follow);
- the reset sampler's masked forms (``G1ResetSampler.*_masked``) credit the
  ended episodes and draw new starts from per-environment counter-based
  uniforms (:mod:`.counter_rng`), and so do the reset noise, the per-episode
  joint offsets and the reset observation noise;
- simulation writes use Newton's ``*_mask`` writers, manager resets take the
  mask, and episode logs live in persistent device scalars.

It is a different random stream from the eager step, so the two are compared
as distributions.
:class:`G1GraphStepper` replays the masked step's segments as CUDA graphs
around Newton's own physics graph; it runs the same kernels and must agree
with the masked step bit for bit.

The reset writes inside the post-physics segment flag their worlds for a
solver reset and forward kinematics. The command metrics after the reset read
the body poses, which runs Newton's ``forward()`` there, as in the eager step;
the graph records that call, so every replay re-solves the worlds it reset.

``final_obs`` keeps the eager index form, ``{"_env_ids", "obs"}``, but builds it
on first read: the reset ids need ``nonzero``, a host synchronization, which a
consumer that never reads them does not pay. ``final_obs_masked`` holds the same
rows at full width with the reset mask, without that synchronization; both are
valid until the next step.

The masked reset follows NDM Gym's ``--reset_masked`` (``reset_masked`` in
``ndm_gym/sim/rigid/isaaclab_mjwarp.py`` @ ``05ea9f3``, MIT; see
``THIRD_PARTY_NOTICES.md``), with counter-based draws in place of its
all-environment draws from the shared generator.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
import warp as wp

from imitationlab.envs.isaaclab.envs.collect_fastpath import ENV_VAR, fastpath_enabled
from imitationlab.envs.isaaclab.envs.graph_step import counter_rng
from imitationlab.envs.isaaclab.envs.graph_step.graph import (
    CAPTURE_FAST_PATHS,
    GraphStepper,
    _clone_structure,
    _copy_structure,
    _groups,
    _nccl_group_initialized,
    lazy_properties_allowed,
)
from imitationlab.envs.isaaclab.envs.graph_step.masked_step import (
    STREAM_RESET_BIN,
    STREAM_RESET_FRAME,
    MaskedStepper,
    UnsupportedMaskedStep,
    _raise_problems,
)
from imitationlab.envs.isaaclab.task_model.quat import quat_exp, quat_mul

if TYPE_CHECKING:
    from imitationlab.envs.isaaclab.envs.g1_env import G1RLEnv

logger = logging.getLogger(__name__)

# Counter-RNG streams of the G1 reset. Changing a value changes the recipe's
# random stream. The bin and frame draws reuse the generic streams 1 and 2.
STREAM_G1_POSITION = 11
STREAM_G1_ROTATION = 12
STREAM_G1_LINEAR_VELOCITY = 13
STREAM_G1_ANGULAR_VELOCITY = 14
STREAM_G1_JOINT = 15
STREAM_G1_OFFSET = 16
STREAM_G1_PROPRIO = 17

_NOISE_STREAMS = (
    ("position", STREAM_G1_POSITION),
    ("rotation", STREAM_G1_ROTATION),
    ("linear_velocity", STREAM_G1_LINEAR_VELOCITY),
    ("angular_velocity", STREAM_G1_ANGULAR_VELOCITY),
)


def _tensor(value: Any) -> torch.Tensor:
    return value.torch if hasattr(value, "torch") else value


def _gather_rows(obs: Any, ids: torch.Tensor) -> Any:
    if isinstance(obs, dict):
        return {key: _gather_rows(value, ids) for key, value in obs.items()}
    return obs[ids]


class LazyTerminalObs(dict):
    """``final_obs`` in the eager index form, built on first access.

    Until something reads it, the mapping holds nothing and no host
    synchronization has happened. A read after the env stepped again would
    see the next step's buffers, so it is refused.
    """

    __slots__ = ("_build",)

    def __init__(self, build: Any) -> None:
        super().__init__()
        self._build = build

    def _resolve(self) -> None:
        build = self._build
        if build is not None:
            dict.update(self, build())
            self._build = None

    def __getitem__(self, key: str) -> Any:
        self._resolve()
        return dict.__getitem__(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        self._resolve()
        return dict.get(self, key, default)

    def __contains__(self, key: object) -> bool:
        self._resolve()
        return dict.__contains__(self, key)

    def __iter__(self):
        self._resolve()
        return dict.__iter__(self)

    def __len__(self) -> int:
        self._resolve()
        return dict.__len__(self)

    def keys(self):
        self._resolve()
        return dict.keys(self)

    def values(self):
        self._resolve()
        return dict.values(self)

    def items(self):
        self._resolve()
        return dict.items(self)

    def __repr__(self) -> str:
        if self._build is not None:
            return "LazyTerminalObs(<unread>)"
        return f"LazyTerminalObs({dict.__repr__(self)})"


class G1MaskedStepper(MaskedStepper):
    """Run ``G1RLEnv.step`` with mask-selected, static-shape resets."""

    def __init__(self, env: G1RLEnv) -> None:
        super().__init__(env)
        n, device = self.num_envs, self.device
        self.sampler = env.g1_reset_sampler
        robot = self.robot
        joint_ids = env._g1_joint_ids
        num_joints = int(robot.num_joints)
        self._joint_ids = joint_ids
        self._joint_mask = torch.zeros(num_joints, device=device, dtype=torch.bool)
        self._joint_mask[joint_ids] = True
        self._joint_mask_wp = wp.from_torch(self._joint_mask, dtype=wp.bool)
        self._joint_full = torch.zeros(n, num_joints, device=device)
        self._zero_action = torch.zeros(n, int(joint_ids.numel()), device=device)
        # A UI reset request, read on the host before each step; the captured
        # segment reads this device flag.
        self.force_reset = torch.zeros((), device=device, dtype=torch.bool)
        self._force_reset_host = False
        self._g1_part_logs: dict[str, torch.Tensor] | None = None
        self._obs: Any = None
        self._terminal_obs: Any = None
        self.steps_taken = 0
        # ``graph_reference_source=library``: the current reference frame's rows
        # (the command term's MPJPE metric, the reset refresh) are derived from the
        # shared library instead of gathered from the host-mapped packed arrays, so
        # no array store need be device-addressable (``pack_reference_rows``).
        self.library_source = (
            str(getattr(env.cfg, "graph_reference_source", "arrays")) == "library"
        )
        if self.library_source:
            lib = env.g1_library
            frame = self.frame
            wanted = {
                "qpos": (36,),
                "qvel": (35,),
                "body_pos_w": (len(lib.body_names), 3),
                "body_quat_w": (len(lib.body_names), 4),
                "body_lin_vel_w": (len(lib.body_names), 3),
                "body_ang_vel_w": (len(lib.body_names), 3),
            }
            if dict(frame.field_shapes) != wanted or set(frame.field_order) != set(
                wanted
            ):
                raise UnsupportedMaskedStep(
                    f"graph_reference_source=library derives {wanted}; the packed layout is {frame.field_shapes}."
                )
            store = getattr(env.expert_data_plane, "_reference_array_store", None)
            names = tuple(getattr(store, "body_names", ()) or ())
            if names and names != tuple(lib.body_names):
                raise UnsupportedMaskedStep(
                    f"graph_reference_source=library: the arrays' bodies {names} differ from the library's {lib.body_names}."
                )
            self._reset_packed = torch.zeros_like(frame.packed)
        elif self.mapped_store is None:
            raise UnsupportedMaskedStep(
                "a host-mapped packed reference store (env.data.reference_arrays_resident=true), "
                "or env.graph_reference_source=library"
            )

    def _pack_library_rows(
        self, rank: torch.Tensor, frame: torch.Tensor, out: torch.Tensor
    ) -> None:
        from imitationlab.envs.isaaclab.task_model.g1_math import pack_reference_rows

        pack_reference_rows(
            self.env.g1_library,
            rank,
            frame,
            field_order=self.frame.field_order,
            field_shapes=self.frame.field_shapes,
            out=out,
        )

    def gather_segment(self) -> None:
        """The next frame's rows: from the packed store, or derived from the library."""
        if not self.library_source:
            super().gather_segment()
            return
        self._pack_library_rows(
            self.trajectory_manager.env_traj_rank,
            self.next_local_steps,
            self.next_packed,
        )

    # ------------------------------------------------------------------
    # Validation.
    # ------------------------------------------------------------------

    @staticmethod
    def validate(env: G1RLEnv) -> None:
        _raise_problems(G1MaskedStepper.g1_problems(env))

    @staticmethod
    def g1_problems(env: G1RLEnv) -> list[str]:
        """The generic masked-step checks that apply, plus the G1 env's own.

        Both captured steps need the capture fast paths: without them the command
        timer calls ``nonzero`` and the step has no device-addressable store.
        """
        library_source = (
            str(getattr(env.cfg, "graph_reference_source", "arrays")) == "library"
        )
        problems = MaskedStepper.problems(
            env, reset_sampler=False, packed_store=not library_source
        )
        if library_source and getattr(env, "g1_library", None) is None:
            problems.append(
                "graph_reference_source=library without the shared G1 library"
            )
        missing = [name for name in CAPTURE_FAST_PATHS if not fastpath_enabled(name)]
        if missing:
            problems.append(
                f"the collection fast paths {missing} off (set {ENV_VAR}=1 before "
                "the process starts)"
            )
        cfg = env.cfg
        if getattr(env, "g1_reset_sampler", None) is None:
            problems.append("an env without the G1 reset sampler")
        if getattr(cfg, "pinned_reset", False):
            problems.append("pinned_reset (boards stay on the eager step)")
        for mode in ("reset", "interval"):
            for name in env.event_manager.active_terms.get(mode, []):
                problems.append(f"{mode} event {name!r} (the G1 env owns its resets)")
        tm = env.trajectory_manager
        if torch.device(tm.state_device) != torch.device(env.device):
            problems.append(f"trajectory cursors on {tm.state_device}")
        return problems

    def _resolve_reset_event(self) -> dict[str, Any]:
        return {}

    # ------------------------------------------------------------------
    # The control step.
    # ------------------------------------------------------------------

    def step(self, action: torch.Tensor):
        env = self.env
        self.ensure_frame()
        self._read_reset_request()
        self._env_window(True)
        self.pre_physics(action)
        self.launch_gather()
        for k in range(self.substeps):
            self.physics(k)
            if k < self.substeps - 1:
                self.between_substeps()
        self.join_gather()
        self.post_physics()
        self._env_window(False)
        env.obs_buf = self._obs
        return self._returns()

    def _env_window(self, on: bool) -> None:
        """``library_env_window``: the step's library reads (one row per env, at
        or after the cursor) come from the per-env chunk; the agent's reads between
        steps go to the bank. A Python flag: it steers the kernels recorded at
        capture, which replay reads of the chunk thereafter."""
        library = self.env.g1_library
        if library.env_win_length:
            library.env_window_activate(on)

    def _returns(self):
        env = self.env
        self.steps_taken += 1
        self._publish_terminal_obs()
        env.extras["log"] = self.log
        return (
            env.obs_buf,
            env.reward_buf,
            env.reset_terminated,
            env.reset_time_outs,
            env.extras,
        )

    def _read_reset_request(self) -> None:
        """The eager step's UI reset, as a device flag (written only on change)."""
        requested = bool(self.env.sim.consume_reset_request())
        if requested != self._force_reset_host:
            self.force_reset.fill_(requested)
            self._force_reset_host = requested

    def post_physics(self) -> None:
        """``G1RLEnv.step`` after physics, masked (capturable)."""
        env = self.env
        env.scene.update(dt=self.update_dt)
        env.episode_length_buf += 1
        env.common_step_counter += 1
        self._install_next_frame()
        library = env.g1_library
        if library.env_win_length:
            library.env_window_advance(
                self.trajectory_manager.env_traj_rank,
                self.data_plane._current_reference_local_step,
            )
        env.reset_buf = env.termination_manager.compute()
        env.reset_terminated = env.termination_manager.terminated
        env.reset_time_outs = env.termination_manager.time_outs
        env.reward_buf = env.reward_manager.compute(dt=env.step_dt)
        self.sampler.accumulate(env.reward_buf)
        self._append_history()
        self._snapshot_terminal_obs(env._g1_observations())
        # A UI reset truncates surviving episodes; it is not a tracking failure.
        force = self.force_reset
        env.reset_time_outs |= force & ~env.reset_buf
        env.reset_buf |= force
        self.reset_mask.copy_(env.reset_buf)
        self._reset_masked()
        env.command_manager.compute(dt=env.step_dt)
        env._append_causal_planner_history()
        self._store_obs(env._g1_observations())
        self._log_reward_parts()

    def _install_next_frame(self) -> None:
        """``finish_next_reference`` with no override rows: every env at f + 1."""
        dp = self.data_plane
        self.frame.packed.copy_(self.next_packed)
        self.frame.refresh_derived()
        dp._current_reference_local_step.copy_(self.next_local_steps)
        dp._invalidate_mdp_cache()

    def _append_history(self) -> None:
        """``G1RLEnv._g1_append_history`` with the rings shifted in place."""
        env = self.env
        state = env.g1_state()
        action = env.action_manager.action
        rank, frame = env.g1_cursor()
        proprio = env._g1_observation_frame(state, action, rank, frame)
        for ring, row in (
            (env._g1_states, state),
            (env._g1_actions, action),
            (env._g1_proprio, proprio),
        ):
            ring.copy_(torch.cat((ring[:, 1:], row[:, None]), 1))

    def _store_obs(self, obs: Any) -> None:
        # The rings change in place, so the returned rows are copies: a caller
        # holding them across the next step sees this step's values, as it
        # does with the eager step's rebound tensors.
        self._obs = _clone_structure(obs)

    def _snapshot_terminal_obs(self, obs: Any) -> None:  # type: ignore[override]
        self._terminal_obs = _clone_structure(obs)

    def _log_reward_parts(self) -> None:
        parts = self.env._g1_reward_parts
        logs = self._g1_part_logs
        if logs is None:
            logs = {
                f"Reward/{key}": torch.zeros((), device=self.device) for key in parts
            }
            self._g1_part_logs = logs
            self.log.update(logs)
        for key, value in parts.items():
            logs[f"Reward/{key}"].copy_(value.detach().mean())

    # ------------------------------------------------------------------
    # Terminal observations.
    # ------------------------------------------------------------------

    def _publish_terminal_obs(self) -> None:
        env = self.env
        env.extras.pop("final_info", None)
        terminal, mask, published = (
            self._terminal_obs,
            self.reset_mask,
            self.steps_taken,
        )

        def build() -> dict[str, Any]:
            if self.steps_taken != published:
                raise RuntimeError(
                    "final_obs was read after the env stepped again; its rows "
                    "are valid only until the next step."
                )
            ids = mask.nonzero(as_tuple=False).squeeze(-1)
            return {"_env_ids": ids, "obs": _gather_rows(terminal, ids)}

        env.extras["final_obs"] = LazyTerminalObs(build)
        env.extras["final_obs_masked"] = {"_env_mask": mask, "obs": terminal}

    # ------------------------------------------------------------------
    # Reset.
    # ------------------------------------------------------------------

    def _uniform(self, stream: int, width: int, dtype=torch.float32) -> torch.Tensor:
        return counter_rng.uniform(
            self.seed,
            stream,
            self.env_index,
            self.episode_index,
            width=width,
            dtype=dtype,
        )

    def _full_joints(self, values: torch.Tensor) -> torch.Tensor:
        """``values`` in G1 joint order as a full-width, sim-order array."""
        full = self._joint_full.clone()
        full.index_copy_(1, self._joint_ids, values)
        return full

    def _reset_masked(self) -> None:
        """``G1RLEnv._reset_idx`` over the reset mask, in the same order."""
        env = self.env
        mask = self.reset_mask
        sampler = self.sampler
        tm = self.trajectory_manager
        dp = self.data_plane
        robot = self.robot
        cfg = env.cfg
        rows = mask.unsqueeze(-1)

        # Credit the ended episodes, then draw every env's next start.
        sampler.end_episodes_masked(mask, env.reset_terminated)
        u_bin = self._uniform(STREAM_RESET_BIN, 1, torch.float64)[:, 0]
        u_frame = self._uniform(STREAM_RESET_FRAME, 1)[:, 0]
        ranks, frames, bins = sampler.sample_masked(u_bin, u_frame, mask.any())

        # trajectory_manager.reset_envs, masked.
        new_ranks = ranks % int(tm.num_trajectories)
        torch.where(mask, new_ranks, tm.env_traj_rank, out=tm.env_traj_rank)
        max_step = (tm._length.index_select(0, tm.env_traj_rank) - 1).clamp(min=0)
        clamped = torch.minimum(frames.clamp(min=0), max_step)
        torch.where(mask, clamped, tm.env_step, out=tm.env_step)
        if env.g1_library.env_win_length:
            # The reset envs' chunks start at their new frame before any read of it.
            env.g1_library.env_window_refill(
                mask, tm.env_traj_rank, tm.env_step.to(dtype=torch.long)
            )
        # _refresh_current_expert_frame(env_ids), masked.
        if self.library_source:
            self._pack_library_rows(
                tm.env_traj_rank, tm.env_step.to(dtype=torch.long), self._reset_packed
            )
            self.frame.select_into(~mask, self._reset_packed)
        else:
            self.frame.gather_masked(
                self.mapped_store, tm.current_global_indices(), mask
            )
        torch.where(
            mask,
            tm.env_step.to(dtype=torch.long),
            dp._current_reference_local_step,
            out=dp._current_reference_local_step,
        )
        dp._invalidate_mdp_cache()

        state = env.g1_library.ref_state(ranks, frames).clone()
        if cfg.reset_noise:
            noise = {}
            for name, stream in _NOISE_STREAMS:
                width = env._g1_reset_widths[name]
                noise[name] = (2 * self._uniform(stream, width.numel()) - 1) * width
            state[:, :3] += noise["position"]
            state[:, 3:7] = quat_mul(quat_exp(noise["rotation"]), state[:, 3:7])
            state[:, 7:10] += noise["linear_velocity"]
            state[:, 10:13] += noise["angular_velocity"]
            state[:, 13:42] += -0.1 + 0.2 * self._uniform(STREAM_G1_JOINT, 29)
        limits = _tensor(robot.data.joint_pos_limits)[0].index_select(
            0, self._joint_ids
        )
        state[:, 13:42] = state[:, 13:42].clamp(limits[:, 0], limits[:, 1])
        context = env._g1_context
        if cfg.randomize_context:
            offsets = -0.01 + 0.02 * self._uniform(STREAM_G1_OFFSET, 29)
            context[:, 1:30] = torch.where(rows, offsets, context[:, 1:30])

        # ManagerBasedRLEnv._reset_idx, masked (the G1 env has no curriculum or events).
        env.curriculum_manager.compute(env_ids=mask)
        robot.reset(env_mask=self.reset_mask_wp)
        for sensor in env.scene.sensors.values():
            sensor.reset(env_mask=self.reset_mask_wp)
        env.observation_manager.reset(mask)
        env.action_manager.reset(mask)
        self._reset_reward_manager(mask)
        self.log.update(env.curriculum_manager.reset(mask))
        self._reset_command_manager(mask)
        self._reset_termination_manager(mask)
        env.episode_length_buf.masked_fill_(mask, 0)

        # The simulation writes, in the eager order.
        mask_wp = self.reset_mask_wp
        quat = state[:, 3:7]
        pose = torch.cat(
            (state[:, :3] + env.scene.env_origins, quat[:, 1:], quat[:, :1]), -1
        )
        robot.write_root_link_pose_to_sim_mask(root_pose=pose, env_mask=mask_wp)
        robot.write_root_link_velocity_to_sim_mask(
            root_velocity=state[:, 7:13].contiguous(), env_mask=mask_wp
        )
        joint_pos = self._full_joints(state[:, 13:42])
        robot.write_joint_position_to_sim_mask(
            position=joint_pos, joint_mask=self._joint_mask_wp, env_mask=mask_wp
        )
        robot.write_joint_velocity_to_sim_mask(
            velocity=self._full_joints(state[:, 42:71]),
            joint_mask=self._joint_mask_wp,
            env_mask=mask_wp,
        )
        robot.actuators.target_command.set_position_mask(
            value=joint_pos, joint_mask=self._joint_mask_wp, env_mask=mask_wp
        )
        offset = env.action_manager.get_term("joint_pos")._offset
        torch.where(rows, env._g1_default_q + context[:, 1:30], offset, out=offset)

        sampler.begin_episodes_masked(mask, ranks, frames, bins)
        hist = mask.view(-1, 1, 1)
        env._g1_states.copy_(torch.where(hist, state[:, None], env._g1_states))
        env._g1_actions.masked_fill_(hist, 0.0)
        proprio = env.g1_mdp.proprio(
            state, self._zero_action, context, ranks, frames, noise=False
        )
        if cfg.observation_noise:
            u = self._uniform(STREAM_G1_PROPRIO, int(proprio.shape[-1]))
            proprio = proprio + (2 * u - 1) * env.g1_mdp.noise_width
        env._g1_proprio.copy_(torch.where(hist, proprio[:, None], env._g1_proprio))
        history = getattr(env, "_causal_planner_history", None)
        if history is not None:
            history.reset_masked(mask, env._current_causal_planner_frame())
        env._invalidate_mdp_cache()
        self.episode_index += mask


class G1GraphStepper(G1MaskedStepper, GraphStepper):
    """:class:`G1MaskedStepper` replayed as CUDA graphs after warm-up.

    ``validate`` is the masked step's (method order puts it before
    ``GraphStepper``'s), which already requires the capture fast paths.
    """

    def _store_obs(self, obs: Any) -> None:
        if self.static_obs is None:
            G1MaskedStepper._store_obs(self, obs)
            return
        _copy_structure(self.static_obs, obs)

    def _snapshot_terminal_obs(self, obs: Any) -> None:  # type: ignore[override]
        if self.static_terminal_obs is None:
            G1MaskedStepper._snapshot_terminal_obs(self, obs)
            return
        _copy_structure(self.static_terminal_obs, obs)

    def step(self, action: torch.Tensor):
        env = self.env
        self.ensure_frame()
        if self.graphs is None and self.capture_ready():
            # Capture runs the segments' Python once: the recorded kernels read the
            # per-env chunk when the window is on, as the eager steps did.
            self._env_window(True)
            try:
                self._capture(action)
            finally:
                self._env_window(False)
        if self.graphs is None:
            self._eager_steps += 1
            return G1MaskedStepper.step(self, action)
        self._read_reset_request()
        self.static_action.copy_(action)
        self._replay()
        # Fresh tensors each step, as the masked step returns: replay
        # overwrites the static buffers in place.
        env.obs_buf = _clone_structure(self.static_obs)
        self._terminal_obs = self.static_terminal_obs
        return self._returns()

    def _capture(self, action: torch.Tensor) -> None:
        """Record the segments without executing them (see ``GraphStepper``)."""
        env = self.env
        device = self.device
        self.static_action = action.detach().clone()
        # The groups this step computes (``wm`` alone under wm_only_observations).
        if env.cfg.wm_only_observations:
            groups = {"wm": self._obs["wm"]}
        else:
            groups = _groups(env, self._obs)
        self.static_obs = _clone_structure(groups)
        self.static_terminal_obs = _clone_structure(self._terminal_obs)
        python_counters = (env.common_step_counter, env._sim_step_counter)
        torch.cuda.synchronize(device)

        segments = [
            ("pre", lambda: self.pre_physics(self.static_action)),
            ("gather", self.gather_segment),
            ("post", self.post_physics),
        ]
        if self.substeps > 1:
            segments.insert(2, ("between", self.between_substeps))
        graphs = {name: torch.cuda.CUDAGraph() for name, _ in segments}
        pool = torch.cuda.graph_pool_handle()
        capture_stream = torch.cuda.Stream(device=device)
        capture_stream.wait_stream(torch.cuda.current_stream(device))
        warp_stream = wp.stream_from_torch(capture_stream)
        guarded = [self.robot.data, *env.scene.sensors.values()]
        guarded += [sensor.data for sensor in env.scene.sensors.values()]
        mode = "thread_local" if _nccl_group_initialized() else "global"
        for name, segment in segments:
            with (
                lazy_properties_allowed(*guarded),
                wp.ScopedStream(warp_stream),
                torch.cuda.graph(
                    graphs[name],
                    pool=pool,
                    stream=capture_stream,
                    capture_error_mode=mode,
                ),
            ):
                segment()
        torch.cuda.synchronize(device)
        env.common_step_counter, env._sim_step_counter = python_counters
        self.graphs = graphs
        self.pool = pool
        self.captured_at_step = int(env.common_step_counter)
        logger.warning(
            "Captured the G1 graph step at control step %d (%d envs, %d "
            "physics sub-steps per control step).",
            self.captured_at_step,
            self.num_envs,
            self.substeps,
        )


__all__ = [
    "G1GraphStepper",
    "G1MaskedStepper",
    "LazyTerminalObs",
    "STREAM_G1_ANGULAR_VELOCITY",
    "STREAM_G1_JOINT",
    "STREAM_G1_LINEAR_VELOCITY",
    "STREAM_G1_OFFSET",
    "STREAM_G1_POSITION",
    "STREAM_G1_PROPRIO",
    "STREAM_G1_ROTATION",
]
