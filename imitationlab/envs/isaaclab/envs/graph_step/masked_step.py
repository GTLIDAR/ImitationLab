"""Masked, static-shape control step of the imitation env (base stepper).

``ImitationRLEnv.step`` resets the environments in
``reset_buf.nonzero()``: every reset consumer takes a variable-length index
tensor, the selection itself is a host synchronization, and random draws are
shaped by the reset count. :class:`MaskedStepper` runs the same control-step
lifecycle with every reset consumer computed for all ``N`` environments and
selected by a boolean mask:

- reset randomness from :mod:`.counter_rng`, keyed by each environment's own
  episode (or push) index, so a row's numbers do not depend on which other
  environments reset;
- the return-based sampler's masked forms (inverse-CDF draw, in-place bin
  update) and a masked cursor reset;
- reset reference rows from the host-mapped store, masked rows only, into a
  persistent frame (:class:`.reference_frame.ReferenceFrameState`);
- Newton ``*_mask`` simulation writes and mask-selected manager resets;
- episode logs in persistent device scalars, updated on steps that reset.

It is a different random stream from the eager step (and a different
recipe), not a reimplementation of it: compare the two statistically. The
eager masked step and the CUDA-graph replay of it (``.graph``) run the same
kernels and must agree bit for bit.

Only the configuration the combo50b recipe uses is supported;
:meth:`MaskedStepper.validate` refuses anything else by name.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

import torch
import warp as wp

from imitationlab.envs.isaaclab.contracts.training_log import (
    TRACKING_FAILED_ENTRY,
    tracking_failed_share,
    tracking_failure_term_mask,
)
from imitationlab.envs.isaaclab.envs.graph_step import counter_rng
from imitationlab.envs.isaaclab.envs.graph_step.reference_frame import (
    ReferenceFrameState,
)

if TYPE_CHECKING:
    from imitationlab.envs.isaaclab.envs.imitation_rl_env import ImitationRLEnv

# Counter-RNG streams. Changing a value changes the recipe's random stream.
STREAM_RESET_BIN = 1
STREAM_RESET_FRAME = 2
STREAM_RESET_POSE = 3
STREAM_RESET_VELOCITY = 4
STREAM_RESET_JOINT = 5
STREAM_PUSH = 6
STREAM_INTERVAL = 7

_VELOCITY_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")


class UnsupportedMaskedStep(RuntimeError):
    """The environment uses a feature the masked step does not implement."""


class MaskedStepper:
    """Run ``env.step`` with mask-selected, static-shape resets."""

    def __init__(self, env: ImitationRLEnv) -> None:
        self.env = env
        self.validate(env)
        n = int(env.num_envs)
        device = torch.device(env.device)
        self.num_envs = n
        self.device = device
        self.seed = int(getattr(env.cfg, "seed", 0) or 0)
        self.env_index = torch.arange(n, device=device, dtype=torch.int64)
        self.reset_mask = torch.zeros(n, device=device, dtype=torch.bool)
        self.reset_mask_wp = wp.from_torch(self.reset_mask, dtype=wp.bool)
        self.push_mask = torch.zeros(n, device=device, dtype=torch.bool)
        self.push_mask_wp = wp.from_torch(self.push_mask, dtype=wp.bool)
        # Per-env counters keying the reset and push random numbers.
        self.episode_index = torch.zeros(n, device=device, dtype=torch.int64)
        self.push_index = torch.zeros(n, device=device, dtype=torch.int64)

        dp = env.expert_data_plane
        self.data_plane = dp
        self.trajectory_manager = env.trajectory_manager
        self.reference_term = env._reference_term
        self.sampler = self.reference_term._return_reset_sampler
        self.robot = env.scene["robot"]

        tm = self.trajectory_manager
        from collections.abc import Mapping

        from imitationlab.envs.isaaclab.envs.reference_arrays import RUNTIME_FIELDS

        shapes = getattr(dp, "_reference_packed_shapes", None)
        if shapes is None:
            # No packed resident store (the memory-mapped arrays): the layout is the
            # runtime fields' own shapes, from the prefetch source or the array store.
            source = getattr(dp, "_reference_prefetch_source", None)
            store = getattr(dp, "_reference_array_store", None)
            if isinstance(source, Mapping) and all(
                name in source for name in RUNTIME_FIELDS
            ):
                shapes = {
                    name: tuple(source[name].shape[1:]) for name in RUNTIME_FIELDS
                }
            elif store is not None:
                shapes = {
                    name: tuple(store.array(name).shape[1:]) for name in RUNTIME_FIELDS
                }
            else:
                raise UnsupportedMaskedStep(
                    "no packed reference layout: neither a packed store nor reference arrays"
                )

        self.frame = ReferenceFrameState(
            num_envs=n,
            device=device,
            field_shapes=shapes,
            field_order=tuple(RUNTIME_FIELDS),
            nan_joint_template=tm._nan_joint_template,
            ref_to_target_map=tm.ref_to_target_map,
            target_to_ref_map=tm.target_to_ref_map,
        )
        self.frame.adopt(dp.current_expert_frame)
        dp.current_expert_frame = self.frame.frame

        # Next-frame staging (one slot): cursors advance before physics, the
        # rows are gathered on a side stream while physics runs, and the
        # post-physics segment installs them.
        self.next_local_steps = torch.zeros(n, device=device, dtype=torch.long)
        self.next_indices = torch.zeros(n, device=device, dtype=torch.long)
        self.next_packed = torch.zeros_like(self.frame.packed)
        self.gather_stream = torch.cuda.Stream(device=device)
        self.gather_done = torch.cuda.Event()
        mapped = getattr(dp, "_reference_packed_mapped", None)
        # None only for a subclass that derives its rows (the G1 library source).
        self.mapped_store = mapped.device_view if mapped is not None else None
        # Physics sub-steps per control step: one when the backend runs the
        # decimation loop itself, else ``decimation`` calls with the action
        # apply / sim write / scene update between them.
        if env._physics_handles_decimation:
            self.substeps, self.update_dt = 1, float(env.step_dt)
            self._counter_per_substep = int(env.cfg.decimation)
        else:
            self.substeps, self.update_dt = (
                int(env.cfg.decimation),
                float(env.physics_dt),
            )
            self._counter_per_substep = 1

        self._reset_event = self._resolve_reset_event()
        self._push_events = self._resolve_push_events()
        self._build_log_buffers()

    # ------------------------------------------------------------------
    # Validation.
    # ------------------------------------------------------------------

    @staticmethod
    def validate(env: ImitationRLEnv) -> None:
        """Raise :class:`UnsupportedMaskedStep` naming every unsupported feature."""
        _raise_problems(MaskedStepper.problems(env))

    @staticmethod
    def problems(
        env: ImitationRLEnv, *, reset_sampler: bool = True, packed_store: bool = True
    ) -> list[str]:
        """Every unsupported feature, by name; ``reset_sampler=False`` skips the
        return-based sampler requirement for an env that owns its resets;
        ``packed_store=False`` skips the host-mapped packed store requirement for a
        stepper that derives its reference rows elsewhere (the G1 library source)."""
        problems: list[str] = []
        dp = env.expert_data_plane
        if getattr(env, "_reference_replay_targets_enabled", False):
            problems.append("reference replay targets")
        if env.video_recorders:
            problems.append("video recorders")
        if len(env.recorder_manager.active_terms) > 0:
            problems.append("recorder terms")
        if dp.reference_prefetch_mode != "next":
            problems.append(f"reference_prefetch_mode={dp.reference_prefetch_mode}")
        if packed_store and getattr(dp, "_reference_packed_mapped", None) is None:
            problems.append(
                "a host-mapped packed reference store (resident arrays, "
                "runtime_cache_device=cpu, device_gather fast path: "
                "IMITATIONLAB_ENV_COLLECT_FASTPATH=1)"
            )
        term = getattr(env, "_reference_term", None)
        if reset_sampler and (
            term is None or getattr(term, "_return_reset_sampler", None) is None
        ):
            problems.append("a reset sampler other than the return-based one")
        if getattr(dp, "_achieved_ring_capacity", 0) > 0:
            problems.append("achieved-pose ring")
        if getattr(dp, "_anchor_walk_state", None) is not None:
            problems.append("anchor jitter")
        if getattr(dp, "_compute_rollout_reference_state_log", None) is not None:
            if dp._compute_rollout_reference_state_log():
                problems.append("rollout reference state logging")
        robot = env.scene["robot"]
        for name, actuator in getattr(robot, "actuators", {}).items():
            if "Implicit" not in type(actuator).__name__:
                problems.append(f"stateful actuator group {name!r}")
        for mode, names in env.event_manager.active_terms.items():
            for name in names:
                cfg = env.event_manager.get_term_cfg(name)
                func_name = getattr(cfg.func, "__name__", type(cfg.func).__name__)
                if mode == "reset" and func_name != (
                    "reset_root_and_joints_to_reference_with_randomization"
                ):
                    problems.append(f"reset event {name!r} ({func_name})")
                if mode == "interval":
                    if func_name != "push_by_setting_velocity":
                        problems.append(f"interval event {name!r} ({func_name})")
                    elif cfg.is_global_time or not cfg.resample_interval_on_reset:
                        problems.append(f"interval event {name!r} timing")
        for label, manager in (
            ("reward", env.reward_manager),
            ("observation", env.observation_manager),
        ):
            class_cfgs = getattr(manager, "_class_term_cfgs", None) or []
            if isinstance(class_cfgs, dict):
                class_cfgs = [c for group in class_cfgs.values() for c in group]
            for cfg in class_cfgs:
                if not getattr(cfg.func, "supports_mask_reset", False):
                    problems.append(
                        f"{label} class term {type(cfg.func).__name__} (reset not masked)"
                    )
        for cfg in getattr(env.termination_manager, "_class_term_cfgs", []):
            if not hasattr(cfg.func, "reset_masked"):
                problems.append(
                    f"termination class term {type(cfg.func).__name__} (no reset_masked)"
                )
        action_terms = env.action_manager._terms.values()
        for action_term in action_terms:
            if getattr(action_term, "_ema_state", None) is not None or getattr(
                action_term, "_delay_enabled", False
            ):
                problems.append(f"stateful action term {type(action_term).__name__}")
        for name, term in env.command_manager._terms.items():
            lower, upper = term.cfg.resampling_time_range
            if float(lower) < 1.0e8 or float(lower) != float(upper):
                problems.append(f"command {name!r} with a live resample timer")
        return problems

    def _resolve_reset_event(self) -> dict[str, Any]:
        manager = self.env.event_manager
        names = manager.active_terms.get("reset", [])
        if len(names) != 1:
            raise UnsupportedMaskedStep("exactly one reset event is supported.")
        cfg = manager.get_term_cfg(names[0])
        params = dict(cfg.params)
        sig = inspect.signature(cfg.func)
        for key, parameter in sig.parameters.items():
            if key not in params and parameter.default is not inspect.Parameter.empty:
                params[key] = parameter.default
        return params

    def _resolve_push_events(self) -> list[tuple[int, Any, torch.Tensor, torch.Tensor]]:
        manager = self.env.event_manager
        out = []
        for index, name in enumerate(manager.active_terms.get("interval", [])):
            cfg = manager.get_term_cfg(name)
            ranges = cfg.params["velocity_range"]
            bounds = torch.tensor(
                [ranges.get(key, (0.0, 0.0)) for key in _VELOCITY_KEYS],
                device=self.device,
                dtype=torch.float32,
            )
            out.append((index, cfg, bounds[:, 0].clone(), bounds[:, 1].clone()))
        return out

    # ------------------------------------------------------------------
    # Logging buffers (persistent; one dict reused every step).
    # ------------------------------------------------------------------

    def _build_log_buffers(self) -> None:
        env = self.env
        rm, tm_ = env.reward_manager, env.termination_manager
        self._reward_log = torch.zeros(len(rm._term_names), device=self.device)
        self._termination_log = torch.zeros(len(tm_._term_names), device=self.device)
        self._command_logs: dict[str, torch.Tensor] = {}
        for term_name, term in env.command_manager._terms.items():
            for metric_name in term.metrics:
                self._command_logs[f"Metrics/{term_name}/{metric_name}"] = torch.zeros(
                    (), device=self.device
                )
        log: dict[str, Any] = {}
        for i, key in enumerate(rm._term_names):
            log["Episode_Reward/" + key] = self._reward_log[i]
        for key, value in self._command_logs.items():
            log[key] = value
        for i, key in enumerate(tm_._term_names):
            log["Episode_Termination/" + key] = self._termination_log[i]
        # The logging contract's failure share, as the eager
        # `_log_tracking_failed_share` publishes it; a recipe with its own
        # `tracking_failed` term already has the entry above. The term mask is
        # resolved here because building it copies from the host.
        self._tracking_failed_log: torch.Tensor | None = None
        if TRACKING_FAILED_ENTRY not in log:
            self._failure_term_mask = tracking_failure_term_mask(tm_, self.device)
            self._tracking_failed_log = torch.zeros((), device=self.device)
            log[TRACKING_FAILED_ENTRY] = self._tracking_failed_log
        self.log = log

    # ------------------------------------------------------------------
    # The control step.
    # ------------------------------------------------------------------

    def step(self, action: torch.Tensor):
        env = self.env
        self.ensure_frame()
        self.pre_physics(action)
        self.launch_gather()
        for k in range(self.substeps):
            self.physics(k)
            if k < self.substeps - 1:
                self.between_substeps()
        self.join_gather()
        self.post_physics()
        self._publish_terminal_obs()
        return (
            env.obs_buf,
            env.reward_buf,
            env.reset_terminated,
            env.reset_time_outs,
            env.extras,
        )

    def ensure_frame(self) -> None:
        """Re-adopt the reference frame after an eager full ``env.reset()``."""
        dp = self.data_plane
        if dp.current_expert_frame is not self.frame.frame:
            self.frame.adopt(dp.current_expert_frame)
            dp.current_expert_frame = self.frame.frame

    def pre_physics(self, action: torch.Tensor) -> None:
        """Cursor advance, action processing and the sim write (capturable)."""
        env = self.env
        local_steps, indices = self.trajectory_manager.advance_cursors()
        self.next_local_steps.copy_(local_steps)
        self.next_indices.copy_(indices)
        env.action_manager.process_action(action)
        env.action_manager.apply_action()
        env.scene.write_data_to_sim()

    def launch_gather(self) -> None:
        """Queue the next-frame row gather on the side stream (overlaps physics)."""
        stream = self.gather_stream
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            self.gather_segment()
            self.gather_done.record(stream)

    def gather_segment(self) -> None:
        """The side-stream work itself (capturable)."""
        torch.index_select(
            self.mapped_store, 0, self.next_indices, out=self.next_packed
        )

    def join_gather(self) -> None:
        torch.cuda.current_stream(self.device).wait_event(self.gather_done)

    def physics(self, substep: int = 0) -> None:
        """The non-capturable call: one Newton step (its own CUDA graph).

        Our simulation writes (reset, pushes) happen after physics, so only
        the first sub-step of a control step can have worlds to re-solve; a
        forced boundary on the later sub-steps would run FK and solver resets
        over all-zero masks.
        """
        env = self.env
        if substep == 0:
            _mark_kinematics_dirty()
        env._sim_step_counter += self._counter_per_substep
        env.sim.step(render=False)

    def between_substeps(self) -> None:
        """Scene update after one sub-step and the next sub-step's sim write."""
        env = self.env
        env.scene.update(dt=self.update_dt)
        env.action_manager.apply_action()
        env.scene.write_data_to_sim()

    def post_physics(self) -> None:
        """Everything after physics through the observations (capturable)."""
        env = self.env
        env.scene.update(dt=self.update_dt)
        env.episode_length_buf += 1
        env.common_step_counter += 1
        env.reset_buf = env.termination_manager.compute()
        env.reset_terminated = env.termination_manager.terminated
        env.reset_time_outs = env.termination_manager.time_outs
        env.reward_buf = env.reward_manager.compute(dt=env.step_dt)
        self.reference_term.record_step_rewards(env.reward_buf)

        mask = self.reset_mask
        mask.copy_(env.reset_buf)
        self._snapshot_terminal_obs()
        self._reset_masked()
        env.command_manager.compute(dt=env.step_dt)
        self._interval_events()
        self._finish_next_reference()
        env._append_causal_planner_history()
        self._store_obs(env.observation_manager.compute(update_history=True))

    def _store_obs(self, obs: Any) -> None:
        self.env.obs_buf = obs

    def _snapshot_terminal_obs(self) -> None:
        """Keep the pre-reset observation for the terminal publish."""
        self._terminal_obs = self.env.obs_buf

    # ------------------------------------------------------------------
    # Terminal observations.
    # ------------------------------------------------------------------

    def _publish_terminal_obs(self) -> None:
        """Eager-compatible ``final_obs`` (the wrapper's index format).

        The terminal observation is the observation buffer before the reset,
        exactly as the eager step publishes it. The static collector reads
        :attr:`reset_mask` and the pre-reset observation directly and never
        calls this (it synchronizes on ``nonzero``).
        """
        env = self.env
        env.extras.pop("final_obs", None)
        env.extras.pop("final_info", None)
        if not getattr(self, "publish_indexed_terminal_obs", True):
            return
        ids = self.reset_mask.nonzero(as_tuple=False).squeeze(-1)
        if ids.numel() == 0:
            return

        def _slice(obs):
            if isinstance(obs, dict):
                return {k: _slice(v) for k, v in obs.items()}
            return obs[ids]

        env.extras["final_obs"] = {"_env_ids": ids, "obs": _slice(self._terminal_obs)}

    # ------------------------------------------------------------------
    # Reset.
    # ------------------------------------------------------------------

    def _reset_masked(self) -> None:
        env = self.env
        mask = self.reset_mask
        tm = self.trajectory_manager
        sampler = self.sampler
        dp = self.data_plane

        # Fold finished episodes into the return bins; publish their returns.
        sampler.end_episodes_masked(mask)
        metrics = self.reference_term.metrics
        torch.where(
            mask,
            sampler.last_normalized_return,
            metrics["reset_return_norm"],
            out=metrics["reset_return_norm"],
        )
        torch.where(
            mask,
            sampler.last_normalized_undiscounted,
            metrics["reset_return_norm_undiscounted"],
            out=metrics["reset_return_norm_undiscounted"],
        )
        diagnostics = sampler.diagnostics()
        metrics["reset_visited_bin_fraction"].masked_fill_(mask, 0.0).add_(
            mask.to(torch.float32) * diagnostics["visited_bin_fraction"]
        )
        metrics["reset_visited_bin_return_mean"].masked_fill_(mask, 0.0).add_(
            mask.to(torch.float32) * diagnostics["visited_bin_return_mean"]
        )

        # New cursors from the counter RNG.
        u_bin = counter_rng.uniform(
            self.seed,
            STREAM_RESET_BIN,
            self.env_index,
            self.episode_index,
            width=1,
            dtype=torch.float64,
        )[:, 0]
        u_frame = counter_rng.uniform(
            self.seed, STREAM_RESET_FRAME, self.env_index, self.episode_index, width=1
        )[:, 0]
        ranks, steps = sampler.sample_from_uniforms(u_bin, u_frame)
        new_ranks = ranks % int(tm.num_trajectories)
        torch.where(mask, new_ranks, tm.env_traj_rank, out=tm.env_traj_rank)
        max_step = (tm._length.index_select(0, tm.env_traj_rank) - 1).clamp(min=0)
        clamped = torch.minimum(steps.clamp(min=0), max_step)
        torch.where(mask, clamped, tm.env_step, out=tm.env_step)
        sampler.begin_episodes_masked(mask, ranks, steps)

        # Reference rows at the new cursors, masked envs only.
        self.frame.gather_masked(
            dp._reference_packed_mapped.device_view, tm.current_global_indices(), mask
        )
        torch.where(
            mask,
            tm.env_step.to(device=self.device, dtype=torch.long),
            dp._current_reference_local_step,
            out=dp._current_reference_local_step,
        )
        dp._invalidate_mdp_cache()

        # Isaac Lab's _reset_idx, masked.
        env.curriculum_manager.compute(env_ids=mask)
        self.robot.reset(env_mask=self.reset_mask_wp)
        for sensor in env.scene.sensors.values():
            sensor.reset(env_mask=self.reset_mask_wp)
        self._reset_event_masked()

        env.observation_manager.reset(mask)
        env.action_manager.reset(mask)
        self._reset_reward_manager(mask)
        self.log.update(env.curriculum_manager.reset(mask))
        self._reset_command_manager(mask)
        self._reset_event_manager(mask)
        self._reset_termination_manager(mask)
        env.episode_length_buf.masked_fill_(mask, 0)
        env.extras["log"] = self.log

        tracked = env._get_tracked_reference_root_pos_w()
        if tracked is not None:
            torch.where(
                mask.unsqueeze(-1),
                tracked,
                env._last_tracked_root_pos_w,
                out=env._last_tracked_root_pos_w,
            )
            env._last_tracked_root_pos_valid.logical_or_(mask)
        history = getattr(env, "_causal_planner_history", None)
        if history is not None:
            history.reset_masked(mask, env._current_causal_planner_frame())
        self.episode_index += mask

    def _reset_event_masked(self) -> None:
        """``reset_root_and_joints_to_reference_with_randomization`` over a mask."""
        from imitationlab.envs.isaaclab.tasks.manager_based.imitation.mdp._compiled import (
            apply_reset_randomization,
            replace_nan_with_default,
        )
        from imitationlab.envs.isaaclab.tasks.manager_based.imitation.mdp.events import (
            _force_refresh_derived_state,
            _initialize_reset_bounds,
        )

        env = self.env
        asset = self.robot
        params = self._reset_event
        reference = self.frame.frame
        device = self.device
        _initialize_reset_bounds(
            env, params.get("pose_range"), params.get("velocity_range"), device
        )
        if env._mdp_reset_root_pose_source == "root":
            root_pos, root_quat = reference["root_pos"], reference["root_quat"]
        else:
            root_pos = reference["body_pos_w"][:, 0, :]
            root_quat = reference["body_quat_w"][:, 0, :]
        if env._mdp_reset_root_velocity_source == "root":
            root_lin_vel = reference["root_lin_vel"]
            root_ang_vel = reference["root_ang_vel"]
        elif env._mdp_reset_root_velocity_source == "body":
            root_lin_vel = reference["body_lin_vel_w"][:, 0, :]
            root_ang_vel = reference["body_ang_vel_w"][:, 0, :]
        else:
            root_lin_vel = torch.zeros_like(root_pos)
            root_ang_vel = torch.zeros_like(root_pos)
        joint_pos = replace_nan_with_default(
            reference["joint_pos"], asset.data.default_joint_pos.torch
        ).clone()
        joint_vel = replace_nan_with_default(
            reference["joint_vel"], asset.data.default_joint_vel.torch
        ).clone()

        keys = (self.env_index, self.episode_index)
        pose_bounds = env._mdp_reset_pose_bounds
        velocity_bounds = env._mdp_reset_velocity_bounds
        pose_delta = counter_rng.uniform_range(
            pose_bounds[:, 0],
            pose_bounds[:, 1],
            self.seed,
            STREAM_RESET_POSE,
            *keys,
            width=6,
        )
        velocity_delta = counter_rng.uniform_range(
            velocity_bounds[:, 0],
            velocity_bounds[:, 1],
            self.seed,
            STREAM_RESET_VELOCITY,
            *keys,
            width=6,
        )
        low, high = params.get("joint_position_range", (0.0, 0.0))
        if low == 0.0 and high == 0.0:
            joint_noise = torch.zeros_like(joint_pos)
        else:
            joint_noise = counter_rng.uniform_range(
                float(low),
                float(high),
                self.seed,
                STREAM_RESET_JOINT,
                *keys,
                width=int(joint_pos.shape[-1]),
            )
        limits = asset.data.soft_joint_pos_limits.torch
        (
            root_pos_r,
            root_quat_r,
            root_lin_vel_r,
            root_ang_vel_r,
            joint_pos_r,
            joint_vel_r,
        ) = apply_reset_randomization(
            root_pos,
            root_quat,
            root_lin_vel,
            root_ang_vel,
            joint_pos,
            joint_vel,
            env.scene.env_origins,
            pose_delta,
            velocity_delta,
            joint_noise,
            limits[:, :, 0],
            limits[:, :, 1],
        )
        mask_wp = self.reset_mask_wp
        asset.write_joint_state_to_sim_mask(
            position=joint_pos_r.contiguous(),
            velocity=joint_vel_r.contiguous(),
            env_mask=mask_wp,
        )
        asset.write_root_link_pose_to_sim_mask(
            root_pose=torch.cat((root_pos_r, root_quat_r), dim=-1).contiguous(),
            env_mask=mask_wp,
        )
        asset.write_root_com_velocity_to_sim_mask(
            root_velocity=torch.cat(
                (root_lin_vel_r, root_ang_vel_r), dim=-1
            ).contiguous(),
            env_mask=mask_wp,
        )
        asset.write_data_to_sim()
        env.scene.update(dt=0.0)
        asset.update(dt=0.0)
        _force_refresh_derived_state(asset)
        env._invalidate_mdp_cache()

    def _reset_reward_manager(self, mask: torch.Tensor) -> None:
        rm = self.env.reward_manager
        weight = mask.to(torch.float32)
        count = weight.sum()
        mean = (rm._episode_sum_buf * weight.unsqueeze(-1)).sum(
            dim=0
        ) / count.clamp_min(1.0)
        mean = mean / self.env.max_episode_length_s
        torch.where(count > 0, mean, self._reward_log, out=self._reward_log)
        rm._episode_sum_buf.masked_fill_(mask.unsqueeze(-1), 0.0)
        for cfg in rm._class_term_cfgs:
            cfg.func.reset(env_ids=mask)

    def _reset_command_manager(self, mask: torch.Tensor) -> None:
        weight = mask.to(torch.float32)
        count = weight.sum()
        has_reset = count > 0
        for term_name, term in self.env.command_manager._terms.items():
            for metric_name, metric_value in term.metrics.items():
                log = self._command_logs[f"Metrics/{term_name}/{metric_name}"]
                mean = (metric_value * weight).sum() / count.clamp_min(1.0)
                torch.where(has_reset, mean, log, out=log)
                metric_value.masked_fill_(mask, 0.0)
            term.command_counter.masked_fill_(mask, 0)
            # CommandTerm._resample: the timer range is constant (validated).
            term.time_left.masked_fill_(mask, float(term.cfg.resampling_time_range[0]))
            term.command_counter += mask
            term._resample_command(mask)
            reset_extra = getattr(term, "reset_masked_extra", None)
            if reset_extra is not None:
                reset_extra(mask)

    def _reset_event_manager(self, mask: torch.Tensor) -> None:
        manager = self.env.event_manager
        for index, cfg, _, _ in self._push_events:
            lower, upper = cfg.interval_range_s
            sampled = (
                counter_rng.uniform(
                    self.seed,
                    STREAM_INTERVAL,
                    self.env_index,
                    self.episode_index,
                    width=1,
                )[:, 0]
                * (upper - lower)
                + lower
            )
            time_left = manager._interval_term_time_left[index]
            torch.where(mask, sampled, time_left, out=time_left)

    def _reset_termination_manager(self, mask: torch.Tensor) -> None:
        manager = self.env.termination_manager
        self._termination_log.copy_(manager._last_episode_dones.float().mean(dim=0))
        if self._tracking_failed_log is not None:
            self._tracking_failed_log.copy_(
                tracking_failed_share(
                    manager._last_episode_dones, self._failure_term_mask
                )
            )
        for cfg in manager._class_term_cfgs:
            cfg.func.reset_masked(mask)

    # ------------------------------------------------------------------
    # Interval events and the next reference frame.
    # ------------------------------------------------------------------

    def _interval_events(self) -> None:
        if not self._push_events:
            return
        manager = self.env.event_manager
        dt = float(self.env.step_dt)
        asset = self.robot
        for index, cfg, low, high in self._push_events:
            time_left = manager._interval_term_time_left[index]
            time_left -= dt
            fire = self.push_mask
            torch.lt(time_left, 1e-6, out=fire)
            lower, upper = cfg.interval_range_s
            sampled = (
                counter_rng.uniform(
                    self.seed,
                    STREAM_INTERVAL,
                    self.env_index,
                    self.push_index + (1 << 40),
                    width=1,
                )[:, 0]
                * (upper - lower)
                + lower
            )
            torch.where(fire, sampled, time_left, out=time_left)
            delta = counter_rng.uniform_range(
                low,
                high,
                self.seed,
                STREAM_PUSH,
                self.env_index,
                self.push_index,
                width=6,
            )
            velocity = asset.data.root_vel_w.torch + delta
            asset.write_root_velocity_to_sim_mask(
                root_velocity=velocity.contiguous(), env_mask=self.push_mask_wp
            )
            self.push_index += fire

    def _finish_next_reference(self) -> None:
        """Install the staged next frame; reset rows keep their new start."""
        dp = self.data_plane
        keep = self.reset_mask
        self.frame.select_into(keep, self.next_packed)
        torch.where(
            keep,
            dp._current_reference_local_step,
            self.next_local_steps,
            out=dp._current_reference_local_step,
        )
        dp._invalidate_mdp_cache()


def _raise_problems(problems: list[str]) -> None:
    if problems:
        raise UnsupportedMaskedStep(
            "The masked step does not support: " + "; ".join(problems) + "."
        )


def _mark_kinematics_dirty() -> None:
    """Make Newton's next step consume its device FK/solver-reset masks.

    Newton gates ``forward()`` (solver reset and FK for written worlds) on a
    Python flag that its ``_mask`` writes set when they run in Python. Under
    graph replay the writes run but the flag is never set, so it is raised
    before every physics step; the device masks select the worlds.
    """
    try:
        from isaaclab_newton.physics.newton_manager import NewtonManager
    except ImportError:  # pragma: no cover - PhysX builds
        return
    NewtonManager.kinematics_dirty = True


__all__ = ["MaskedStepper", "UnsupportedMaskedStep"]
