"""``imitation-g1-v1``: the G1 full-body tracking environment.

MDP design reimplemented from zita-ch/supermimic_vibed 0525a23 (MIT).
ImitationLab's ExpertDataPlane owns loading and delivering the reference
arrays. This class changes only the protocol: the post-action state
is scored at f+1, context randomization and return-based resets are coupled,
and terminal observations contain the actual pre-reset state and history.
"""

from __future__ import annotations

import torch
import warp as wp
from newton import ModelFlags
from isaaclab.envs import ManagerBasedRLEnv

from imitationlab.envs.isaaclab.contracts.g1_reset_sampling import G1ResetSampler
from imitationlab.envs.isaaclab.envs.collect_fastpath import fastpath_enabled
from imitationlab.envs.isaaclab.envs.imitation_rl_env import ImitationRLEnv
from imitationlab.envs.isaaclab.task_model.g1_kinematics import (
    G1_ISAACLAB_JOINT_NAMES,
    G1_TRACKED_BODY_NAMES,
    quat_wxyz_to_matrix,
)
from imitationlab.envs.isaaclab.task_model.quat import quat_exp, quat_mul


#: The observation group WMBPTT reads, and the only one `wm_only_observations`
#: computes.
WM_GROUP = "wm"

#: Half-widths of the reset perturbations.
RESET_NOISE_WIDTHS = {
    "position": (0.05, 0.05, 0.01),
    "rotation": (0.1, 0.1, 0.2),
    "linear_velocity": (0.5, 0.5, 0.2),
    "angular_velocity": (0.52, 0.52, 0.78),
}

_FAST_MASKED_CREDIT = fastpath_enabled("masked_credit")


def _tensor(value):
    return value.torch if hasattr(value, "torch") else value


class G1RLEnv(ImitationRLEnv):
    """The replica MDP on the repository's arrays and simulator integration."""

    #: The control steps this class runs. It overrides ``step`` and ``_reset_idx``,
    #: so the base class's masked and graph steps would be ignored without notice.
    g1_step_modes: tuple[str, ...] = ("eager",)
    g1_step_mode_error = "G1RLEnv runs only step_mode='eager'."

    @property
    def terminal_observation_is_post_step(self) -> bool:
        """True; the qualification tools read this before they touch the env.

        ``g1_env_validation``, ``prefetch_validation`` and
        ``validate_world_model_task`` need the policy and critic groups, which
        ``wm_only_observations`` removes, so the read itself refuses that mode.
        """
        if self.cfg.wm_only_observations:
            raise ValueError(
                "Environment qualification needs every observation group; run it with "
                "env.wm_only_observations=false."
            )
        return True

    def __init__(self, cfg, render_mode=None, **kwargs):
        if str(getattr(cfg, "step_mode", "eager")) not in self.g1_step_modes:
            raise ValueError(self.g1_step_mode_error)
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        ids, _ = self.robot.find_joints(
            list(G1_ISAACLAB_JOINT_NAMES), preserve_order=True
        )
        self._g1_joint_ids = torch.tensor(ids, device=self.device, dtype=torch.long)
        body_ids, _ = self.robot.find_bodies(
            list(G1_TRACKED_BODY_NAMES), preserve_order=True
        )
        self._g1_body_ids = torch.tensor(body_ids, device=self.device, dtype=torch.long)
        self._g1_context = torch.zeros(self.num_envs, 33, device=self.device)
        self._g1_default_q = _tensor(self.robot.data.default_joint_pos)[
            0, self._g1_joint_ids
        ].clone()
        self._g1_states = torch.zeros(self.num_envs, 10, 71, device=self.device)
        self._g1_actions = torch.zeros(self.num_envs, 10, 29, device=self.device)
        self._g1_proprio = torch.zeros(self.num_envs, 10, 99, device=self.device)
        self._g1_reward_parts = {}
        self._g1_reset_widths = {
            name: torch.tensor(width, device=self.device)
            for name, width in RESET_NOISE_WIDTHS.items()
        }
        self.g1_task = self._make_tracking_task()
        self.g1_library = self.g1_task.library
        self.g1_mdp = self.g1_task.mdp
        self._g1_check_reward_kernels()
        env_window = int(getattr(cfg, "library_env_window", 0) or 0)
        if env_window:
            self.g1_library.env_window_open(self.num_envs, env_window)
        self.g1_reset_sampler = G1ResetSampler(
            self.g1_library.length,
            num_envs=self.num_envs,
            discount=cfg.reset_discount,
            fail_share=cfg.reset_fail_share,
        )
        self._g1_exclude_resets(cfg.reset_exclude_names_file)
        self._g1_randomize_startup()

    def _g1_exclude_resets(self, names_file: str) -> None:
        """`reset_exclude_names_file`: keep the sampled resets out of the named clips."""
        if not names_file:
            return
        from pathlib import Path

        path = Path(names_file)
        if not path.is_absolute():
            # imitationlab/envs/isaaclab/envs -> repository root.
            path = Path(__file__).resolve().parents[4] / path
        names = [line.split("#")[0].strip() for line in path.read_text().splitlines()]
        names = [name for name in names if name]
        bank = self.expert_trajectory_motion_names()
        if len(bank) != int(self.g1_library.length.numel()):
            raise ValueError(
                f"{len(bank)} motion names for {int(self.g1_library.length.numel())} "
                "library clips; the names cannot be mapped to ranks."
            )
        rank_of = {name: rank for rank, name in enumerate(bank)}
        missing = [name for name in names if name not in rank_of]
        if missing:
            raise ValueError(
                f"{path}: {len(missing)} names are not in the bank, e.g. {missing[:3]}."
            )
        ranks = torch.tensor(
            sorted({rank_of[name] for name in names}),
            device=self.device,
            dtype=torch.long,
        )
        self.g1_reset_sampler.exclude(ranks)
        kept = int(self.g1_reset_sampler.bin_keep.sum())
        print(
            f"[INFO] Reset sampler excludes {ranks.numel()} clips from {path.name}: "
            f"{kept} of {self.g1_reset_sampler.num_bins} bins remain.",
            flush=True,
        )

    def _g1_check_reward_kernels(self):
        """Refuse a scale the shared MDP did not take; it would change nothing."""
        built = getattr(self.g1_mdp, "reward_kernel_scale", 1.0)
        if built != self.cfg.reward_kernel_scale:
            raise ValueError(
                f"env.reward_kernel_scale={self.cfg.reward_kernel_scale} but the "
                f"shared G1MDP was built with {built}."
            )

    def _make_tracking_task(self):
        from imitationlab.envs.isaaclab.task_model.g1_task import G1TrackingTask

        return G1TrackingTask(self, torch.device(self.device))

    def g1_cursor(self):
        return (
            self.trajectory_manager.env_traj_rank.to(self.device),
            self._current_reference_local_step,
        )

    def g1_state(self):
        data = self.robot.data
        quat = _tensor(data.root_quat_w)
        return torch.cat(
            (
                _tensor(data.root_pos_w) - self.scene.env_origins,
                quat[:, 3:4],
                quat[:, :3],
                _tensor(data.root_link_lin_vel_w),
                _tensor(data.root_link_ang_vel_w),
                _tensor(data.joint_pos).index_select(1, self._g1_joint_ids),
                _tensor(data.joint_vel).index_select(1, self._g1_joint_ids),
            ),
            -1,
        )

    def g1_body_state(self):
        """Live link-origin state, independent of the task model's FK."""
        data = self.robot.data
        p = _tensor(data.body_pos_w).index_select(1, self._g1_body_ids)
        q = _tensor(data.body_quat_w).index_select(1, self._g1_body_ids)
        R = quat_wxyz_to_matrix(torch.cat((q[..., 3:4], q[..., :3]), -1))
        v = _tensor(data.body_link_lin_vel_w).index_select(1, self._g1_body_ids)
        w = _tensor(data.body_link_ang_vel_w).index_select(1, self._g1_body_ids)
        return p - self.scene.env_origins[:, None, :], R, v, w

    def _g1_randomize_startup(self):
        """One friction and torso COM offset per environment, with no pushes."""
        ground_mu = 0.3
        if self.cfg.randomize_context:
            self._g1_context[:, 0].uniform_(0.3, 1.2)
            com_width = torch.tensor([0.025, 0.05, 0.05], device=self.device)
            self._g1_context[:, 30:33] = (
                2 * torch.rand(self.num_envs, 3, device=self.device) - 1
            ) * com_width
        else:
            ground_mu = float(self.cfg.clean_friction)
            self._g1_context[:, 0] = ground_mu
        torso_id = self.robot.find_bodies("torso_link")[0][0]
        com = _tensor(self.robot.data.body_com_pos_b)[:, torso_id].clone()
        self.robot.set_coms_index(
            coms=(com + self._g1_context[:, 30:33]).unsqueeze(1),
            body_ids=[torso_id],
        )
        manager = self.sim.physics_manager
        model = manager.backend.model
        mu_view = self.robot._root_view.get_attribute("shape_material_mu", model)[:, 0]
        wp.to_torch(mu_view)[:] = self._g1_context[:, :1]
        # MuJoCo combines friction with max(), so the ground is the range floor.
        shape_body = wp.to_torch(model.shape_body)
        wp.to_torch(model.shape_material_mu)[shape_body < 0] = ground_mu
        manager.add_model_change(ModelFlags.SHAPE_PROPERTIES)

    def _g1_observation_frame(self, state, action, rank, frame):
        return self.g1_mdp.proprio(
            state,
            action,
            self._g1_context,
            rank,
            frame,
            noise=bool(self.cfg.observation_noise),
        )

    def _g1_append_history(self):
        state = self.g1_state()
        action = self.action_manager.action
        rank, frame = self.g1_cursor()
        proprio = self._g1_observation_frame(state, action, rank, frame)
        # These are batched ring shifts, independent of the number of envs.
        self._g1_states = torch.cat((self._g1_states[:, 1:], state[:, None]), 1)
        self._g1_actions = torch.cat((self._g1_actions[:, 1:], action[:, None]), 1)
        self._g1_proprio = torch.cat((self._g1_proprio[:, 1:], proprio[:, None]), 1)

    def _g1_observations(self):
        """Every group, or only the one WMBPTT reads; no group draws randomness."""
        manager = self.observation_manager
        if self.cfg.wm_only_observations:
            return {WM_GROUP: manager.compute_group(WM_GROUP, update_history=False)}
        return manager.compute(update_history=False)

    #: ``reset_to_reference``'s starts while it runs: (ranks, local frames) of all envs.
    _g1_explicit_starts = None

    def reset_to_reference(self, ranks, frames):
        """Reset every env at the given (rank, local frame); returns ``reset()``'s.

        The evaluation probe (``wm.eprobe_share``) starts its worlds here. It is
        the ordinary full reset, so noise, context and observations follow the
        cfg; only the starts are given. Later resets of the episodes are drawn
        as the cfg says.
        """
        n = self.num_envs
        ranks = torch.as_tensor(ranks, device=self.device, dtype=torch.long)
        frames = torch.as_tensor(frames, device=self.device, dtype=torch.long)
        if ranks.shape != (n,) or frames.shape != (n,):
            raise ValueError(f"reset_to_reference needs {n} ranks and frames.")
        lengths = self.g1_library.length.to(self.device)
        if bool(((ranks < 0) | (ranks >= lengths.numel())).any()):
            raise ValueError("reset_to_reference: a rank outside the bank.")
        if bool(((frames < 0) | (frames > lengths[ranks] - 2)).any()):
            raise ValueError("reset_to_reference: a start needs a following frame.")
        self._g1_explicit_starts = (ranks, frames)
        try:
            return self.reset()
        finally:
            self._g1_explicit_starts = None

    def make_eval_probe(self, wm, run_dir):
        """``wm.eprobe_share``: the evaluation probe of a WMBPTT run on this env.

        Called by the agent at the start of training. The probe sweeps the
        deployed policy in a worker process and sets the reset sampler's focus.
        """
        from imitationlab.evaluation.world_model.g1_eval_probe import EvalProbe

        if self.cfg.pinned_reset:
            raise ValueError(
                "wm.eprobe_share focuses the sampled resets; pinned_reset samples none."
            )
        return EvalProbe(self, wm, run_dir)

    def _g1_episode_starts(self, env_ids):
        """Ranks, local frames and return bins for the episodes starting now."""
        starts = self._g1_explicit_starts
        if starts is not None:
            ranks, frames = starts
            return ranks[env_ids], frames[env_ids], None
        if not self.cfg.pinned_reset:
            return self.g1_reset_sampler.sample(env_ids.numel())
        selection = self.cfg.command_interface.reference.selection
        if selection.custom_fn is None:
            raise ValueError(
                "pinned_reset needs selection.custom_fn; pass --trajectory-ranks."
            )
        ranks = selection.custom_fn(env_ids, int(self.g1_library.length.numel()))
        ranks = ranks.to(device=self.device, dtype=torch.long)
        return ranks, torch.full_like(ranks, int(selection.start_frame)), None

    def _reset_idx(self, env_ids):
        if not hasattr(self, "g1_reset_sampler"):
            # Base construction does not reset episodes, but keep the manager
            # initialization path valid if Isaac Lab requests a scene reset.
            return ManagerBasedRLEnv._reset_idx(self, env_ids)
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return
        n = env_ids.numel()
        sampler = self.g1_reset_sampler
        # Initial/manual resets have no completed transition to credit.
        completed = sampler.ep_step[env_ids] > 0
        # The first reset precedes every step: no termination buffer exists yet,
        # and no row can be complete.
        if _FAST_MASKED_CREDIT and hasattr(self, "reset_terminated"):
            sampler.end_episodes(
                env_ids, self.reset_terminated[env_ids], completed=completed
            )
        else:
            credit_ids = env_ids[completed]
            if credit_ids.numel():
                sampler.end_episodes(credit_ids, self.reset_terminated[credit_ids])
        ranks, frames, bins = self._g1_episode_starts(env_ids)
        tm = self.trajectory_manager
        tm.reset_envs(
            env_ids.to(tm.state_device),
            ranks=ranks.to(tm.state_device),
            steps=frames.to(tm.state_device),
        )
        self._refresh_current_expert_frame(env_ids, advance=False)
        if self.g1_library.env_win_length:
            # The reset envs' chunks start at their new frame before any read of it.
            mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            mask[env_ids] = True
            self.g1_library.env_window_refill(
                mask,
                tm.env_traj_rank.to(self.device),
                tm.env_step.to(device=self.device, dtype=torch.long),
            )
        state = self.g1_library.ref_state(ranks, frames).clone()
        if self.cfg.reset_noise:

            def uniform(name):
                width = self._g1_reset_widths[name]
                return (
                    2 * torch.rand(n, width.numel(), device=self.device) - 1
                ) * width

            state[:, :3] += uniform("position")
            state[:, 3:7] = quat_mul(quat_exp(uniform("rotation")), state[:, 3:7])
            state[:, 7:10] += uniform("linear_velocity")
            state[:, 10:13] += uniform("angular_velocity")
            state[:, 13:42] += torch.empty(n, 29, device=self.device).uniform_(
                -0.1, 0.1
            )
        limits = _tensor(self.robot.data.joint_pos_limits)[0].index_select(
            0, self._g1_joint_ids
        )
        state[:, 13:42] = state[:, 13:42].clamp(limits[:, 0], limits[:, 1])
        # Offsets are independent calibration perturbations, redrawn per episode.
        if self.cfg.randomize_context:
            self._g1_context[env_ids, 1:30] = torch.empty(
                n, 29, device=self.device
            ).uniform_(-0.01, 0.01)
        ManagerBasedRLEnv._reset_idx(self, env_ids)
        quat = state[:, 3:7]
        pose = torch.cat(
            (state[:, :3] + self.scene.env_origins[env_ids], quat[:, 1:], quat[:, :1]),
            -1,
        )
        self.robot.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self.robot.write_root_link_velocity_to_sim_index(
            root_velocity=state[:, 7:13], env_ids=env_ids
        )
        self.robot.write_joint_position_to_sim_index(
            position=state[:, 13:42], joint_ids=self._g1_joint_ids, env_ids=env_ids
        )
        self.robot.write_joint_velocity_to_sim_index(
            velocity=state[:, 42:71], joint_ids=self._g1_joint_ids, env_ids=env_ids
        )
        self.robot.set_joint_position_target_index(
            target=state[:, 13:42], joint_ids=self._g1_joint_ids, env_ids=env_ids
        )
        action_term = self.action_manager.get_term("joint_pos")
        action_term._offset[env_ids] = (
            self._g1_default_q + self._g1_context[env_ids, 1:30]
        )
        sampler.begin_episodes(env_ids, ranks, frames, bins=bins)
        self._g1_states[env_ids] = state[:, None]
        self._g1_actions.index_fill_(0, env_ids, 0.0)
        proprio = self.g1_mdp.proprio(
            state,
            torch.zeros(n, 29, device=self.device),
            self._g1_context[env_ids],
            ranks,
            frames,
            noise=bool(self.cfg.observation_noise),
        )
        self._g1_proprio[env_ids] = proprio[:, None]
        self.expert_data_plane.reset_anchor_jitter(env_ids)
        self._reset_causal_planner_history(env_ids)
        self.expert_data_plane.clear_achieved_ring(env_ids)

    def step(self, action):
        """Score (s_t, a_t, s_t+1) against f+1 before any episode reset."""
        self.expert_data_plane.begin_next_reference()
        self.action_manager.process_action(action.to(self.device))
        self.recorder_manager.record_pre_step()
        is_rendering = self.sim.is_rendering
        if self._physics_handles_decimation:
            self._sim_step_counter += self.cfg.decimation
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            if (
                self._sim_step_counter % self.cfg.sim.render_interval == 0
                and is_rendering
            ):
                self.sim.render(skip_app_pumping=not self.render_enabled)
            self.scene.update(dt=self.step_dt)
        else:
            for _ in range(self.cfg.decimation):
                self._sim_step_counter += 1
                self.action_manager.apply_action()
                self.scene.write_data_to_sim()
                self.sim.step(render=False)
                self.recorder_manager.record_post_physics_decimation_step()
                if (
                    self._sim_step_counter % self.cfg.sim.render_interval == 0
                    and is_rendering
                ):
                    self.sim.render(skip_app_pumping=not self.render_enabled)
                self.scene.update(dt=self.physics_dt)
        self.episode_length_buf += 1
        self.common_step_counter += 1
        # Keep the existing async array delivery; only its installation moves
        # before reward/termination so both see the required f+1 reference.
        self.expert_data_plane.finish_next_reference(
            torch.empty(0, dtype=torch.long, device=self.device)
        )
        library = self.g1_library
        if library.env_win_length:
            # From here to the return every library read is one row per env at or
            # after the cursor: serve them from the per-env chunk.
            library.env_window_advance(*self.g1_cursor())
            library.env_window_activate(True)
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        self.g1_reset_sampler.accumulate(self.reward_buf)
        self._g1_append_history()
        # The WM collector needs actual terminal s_t+1, including its history.
        self.obs_buf = self._g1_observations()
        self.extras.pop("final_obs", None)
        self.extras.pop("final_info", None)
        if self.sim.consume_reset_request():
            # A UI reset truncates surviving episodes; it is not a tracking
            # failure. Capture all ending rows together before any reset can
            # replace their true post-step states or histories.
            self.reset_time_outs |= ~self.reset_buf
            self.reset_buf.fill_(True)
        reset_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if reset_ids.numel():
            self._publish_terminal_obs(reset_ids)
            recorder_active = len(self.recorder_manager.active_terms) > 0
            if recorder_active:
                self.recorder_manager.record_pre_reset(reset_ids.tolist())
            self._reset_idx(reset_ids)
            if recorder_active:
                self.recorder_manager.record_post_reset(reset_ids.tolist())
            if self.render_enabled and is_rendering and self.has_rtx_sensors:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()
        self.command_manager.compute(dt=self.step_dt)
        self._append_causal_planner_history()
        for recorder in self.video_recorders:
            recorder.step()
        self.obs_buf = self._g1_observations()
        if len(self.recorder_manager.active_terms) > 0:
            self.recorder_manager.record_post_step()
        log = self.extras.setdefault("log", {})
        log.update(
            {
                f"Reward/{key}": value.detach().mean()
                for key, value in self._g1_reward_parts.items()
            }
        )
        if library.env_win_length:
            library.env_window_activate(False)  # the agent reads the bank by rank
        return (
            self.obs_buf,
            self.reward_buf,
            self.reset_terminated,
            self.reset_time_outs,
            self.extras,
        )


class G1GraphRLEnv(G1RLEnv):
    """``env.step_mode`` masked or graph: the same MDP with mask-selected resets.

    ``step_mode=masked`` runs ``graph_step.g1.G1MaskedStepper`` eagerly and
    ``step_mode=graph`` replays it as CUDA graphs, bit for bit. Both draw their
    reset randomness per environment, a different stream from the eager step of
    the eager step. ``env.reset()`` keeps the eager reset.
    """

    g1_step_modes = ("masked", "graph")
    g1_step_mode_error = (
        "G1GraphRLEnv runs step_mode 'masked' or 'graph'; G1RLEnv is the eager step."
    )

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        # Refuse an unsupported setting now, not at the first step.
        self._g1_stepper_class().validate(self)

    def _g1_stepper_class(self):
        from imitationlab.envs.isaaclab.envs.graph_step.g1 import (
            G1GraphStepper,
            G1MaskedStepper,
        )

        return G1GraphStepper if self._step_mode == "graph" else G1MaskedStepper

    def step(self, action):
        return self._masked_step(action)

    def _masked_step(self, action):
        stepper = self._masked_stepper
        if stepper is None:
            stepper = self._g1_stepper_class()(self)
            self._masked_stepper = stepper
        return stepper.step(action)


class G1LatentRLEnv(G1RLEnv):
    """Load one frozen encoder for both live observations and model rollouts."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        from imitationlab.envs.isaaclab.task_model.g1_task import FrozenSkillEncoder

        if not cfg.skill_encoder_path:
            raise ValueError(
                "Set env.skill_encoder_path to a pretrained command encoder."
            )
        self.g1_skill = FrozenSkillEncoder(
            cfg.skill_encoder_path,
            torch.device(cfg.sim.device),
            latent_mode=cfg.skill_encoder_mode,
        )
        if (
            cfg.skill_encoder_sha256
            and cfg.skill_encoder_sha256 != self.g1_skill.weights_sha256
        ):
            raise ValueError(
                "The skill encoder SHA-256 does not match the pinned weights."
            )
        cfg.skill_encoder_sha256 = self.g1_skill.weights_sha256
        super().__init__(cfg, render_mode=render_mode, **kwargs)

    def _make_tracking_task(self):
        from imitationlab.envs.isaaclab.task_model.g1_task import (
            G1LatentTrackingTask,
        )

        return G1LatentTrackingTask(self, torch.device(self.device))

    def make_eval_probe(self, wm, run_dir):
        raise ValueError(
            "wm.eprobe_share sweeps the explicit-goal policy; the latent env has no probe."
        )


def make_imitation_g1_v1(cfg, **kwargs):
    """Select the environment class after the task's overrides are applied."""
    if cfg.skill_encoder_path:
        if cfg.step_mode != "eager":
            raise ValueError("The frozen command encoders require env.step_mode=eager.")
        return G1LatentRLEnv(cfg, **kwargs)
    if cfg.step_mode in ("masked", "graph"):
        return G1GraphRLEnv(cfg, **kwargs)
    return G1RLEnv(cfg, **kwargs)
