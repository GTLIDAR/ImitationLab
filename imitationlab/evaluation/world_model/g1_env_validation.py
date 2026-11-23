"""Live simulator qualification of the G1 env's non-reward contracts.

This audit is intentionally separate from training. It accepts an already
constructed environment, probes physical buffers and controlled episode
boundaries, and leaves the environment freshly reset. It does not launch a
simulator, step physics, or change the reference loading configuration.
"""

from __future__ import annotations

import json
import re
import random

import numpy as np
from typing import Any

import torch
from torch import Tensor

from imitationlab.envs.isaaclab.task_model.g1_kinematics import (
    G1_ISAACLAB_JOINT_NAMES,
    quat_wxyz_to_matrix,
)
from imitationlab.envs.isaaclab.task_model.rotations import so3_log


def _tensor(value) -> Tensor:
    return value.torch if hasattr(value, "torch") else value


def _nominal_pose_and_scale(device) -> tuple[Tensor, Tensor]:
    """Independent replica motor/pose table, not the environment's buffers."""
    motor = {
        "hip_pitch": (88.0, 0.010177520),
        "hip_roll": (139.0, 0.025101925),
        "hip_yaw": (88.0, 0.010177520),
        "knee": (139.0, 0.025101925),
        "ankle_pitch": (50.0, 0.007219450),
        "ankle_roll": (50.0, 0.007219450),
        "waist_yaw": (88.0, 0.010177520),
        "waist_roll": (50.0, 0.007219450),
        "waist_pitch": (50.0, 0.007219450),
        "shoulder_pitch": (25.0, 0.003609725),
        "shoulder_roll": (25.0, 0.003609725),
        "shoulder_yaw": (25.0, 0.003609725),
        "elbow": (25.0, 0.003609725),
        "wrist_roll": (25.0, 0.003609725),
        "wrist_pitch": (5.0, 0.00425),
        "wrist_yaw": (5.0, 0.00425),
    }
    default = {
        "hip_pitch": -0.312,
        "knee": 0.669,
        "ankle_pitch": -0.363,
        "elbow": 0.6,
        "shoulder_pitch": 0.2,
    }
    q0, scale = [], []
    frequency = 10 * 2 * 3.1415926535
    for name in G1_ISAACLAB_JOINT_NAMES:
        key = re.sub(r"^(left_|right_)", "", name).removesuffix("_joint")
        q = default.get(key, 0.0)
        if key == "shoulder_roll":
            q = 0.2 if name.startswith("left_") else -0.2
        q0.append(q)
        effort, armature = motor[key]
        scale.append(0.25 * effort / (armature * frequency**2))
    return torch.tensor(q0, device=device), torch.tensor(scale, device=device)


def _static_usd_torso_com(env) -> tuple[Tensor, float]:
    """Read nominal mass properties independently from the retained USD asset."""
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(env.cfg.scene.robot.spawn.usd_path))
    if stage is None:
        raise ValueError(
            "Could not open the configured robot USD for COM qualification."
        )
    for prim in stage.Traverse():
        if prim.GetName() == "torso_link":
            mass = UsdPhysics.MassAPI(prim)
            com = mass.GetCenterOfMassAttr().Get()
            value = mass.GetMassAttr().Get()
            if com is None or value is None:
                raise ValueError("Configured USD torso has no authored COM/mass.")
            return torch.tensor(tuple(com), device=env.device), float(value)
    raise ValueError("Configured robot USD has no torso_link prim.")


def qualify_g1_environment(
    env, *, reset_rounds: int = 3, seed: int = 137
) -> dict[str, Any]:
    """Check live physics/context, action targets, resets and episode boundaries.

    Call after rollout-based reward validation, outside any active training
    loop. This uses the actual Newton model, articulation setters and managers.
    It resets episodes repeatedly and probes large raw actions without stepping
    physics. Sampler statistics and the caller's RNG streams are restored at
    exit; the final simulator state is a fresh valid reset, not the incoming
    episode. Raises ``ValueError`` with a JSON report if any check fails.
    """
    if reset_rounds < 2:
        raise ValueError(
            "At least two reset rounds are required to test offset resampling."
        )
    if not getattr(env, "terminal_observation_is_post_step", False):
        raise ValueError(
            "Expected a G1 environment with post-step terminal observations."
        )
    import warp as wp

    report: dict[str, Any] = {
        "checks": {},
        "failures": [],
        "reset_rounds": reset_rounds,
        "prefetch_requested": env.cfg.data.resolved_reference_prefetch_mode(),
        "prefetch_resolved": env.expert_data_plane.reference_prefetch_mode,
    }
    checks = report["checks"]

    def check(name: str, condition, **details):
        passed = bool(condition)
        checks[name] = {"passed": passed, **details}
        if not passed:
            report["failures"].append(name)

    def maxerr(a: Tensor, b: Tensor) -> float:
        return float((a - b).abs().max())

    def close(name: str, actual: Tensor, expected: Tensor, tolerance=2e-5):
        error = maxerr(actual, expected)
        check(name, error <= tolerance, max_abs_error=error, tolerance=tolerance)

    device = torch.device(env.device)
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    sampler = env.g1_reset_sampler
    saved_sampler = {
        k: v.clone() for k, v in vars(sampler).items() if isinstance(v, Tensor)
    }
    joint_ids, joint_names = env.robot.find_joints(
        list(G1_ISAACLAB_JOINT_NAMES), preserve_order=True
    )
    joint_ids = torch.tensor(joint_ids, device=device, dtype=torch.long)
    ids = torch.arange(env.num_envs, device=device)
    check("named_joint_order", tuple(joint_names) == G1_ISAACLAB_JOINT_NAMES)
    q0, alpha = _nominal_pose_and_scale(device)
    close(
        "nominal_joint_pose",
        _tensor(env.robot.data.default_joint_pos)[:, joint_ids],
        q0[None].expand(env.num_envs, -1),
    )
    solver = env.cfg.sim.physics.solver_cfg
    declared = {
        "njmax": 500,
        "nconmax": 128,
        "cone": "pyramidal",
        "integrator": "implicitfast",
        "use_mujoco_contacts": True,
    }
    check(
        "solver_configuration",
        all(getattr(solver, k) == v for k, v in declared.items()),
        actual={k: getattr(solver, k) for k in declared},
        tolerance=float(solver.tolerance),
    )
    check(
        "control_and_substeps",
        env.step_dt == 0.02
        and env.cfg.decimation == 1
        and env.cfg.sim.physics.num_substeps == 4,
        control_dt=env.step_dt,
        decimation=env.cfg.decimation,
        substeps=env.cfg.sim.physics.num_substeps,
    )
    check(
        "timeout_configuration",
        env.max_episode_length == 500 and not env.cfg.is_finite_horizon,
        max_episode_length=env.max_episode_length,
    )

    manager = env.sim.physics_manager
    live_solver = manager.backend.solver
    opt = live_solver.mjw_model.opt

    def scalar(value):
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value).reshape(-1)[0].item()

    import mujoco

    actual_options = {
        "cone": int(scalar(opt.cone)),
        "integrator": int(scalar(opt.integrator)),
        "mujoco_contacts": bool(scalar(opt.run_collision_detection)),
        "tolerance": float(scalar(opt.tolerance)),
        "substep_dt": float(manager.backend.solver_dt),
        "num_substeps": int(manager.backend.num_substeps),
    }
    check(
        "applied_solver_options",
        actual_options["cone"] == int(mujoco.mjtCone.mjCONE_PYRAMIDAL)
        and actual_options["integrator"] == int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
        and actual_options["mujoco_contacts"]
        and abs(actual_options["tolerance"] - 1e-6) < 1e-10
        and abs(actual_options["substep_dt"] - 0.005) < 1e-9
        and actual_options["num_substeps"] == 4,
        actual=actual_options,
    )
    model = manager.backend.model
    view = env.robot._root_view
    context_before = env._g1_context.clone()
    friction = wp.to_torch(view.get_attribute("shape_material_mu", model)[:, 0])
    close(
        "backend_friction_matches_context",
        friction,
        context_before[:, :1].expand_as(friction),
        tolerance=1e-7,
    )
    ground = wp.to_torch(model.shape_body) < 0
    ground_friction = wp.to_torch(model.shape_material_mu)[ground]
    check(
        "ground_shapes_present",
        ground_friction.numel() > 0,
        count=ground_friction.numel(),
    )
    if ground_friction.numel():
        close(
            "ground_friction",
            ground_friction,
            torch.full_like(ground_friction, 0.3),
            tolerance=1e-7,
        )
    check(
        "friction_range",
        bool(((context_before[:, 0] >= 0.3) & (context_before[:, 0] <= 1.2)).all()),
        minimum=float(context_before[:, 0].min()),
        maximum=float(context_before[:, 0].max()),
    )
    torso_id = env.robot.find_bodies("torso_link")[0][0]
    backend_torso = int(env.robot.map_body_ids_to_backend([torso_id])[0])
    com = wp.to_torch(view.get_attribute("body_com", model)[:, 0])[:, backend_torso]
    nominal_com, nominal_mass = _static_usd_torso_com(env)
    close(
        "backend_torso_com_matches_context", com, nominal_com + context_before[:, 30:33]
    )
    mass = wp.to_torch(view.get_attribute("body_mass", model)[:, 0])[:, backend_torso]
    close(
        "backend_torso_mass_not_randomized", mass, torch.full_like(mass, nominal_mass)
    )
    report["retained_asset_nominal_torso"] = {
        "com": nominal_com.tolist(),
        "mass": nominal_mass,
    }
    com_width = torch.tensor([0.025, 0.05, 0.05], device=device)
    check(
        "com_offset_range",
        bool((context_before[:, 30:33].abs() <= com_width + 1e-7).all()),
        maximum_absolute=context_before[:, 30:33].abs().amax(0).tolist(),
    )

    def restore_sampler():
        for name, value in saved_sampler.items():
            getattr(sampler, name).copy_(value)

    def install_reference(
        frame: Tensor, episode_step: int, *, altered_state: Tensor | None = None
    ):
        rank = env.trajectory_manager.env_traj_rank.to(device)
        tm = env.trajectory_manager
        tm.set_env_cursor(
            env_ids=ids.to(tm.state_device),
            ranks=rank.to(tm.state_device),
            steps=frame.to(tm.state_device),
        )
        env._refresh_current_expert_frame(advance=False)
        state = (
            env.g1_library.ref_state(rank, frame).clone()
            if altered_state is None
            else altered_state
        )
        quat = state[:, 3:7]
        pose = torch.cat(
            (state[:, :3] + env.scene.env_origins, quat[:, 1:], quat[:, :1]), -1
        )
        env.robot.write_root_link_pose_to_sim_index(root_pose=pose)
        env.robot.write_root_link_velocity_to_sim_index(root_velocity=state[:, 7:13])
        env.robot.write_joint_position_to_sim_index(
            position=state[:, 13:42], joint_ids=joint_ids
        )
        env.robot.write_joint_velocity_to_sim_index(
            velocity=state[:, 42:71], joint_ids=joint_ids
        )
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.scene.update(0.0)
        env.episode_length_buf.fill_(episode_step)
        env.termination_manager.compute()
        return (
            env.termination_manager.terminated.clone(),
            env.termination_manager.time_outs.clone(),
        )

    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    with torch.random.fork_rng(devices=devices):
        try:
            maxima = {
                name: 0.0
                for name in (
                    "position",
                    "rotation",
                    "linear_velocity",
                    "angular_velocity",
                    "joint_velocity",
                )
            }
            last_offset = None
            offsets_change = True
            limits = _tensor(env.robot.data.joint_pos_limits)[0, joint_ids]
            for iteration in range(reset_rounds):
                obs, _ = env.reset(seed=seed + iteration)
                state = env.g1_state()
                rank, frame = env.g1_cursor()
                reference = env.g1_library.ref_state(rank, frame)
                widths = (
                    ("position", state[:, :3] - reference[:, :3], [0.05, 0.05, 0.01]),
                    (
                        "rotation",
                        so3_log(
                            quat_wxyz_to_matrix(state[:, 3:7])
                            @ quat_wxyz_to_matrix(reference[:, 3:7]).transpose(-1, -2)
                        ),
                        [0.1, 0.1, 0.2],
                    ),
                    (
                        "linear_velocity",
                        state[:, 7:10] - reference[:, 7:10],
                        [0.5, 0.5, 0.2],
                    ),
                    (
                        "angular_velocity",
                        state[:, 10:13] - reference[:, 10:13],
                        [0.52, 0.52, 0.78],
                    ),
                )
                for name, difference, width in widths:
                    maxima[name] = max(maxima[name], float(difference.abs().max()))
                    bound = (
                        torch.tensor(width, device=device)
                        if env.cfg.reset_noise
                        else torch.zeros(len(width), device=device)
                    )
                    check(
                        f"reset_{iteration}_{name}_range",
                        bool((difference.abs() <= bound + 3e-5).all()),
                        maximum_absolute=difference.abs().amax(0).tolist(),
                    )
                noise = 0.1 if env.cfg.reset_noise else 0.0
                lower = (reference[:, 13:42] - noise).clamp(limits[:, 0], limits[:, 1])
                upper = (reference[:, 13:42] + noise).clamp(limits[:, 0], limits[:, 1])
                check(
                    f"reset_{iteration}_joint_position_range",
                    bool(
                        (
                            (state[:, 13:42] >= lower - 2e-5)
                            & (state[:, 13:42] <= upper + 2e-5)
                        ).all()
                    ),
                )
                close(
                    f"reset_{iteration}_joint_velocity_unchanged",
                    state[:, 42:],
                    reference[:, 42:],
                )
                close(
                    f"reset_{iteration}_stored_history_matches_live",
                    env._g1_states,
                    state[:, None].expand_as(env._g1_states),
                )
                close(
                    f"reset_{iteration}_previous_action_zero",
                    env._g1_actions,
                    torch.zeros_like(env._g1_actions),
                    0.0,
                )
                close(
                    f"reset_{iteration}_action_manager_zero",
                    env.action_manager.action,
                    torch.zeros_like(env.action_manager.action),
                    0.0,
                )
                close(
                    f"reset_{iteration}_noisy_history_repeats_first_frame",
                    env._g1_proprio,
                    env._g1_proprio[:, -1:].expand_as(env._g1_proprio),
                    0.0,
                )
                clean = env.g1_mdp.proprio(
                    state,
                    torch.zeros(env.num_envs, 29, device=device),
                    env._g1_context,
                    rank,
                    frame,
                    noise=False,
                )
                sensor_width = (
                    env.g1_mdp.noise_width
                    if env.cfg.observation_noise
                    else torch.zeros(99, device=device)
                )
                check(
                    f"reset_{iteration}_sensor_noise_range",
                    bool(
                        (
                            (env._g1_proprio[:, -1] - clean).abs()
                            <= sensor_width + 2e-5
                        ).all()
                    ),
                )
                close(
                    f"reset_{iteration}_policy_uses_persistent_history",
                    obs["policy"]["features"][:, :990],
                    env._g1_proprio.flatten(1),
                    0.0,
                )
                offset = env._g1_context[:, 1:30]
                check(
                    f"reset_{iteration}_joint_offset_range",
                    bool((offset.abs() <= 0.0100001).all()),
                )
                if last_offset is not None:
                    offsets_change &= not torch.equal(offset, last_offset)
                last_offset = offset.clone()
            check("joint_offsets_resampled", offsets_change)
            close(
                "startup_friction_unchanged_across_resets",
                env._g1_context[:, :1],
                context_before[:, :1],
                0.0,
            )
            close(
                "startup_com_unchanged_across_resets",
                env._g1_context[:, 30:33],
                context_before[:, 30:33],
                0.0,
            )
            report["reset_maxima"] = maxima

            # Capture live targets after invoking the real action manager.
            action = (
                torch.linspace(-21.0, 21.0, 29, device=device)
                .expand(env.num_envs, -1)
                .clone()
            )
            env.action_manager.process_action(action)
            env.action_manager.apply_action()
            actual_target = _tensor(env.robot.data.joint_pos_target)[:, joint_ids]
            expected_target = q0 + env._g1_context[:, 1:30] + alpha * action
            close("live_action_target_mapping", actual_target, expected_target, 2e-6)
            report["action_probe_raw_range"] = [-21.0, 21.0]

            # Restore sampler statistics before both seed-identical resets.
            outputs = []
            for _ in range(2):
                restore_sampler()
                env.reset(seed=seed + 1000)
                outputs.append(
                    (
                        env.g1_state().clone(),
                        env._g1_proprio.clone(),
                        env._g1_context.clone(),
                        env.g1_cursor()[0].clone(),
                        env.g1_cursor()[1].clone(),
                    )
                )
            for name, first, second in zip(
                ("state", "proprio", "context", "rank", "frame"), outputs[0], outputs[1]
            ):
                close(f"seeded_reset_reproducible_{name}", first, second, 0.0)

            frame0 = torch.zeros(env.num_envs, dtype=torch.long, device=device)
            term, trunc = install_reference(frame0, 499)
            check(
                "step499_is_not_timeout", not bool(term.any()) and not bool(trunc.any())
            )
            term, trunc = install_reference(frame0, 500)
            check(
                "step500_is_truncation_not_terminal",
                not bool(term.any()) and bool(trunc.all()),
            )
            rank, _ = env.g1_cursor()
            last = env.g1_library.length[rank] - 1
            term, trunc = install_reference(last - 1, 1)
            check(
                "penultimate_reference_not_truncated",
                not bool(term.any()) and not bool(trunc.any()),
            )
            term, trunc = install_reference(last, 2)
            check(
                "last_reference_is_truncation_not_terminal",
                not bool(term.any()) and bool(trunc.all()),
            )
            reference = env.g1_library.ref_state(rank, frame0).clone()
            high = reference.clone()
            high[:, 2] += 0.26
            term, trunc = install_reference(frame0, 1, altered_state=high)
            check(
                "height_failure_is_terminal", bool(term.all()) and not bool(trunc.any())
            )
            tilted = reference.clone()
            ref_zz = quat_wxyz_to_matrix(reference[:, 3:7])[:, 2, 2]
            tilted[:, 3:7] = 0.0
            tilted[ref_zz <= 0, 3] = 1.0
            tilted[ref_zz > 0, 4] = 1.0
            term, trunc = install_reference(frame0, 1, altered_state=tilted)
            check(
                "tilt_failure_is_terminal", bool(term.all()) and not bool(trunc.any())
            )
        finally:
            # Do not credit synthetic audit boundaries or leave probe targets.
            restore_sampler()
            sampler.ep_step.zero_()
            env.reset(seed=seed + 2000)
            final_bins = sampler.start_bin.clone()
            restore_sampler()
            # Incoming sampler episode buffers refer to different physical
            # states; begin fresh episodes without changing learned bin stats.
            rank, frame = env.g1_cursor()
            sampler.begin_episodes(ids, rank, frame, bins=final_bins)
            random.setstate(python_rng_state)
            np.random.set_state(numpy_rng_state)
    report["passed"] = not report["failures"]
    if report["failures"]:
        raise ValueError(
            "G1 env live qualification failed:\n" + json.dumps(report, indent=2)
        )
    return report
