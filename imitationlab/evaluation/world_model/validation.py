"""Numerical gates for live environment / world-model parity measurements."""

from __future__ import annotations

import math


def comparison_stats(pairs):
    """Summarize paired analytic/live values without accepting constant mismatches."""
    import torch

    if pairs.numel() == 0 or not bool(torch.isfinite(pairs).all()):
        return {"n": int(pairs.shape[0]), "finite": False}
    mine, ref = pairs[:, 0], pairs[:, 1]
    mae = float((mine - ref).abs().mean())
    scale = float(ref.abs().mean())
    variable = bool(
        (ref.std(unbiased=False) > 1e-8) & (mine.std(unbiased=False) > 1e-8)
    )
    corr = float(torch.corrcoef(pairs.T)[0, 1]) if variable else None
    return {
        "mae": mae,
        "env_mean": float(ref.mean()),
        "mine_mean": float(mine.mean()),
        "rel_mae": mae / max(scale, 1e-9),
        "corr": corr,
        "constant": not variable,
        "finite": True,
        "n": int(pairs.shape[0]),
    }


def qualification_failures(
    report, *, tolerance, min_correlation, min_termination_agreement
):
    """Fail closed on missing comparisons, history drift and hard-rule mismatches."""
    failures = []
    if not report["terms"]:
        failures.append("no reward terms compared")
    for name, stats in report["terms"].items():
        if not stats.get("finite") or stats.get("n", 0) < 2:
            failures.append(f"{name}: missing or nonfinite reward samples")
            continue
        if stats["rel_mae"] > tolerance and stats["mae"] > 1e-7:
            failures.append(f"{name}: relative MAE {stats['rel_mae']:.6g}")
        corr = stats["corr"]
        if corr is not None and (not math.isfinite(corr) or corr < min_correlation):
            failures.append(f"{name}: correlation {corr:.6g}")
        if corr is None and stats["mae"] > 1e-7:
            failures.append(f"{name}: constant values differ")
    if (
        report["termination_samples"] == 0
        or report["termination_agreement"] < min_termination_agreement
    ):
        failures.append("hard termination agreement below threshold")
    if report["history_samples"] == 0 or report["history_max_abs_err"] > 1e-5:
        failures.append("replay history differs from environment history")
    if report.get("require_terminal_samples") and report["terminal_samples"] == 0:
        failures.append("no terminal transitions exercised")
    for name, error in report.get("skill_goal_max_err", {}).items():
        if name.endswith("_command_max_abs") and (
            not math.isfinite(error) or error > 1e-5
        ):
            failures.append(f"{name}: latent command differs from task")
    failures.extend(
        f"reward term not compared: {name}"
        for name in report.get("missing_reward_terms", [])
    )
    return failures


def frozen_replay_equivalence_stats(agent, out):
    """Compare every valid replay action and ordered actor input to collection.

    A zero-initialized dynamics model can reproduce states through its recorded
    residual even when later actions are wrong. Check the actor mapping itself,
    independently rebuilding each input from the recorded history and key.
    """
    import torch

    task = agent.task
    inputs = []
    with torch.no_grad():
        for i in range(out["actor_obs"].shape[1]):
            step = out["tau"] + i
            rows = agent.replay.history(step, out["env"], task.history)
            kwargs = {"noise_key": (agent._noise_seed, out["env"], step)}
            if getattr(task, "proprio_at", None) is not None:
                kwargs["episode_step"] = rows["episode_step"]
            inputs.append(
                task.policy_obs(
                    rows["hist_states"],
                    rows["hist_actions"],
                    rows["context"],
                    rows["rank"],
                    rows["frame"],
                    **kwargs,
                )
            )
        expected = torch.stack(inputs, 1)
        valid = out["inep"][:, :-1].bool()
        count = int(valid.sum())
        action_delta = (out["actions"] - out["recorded_action"])[valid]
        input_delta = (out["actor_obs"] - expected)[valid]
        finite = bool(
            torch.isfinite(action_delta).all() and torch.isfinite(input_delta).all()
        )
        action_error = float(action_delta.abs().max()) if count else float("inf")
        input_error = float(input_delta.abs().max()) if count else float("inf")
    return {
        "action_samples": count,
        "actor_input_samples": count,
        "max_action_error": action_error,
        "max_actor_input_error": input_error,
        "actor_comparison_finite": finite,
        "actor_comparison_passed": count > 0
        and finite
        and action_error < 1e-5
        and input_error < 1e-5,
    }


def frozen_replay_roundoff_diagnostics(agent, out):
    """Separate feature/history errors from numerical transition recurrence.

    Teacher forcing uses recorded states/actions at every step while retaining
    the replay's rolling proprio history. Three one-step comparisons then
    measure the state-coordinate round trip, residual cancellation with one
    dynamics batch shape, and the mixed batch shapes used by real replay.
    These are diagnostics only; they do not change any acceptance tolerance.
    """
    import torch
    from imitationlab.algorithms.common.world_model.task import (
        history_proprio,
        task_proprio,
    )

    task, rp = agent.task, agent.replay
    horizon = out["actor_obs"].shape[1]
    win = rp.window(out["tau"], out["env"], horizon + 1)
    start = rp.history(out["tau"], out["env"], task.history)
    ctx, rank, frame0 = start["context"], start["rank"], start["frame"]
    action_mask = out["inep"][:, :-1].bool()
    state_mask = out["inep"][:, 1:].bool()
    series = {}
    series_finite = {}

    def record(name, difference, valid):
        selected = difference[valid]
        finite = bool(torch.isfinite(selected).all())
        value = float(selected.abs().max()) if selected.numel() and finite else None
        series.setdefault(name, []).append(value)
        series_finite[name] = series_finite.get(name, True) and finite

    with torch.no_grad():
        prop = history_proprio(task, start, False)
        batch, _, state_dim = win["state"].shape
        recorded_states = win["state"][:, :horizon].reshape(-1, state_dim)
        recorded_actions = win["action"][:, :horizon].reshape(-1, task.action_dim)
        # Match the flattened residual-inference call in _real_rollout.
        flat_predictions = agent.dynamics(
            task.wm_input(
                recorded_states, recorded_actions, ctx.repeat_interleave(horizon, 0)
            ),
            bf16=agent.config.wm.bf16,
        ).reshape(batch, horizon, -1)
        for i in range(horizon):
            state, next_state = win["state"][:, i], win["state"][:, i + 1]
            action = win["action"][:, i]
            frame = frame0 + i
            key = (agent._noise_seed, out["env"], out["tau"] + i)
            kwargs = {}
            if getattr(task, "proprio_at", None) is not None:
                kwargs["episode_step"] = start["episode_step"] + i
            prop_noise = task.keyed_proprio_noise(*key, **kwargs)
            teacher_obs = task.policy_obs_from_parts(
                prop + prop_noise, state, rank, frame, task.keyed_goal_noise(*key)
            )
            rows = rp.history(out["tau"] + i, out["env"], task.history)
            expected_kwargs = {"noise_key": key}
            if getattr(task, "proprio_at", None) is not None:
                expected_kwargs["episode_step"] = rows["episode_step"]
            expected_obs = task.policy_obs(
                rows["hist_states"],
                rows["hist_actions"],
                rows["context"],
                rows["rank"],
                rows["frame"],
                **expected_kwargs,
            )
            teacher_action = (
                agent.actor(teacher_obs, bf16=agent.config.wm.bf16)
                + out["eta_recorded"][:, i]
            )
            record(
                "teacher_forced_actor_input",
                teacher_obs - expected_obs,
                action_mask[:, i],
            )
            record("teacher_forced_action", teacher_action - action, action_mask[:, i])
            record(
                "recursive_actor_input",
                out["actor_obs"][:, i] - expected_obs,
                action_mask[:, i],
            )
            record("recursive_action", out["actions"][:, i] - action, action_mask[:, i])
            record(
                "recursive_state",
                task.state_error(out["states"][:, i + 1], next_state),
                state_mask[:, i],
            )

            target = task.increment(state, next_state)
            prediction = agent.dynamics(
                task.wm_input(state, action, ctx), bf16=agent.config.wm.bf16
            )
            flat = flat_predictions[:, i]
            record("dynamics_batch_shape", prediction - flat, state_mask[:, i])
            for name, increment in (
                ("coordinate_roundtrip", target),
                ("same_batch_residual", prediction + (target - prediction)),
                ("mixed_batch_residual", prediction + (target - flat)),
            ):
                recovered = task.integrate(state, increment)
                record(name, task.state_error(recovered, next_state), state_mask[:, i])
            fresh = task_proprio(task, next_state, action, ctx, rank, frame + 1, False)
            prop = torch.cat((prop[:, 1:], fresh.unsqueeze(1)), 1)
    maxima = {
        name: max((v for v in values if v is not None), default=None)
        for name, values in series.items()
    }
    teacher_passed = int(action_mask.sum()) > 0 and all(
        series_finite[name] and maxima[name] is not None and maxima[name] < 1e-5
        for name in ("teacher_forced_actor_input", "teacher_forced_action")
    )
    first_action_error = series["recursive_action"][0]
    return {
        "scope": "Diagnostic only; teacher forcing removes state/action feedback, and transition checks restart from recorded states.",
        "horizon": horizon,
        "state_dtype": str(win["state"].dtype),
        "bf16": bool(agent.config.wm.bf16),
        "action_samples": int(action_mask.sum()),
        "transition_samples": int(state_mask.sum()),
        "teacher_forced_mapping_passed": teacher_passed,
        "first_action_matches": first_action_error is not None
        and first_action_error < 1e-5,
        "diagnostic_atol": 1e-5,
        "finite": series_finite,
        "max_abs_by_step": series,
        "max_abs": maxima,
    }


def frozen_policy_replay_check(agent, *, collect_steps: int = 80, batch: int = 64):
    """Qualify noise inference at identical, frozen collecting-policy weights.

    An updating actor's replay deviation is not an equivalence test. Freeze
    its observation normalizer as well as the weights, collect a fresh replay,
    and compare the differentiable replay against actual recorded transitions.
    This exercises reference-aware histories and persistent sensor-noise keys.
    The caller supplies a disposable qualification agent (fp32, eager).
    """
    from collections import defaultdict
    import torch

    if collect_steps <= agent.task.history + agent.config.wm.horizon:
        raise ValueError("collect_steps must cover both a history and a full rollout")
    agent.ensure_replay()
    obs, _ = agent._isaac.reset()
    stats = defaultdict(list)
    with torch.no_grad():
        for _ in range(agent.task.history):
            obs = agent._collect_step(obs, stats)
    # Freeze every part of the actor mapping, including its running statistics.
    agent.actor.obs_norm.frozen.fill_(True)
    agent.replay = None
    agent.ensure_replay()
    obs, _ = agent._isaac.reset()
    with torch.no_grad():
        for _ in range(collect_steps):
            obs = agent._collect_step(obs, stats)
    agent.dynamics.requires_grad_(False)
    agent.critic.requires_grad_(False)
    out = agent._real_rollout(batch, grad=True, track_noise=True)
    mask = out["inep"].bool()
    error = agent.task.state_error(out["states"][mask], out["recorded_states"][mask])
    grad = torch.autograd.grad(out["objective"].mean(), out["eta"])[0]
    result = {
        "action_dev0": float(out["action_dev0"]),
        "pelvis_dev_mm": float(out["pelvis_dev_mm"]),
        "max_state_error": float(error.detach().abs().max()),
        "state_samples": int(mask.sum()),
        "action_gradient_finite": bool(torch.isfinite(grad).all()),
        "action_gradient_norm": float(grad.norm()),
        "actor_normalization_frozen": bool(agent.actor.obs_norm.frozen),
    }
    result.update(frozen_replay_equivalence_stats(agent, out))
    result["roundoff_diagnostics"] = frozen_replay_roundoff_diagnostics(agent, out)
    result["passed"] = (
        result["action_dev0"] < 1e-5
        and result["pelvis_dev_mm"] < 0.1
        and result["max_state_error"] < 0.002
        and result["action_gradient_finite"]
        and result["action_gradient_norm"] > 0
        and result["actor_comparison_passed"]
    )
    return result
