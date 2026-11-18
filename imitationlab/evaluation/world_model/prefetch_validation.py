"""Capture and compare the G1 env's prefetch modes without conflating math parity.

The caller constructs each environment in its own process. Capturing on CPU
synchronizes each step intentionally: this qualifies values, not throughput.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def _tensor_leaves(value, prefix=""):
    import torch

    for key, item in value.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(item, torch.Tensor):
            yield name, item
        elif isinstance(item, Mapping) or hasattr(item, "items"):
            yield from _tensor_leaves(item, name)


def capture_trace(env, *, steps: int, seed: int, action_std: float):
    """Record ordered inputs, reference rows, outputs and dense terminal rows."""
    import torch

    if steps <= 0 or env.num_envs <= 0:
        raise ValueError("Trace capture requires positive steps and environments.")
    if not getattr(env, "terminal_observation_is_post_step", False):
        raise ValueError("Prefetch qualification requires the G1 env lifecycle.")
    # A separate CPU generator keeps actions identical even if reset RNG use
    # diverges; such divergence then remains visible in the captured trace.
    actions = action_std * torch.randn(
        steps,
        env.num_envs,
        29,
        generator=torch.Generator(device="cpu").manual_seed(seed + 50001),
    )
    chunks = {}

    def add(name, tensor):
        chunks.setdefault(name, []).append(tensor.detach().cpu().clone().numpy())

    def snapshot(obs):
        for name, value in _tensor_leaves(obs):
            add(f"obs/{name}", value)
        for name, value in _tensor_leaves(env.current_expert_frame):
            add(f"reference/{name}", value)
        rank, frame = env.g1_cursor()
        add("cursor/rank", rank)
        add("cursor/frame", frame)
        add("cursor/episode_step", env.episode_length_buf)

    obs, _ = env.reset(seed=seed)
    snapshot(obs)
    with torch.no_grad():
        for action in actions:
            obs, reward, terminated, truncated, extras = env.step(action.to(env.device))
            add("transition/action", action)
            add("transition/reward", reward)
            add("transition/terminated", terminated)
            add("transition/truncated", truncated)
            for name, value in env._g1_reward_parts.items():
                add(f"reward_parts/{name}", value)
            mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            final = extras.get("final_obs")
            if final is not None:
                ids = final["_env_ids"].long()
                mask[ids] = True
                final_values = dict(_tensor_leaves(final["obs"]))
            else:
                ids, final_values = None, {}
            add("terminal/mask", mask)
            for name, value in _tensor_leaves(obs):
                dense = torch.zeros_like(value)
                if ids is not None:
                    dense[ids] = final_values[name]
                add(f"terminal/obs/{name}", dense)
            snapshot(obs)
    trace = {name: np.stack(values) for name, values in chunks.items()}
    metadata = {
        "seed": seed,
        "steps": steps,
        "num_envs": env.num_envs,
        "action_std": action_std,
        "requested_mode": env.cfg.data.resolved_reference_prefetch_mode(),
        "resolved_mode": env.expert_data_plane.reference_prefetch_mode,
        "reset_samples": int(trace["terminal/mask"].sum()),
        "terminated_samples": int(trace["transition/terminated"].sum()),
        "truncated_samples": int(trace["transition/truncated"].sum()),
        "scope": "same-seed random-action rollouts; observations include persistent noise",
    }
    return trace, metadata


def compare_traces(off, nxt, off_meta, next_meta, *, atol=1e-6):
    """Require exact data delivery and separately bound numeric rollout drift."""
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and non-negative.")
    failures = []
    for mode, meta in (("off", off_meta), ("next", next_meta)):
        if meta.get("requested_mode") != mode or meta.get("resolved_mode") != mode:
            failures.append(f"{mode}: requested/resolved prefetch mode mismatch")
        if meta.get("steps", 0) <= 0 or meta.get("num_envs", 0) <= 0:
            failures.append(f"{mode}: empty rollout metadata")
        if meta.get("reset_samples", 0) <= 0:
            failures.append(f"{mode}: no asynchronous reset evidence")
    for key in (
        "seed",
        "steps",
        "num_envs",
        "action_std",
        "reset_noise",
        "observation_noise",
        "step_dt",
    ):
        if off_meta.get(key) != next_meta.get(key):
            failures.append(f"Protocol differs: {key}")
    mandatory = {
        "reference/qpos",
        "cursor/rank",
        "cursor/frame",
        "cursor/episode_step",
        "obs/policy/features",
        "obs/wm/root_state",
        "obs/wm/joint_state",
        "obs/wm/last_action",
        "obs/wm/proprio_history",
        "obs/wm/sim_context",
        "terminal/obs/wm/root_state",
        "terminal/mask",
        "transition/action",
        "transition/reward",
        "transition/terminated",
        "transition/truncated",
    }
    for mode, trace in (("off", off), ("next", nxt)):
        missing = sorted(mandatory - set(trace))
        if missing:
            failures.append(f"{mode}: missing required trace fields: {missing}")
        meta = off_meta if mode == "off" else next_meta
        for name, value in trace.items():
            snapshots = name.startswith(("obs/", "reference/", "cursor/"))
            expected = (meta.get("steps", 0) + int(snapshots), meta.get("num_envs", 0))
            if np.asarray(value).shape[:2] != expected:
                failures.append(
                    f"{mode}: {name} has incomplete time/environment dimensions"
                )
    if set(off) != set(nxt):
        failures.append("Trace field sets differ")
    evidence_valid = not failures
    fields = {}
    for name in sorted(set(off) & set(nxt)):
        left, right = np.asarray(off[name]), np.asarray(nxt[name])
        exact_required = name.startswith(("reference/", "cursor/")) or name in {
            "transition/action",
            "transition/terminated",
            "transition/truncated",
            "terminal/mask",
            "obs/wm/sim_context",
            "terminal/obs/wm/sim_context",
        }
        record = {"exact_required": exact_required, "shape": list(left.shape)}
        fields[name] = record
        if left.shape != right.shape or left.dtype != right.dtype or left.size == 0:
            record["passed"] = False
            failures.append(f"{name}: shape/dtype differs or empty field")
            continue
        finite = bool(np.isfinite(left).all() and np.isfinite(right).all())
        exact = bool(np.array_equal(left, right))
        delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
        passed = finite and (exact if exact_required else bool((delta <= atol).all()))
        record.update(
            finite=finite,
            exact=exact,
            passed=passed,
            max_abs_error=float(delta.max()) if finite else None,
        )
        if not passed:
            bad = (~np.isfinite(delta)) | (delta > (0.0 if exact_required else atol))
            record["first_mismatch_index"] = np.argwhere(bad)[0].tolist()
            failures.append(f"{name}: trace mismatch")
    for mode, trace, meta in (("off", off, off_meta), ("next", nxt, next_meta)):
        if "terminal/mask" in trace:
            masks = np.asarray(trace["terminal/mask"])
            if int(masks.sum()) != meta.get("reset_samples") or not masks.any():
                failures.append(f"{mode}: reset count is missing or inconsistent")
                evidence_valid = False
            if {"transition/terminated", "transition/truncated"} <= set(trace):
                dones = trace["transition/terminated"] | trace["transition/truncated"]
                if not np.array_equal(masks, dones):
                    failures.append(
                        f"{mode}: terminal observations do not cover all done rows"
                    )
                    evidence_valid = False
    reference_fields = {
        name: value
        for name, value in fields.items()
        if name.startswith(("reference/", "cursor/"))
    }
    physics_fields = {
        name: value for name, value in fields.items() if name not in reference_fields
    }
    return {
        "passed": not failures,
        "reference_delivery_passed": evidence_valid
        and bool(reference_fields)
        and all(value["passed"] for value in reference_fields.values()),
        "physics_trace_passed": evidence_valid
        and bool(physics_fields)
        and all(value["passed"] for value in physics_fields.values()),
        "subresult_scope": {
            "reference_delivery": "All captured reference fields and logical cursors, exactly equal.",
            "physics_trace": "All other fields, including states, observations, actions, rewards and episode boundaries.",
        },
        "failures": failures,
        "atol": atol,
        "rtol": 0.0,
        "off": off_meta,
        "next": next_meta,
        "fields": fields,
        "all_fields_exact": bool(fields)
        and all(v.get("exact", False) for v in fields.values()),
        "scope": "Reference/cursor/action/flags/context equality is exact; other fields use atol.",
    }
