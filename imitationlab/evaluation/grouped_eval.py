"""Grouped replica evaluation: several checkpoints, each several times, in one simulator.

A replica evaluation plays C clips once from frame 0 in C envs. Its wall clock is
the longest clip's step count (13,653 steps on LAFAN1), and at 40 envs the step is
latency-bound, so more envs cost little more time. NDM Gym's grouped mode
(``ndm_gym/evaluate.py``, ``evaluate_group``; MIT, see ``THIRD_PARTY_NOTICES.md``)
puts several evaluations side by side in one simulator. This module is that mode
on our evaluator (``lowlevel/evaluate_checkpoint.py --grouped``).

Layout. A group is B blocks of C envs. Env ``e = b * C + c`` plays clip
``ranks[c]`` under the checkpoint of block b. The tree planner
(``imitationlab.evaluation.evaluate_checkpoint_tree --group G --repeats R``) orders blocks
checkpoint-major, so the R repeats of checkpoint m are blocks ``m * R + r``: NDM
Gym's world ``(m * repeats + r) * C + c``.

Policies. Each distinct checkpoint keeps its own copy of the deployed actor. The
shared task computes the policy observation once for all envs (row by row), and
each actor acts on its run of consecutive blocks (``BlockPolicy``), as NDM Gym's
``policy(hist, goal)`` closure does.

Repeats. A repeat is a separate block of envs that plays the same clips under the
same policy. What makes two repeats differ is whatever makes two single-checkpoint
runs differ: the replica protocol turns off observation noise, reset noise and
context randomization, and actions are the deterministic mode, so the difference
is the simulator's run-to-run variation. NDM Gym's repeats differ the same way
("the simulation is not bit-reproducible, so borderline clips can flip").

Accumulation. ``BlockAccumulator`` keeps, per env and on the device, every sum of
``evaluate_checkpoint``'s rollout loop: return, survival, done, terminated and
truncated events, termination terms, and the per-env and all-transition tracking
and action metrics, with the same masks. ``block_result`` then reduces one block
with the same formulas as the single-checkpoint summary, so each block's JSON has
the layout ``replica_readout``, ``time_to_target`` and the board readers read.
There is no host read per step, except one ``any`` every ``check_every`` steps.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

__all__ = [
    "BlockAccumulator",
    "BlockPolicy",
    "PolicyRun",
    "block_seed_indices",
    "block_slices",
    "grouped_rank_assignment",
    "policy_runs",
    "rollout",
]


def grouped_rank_assignment(ranks: Sequence[int], num_envs: int) -> list[int]:
    """The clip of every env: env ``e`` plays ``ranks[e % C]``.

    ``num_envs`` must be a positive multiple of the ``C = len(ranks)`` distinct,
    non-negative ranks; the quotient is the number of blocks.
    """
    ranks = [int(rank) for rank in ranks]
    if not ranks:
        raise ValueError("a grouped evaluation needs at least one trajectory rank.")
    if any(rank < 0 for rank in ranks):
        raise ValueError("trajectory ranks must be non-negative.")
    if len(set(ranks)) != len(ranks):
        raise ValueError("trajectory ranks must be unique.")
    clips = len(ranks)
    if int(num_envs) < clips or int(num_envs) % clips != 0:
        raise ValueError(
            f"num_envs={num_envs} is not a positive multiple of the {clips} ranks."
        )
    return [ranks[env % clips] for env in range(int(num_envs))]


def block_slices(blocks: int, clips: int) -> list[slice]:
    """The env slice of every block."""
    return [slice(b * clips, (b + 1) * clips) for b in range(int(blocks))]


def block_seed_indices(
    seed_indices: Sequence[int] | None, blocks: int
) -> list[int | None]:
    """The seed index of every block: the evaluation index that names its row.

    The tree planner passes one per block (``--grouped-seed-indices``); a rerun
    plans only the missing rows, so a block's index is not its order among the
    blocks of its checkpoint. Without the list every block's index is None.
    """
    if seed_indices is None:
        return [None] * int(blocks)
    indices = [int(index) for index in seed_indices]
    if len(indices) != int(blocks):
        raise ValueError(f"{len(indices)} seed indices for {blocks} blocks.")
    return indices


@dataclass(frozen=True)
class PolicyRun:
    """Consecutive blocks ``[first, end)`` that one policy acts on."""

    policy: int
    first: int
    end: int


def policy_runs(block_keys: Sequence[Any]) -> tuple[list[Any], list[PolicyRun]]:
    """The distinct keys in order of first appearance, and the runs of blocks.

    ``block_keys[b]`` names block b's policy (a checkpoint path). Consecutive
    blocks with one key form one run, so one actor call covers a checkpoint's
    repeats.
    """
    if not block_keys:
        raise ValueError("a grouped evaluation needs at least one block.")
    unique: list[Any] = []
    index: dict[Any, int] = {}
    runs: list[PolicyRun] = []
    for block, key in enumerate(block_keys):
        if key not in index:
            index[key] = len(unique)
            unique.append(key)
        policy = index[key]
        if runs and runs[-1].policy == policy and runs[-1].end == block:
            runs[-1] = PolicyRun(policy, runs[-1].first, block + 1)
        else:
            runs.append(PolicyRun(policy, block, block + 1))
    return unique, runs


class BlockPolicy(nn.Module):
    """Each policy acts on its own runs of blocks; the observation is shared.

    The forward is ``WMPolicy``'s (``algorithms/common/world_model/learner.py``) with one actor per
    run of blocks: ``actor(obs, bf16=...)`` plus the task's reference setpoint
    when that checkpoint is residual. ``task.policy_obs_from_env`` must be row by
    row, as the G1 env's is (each row reads its own env's history and its own clip's
    goal), so the rows a block's actor sees are exactly the ones a single run of
    its checkpoint would see.
    """

    def __init__(
        self,
        task: Any,
        actors: Sequence[nn.Module],
        runs: Sequence[PolicyRun],
        rows_per_block: int,
        *,
        bf16: Sequence[bool],
        residual: Sequence[bool],
    ) -> None:
        super().__init__()
        if len(bf16) != len(actors) or len(residual) != len(actors):
            raise ValueError("one bf16 and one residual flag per actor.")
        used = sorted({run.policy for run in runs})
        if used != list(range(len(actors))):
            raise ValueError(
                f"runs use policies {used}, but there are {len(actors)} actors."
            )
        if (
            any(run.first != prev.end for prev, run in zip(runs, runs[1:]))
            or runs[0].first != 0
        ):
            raise ValueError("runs must tile the blocks from block 0 without gaps.")
        self.task = task
        self.actors = nn.ModuleList(actors)
        self.bf16 = [bool(flag) for flag in bf16]
        self.residual = [bool(flag) for flag in residual]
        rows = int(rows_per_block)
        self.rows = [(run.policy, run.first * rows, run.end * rows) for run in runs]
        self.num_rows = runs[-1].end * rows

    def forward(self, *tensors: Tensor) -> Tensor:
        obs = self.task.policy_obs_from_env(*tensors)
        if obs.shape[0] != self.num_rows:
            raise RuntimeError(
                f"the policy observation has {obs.shape[0]} rows; the group has {self.num_rows} envs."
            )
        parts = []
        for policy, lo, hi in self.rows:
            rows = obs[lo:hi]
            action = self.actors[policy](rows, bf16=self.bf16[policy])
            if self.residual[policy]:
                action = action + self.task.reference_setpoint(rows)
            parts.append(action)
        return torch.cat(parts, 0)


class _Sums:
    """Per-env sums of one set of metrics that share a mask, as [N, k] tensors."""

    def __init__(
        self, num_envs: int, width: int, device: torch.device, per_env: bool
    ) -> None:
        zeros = lambda dtype: torch.zeros(num_envs, width, dtype=dtype, device=device)  # noqa: E731
        # All-transition statistics (``_accumulate_metric``): masked, not filtered.
        self.stat_sum = zeros(torch.float64)
        self.stat_sumsq = zeros(torch.float64)
        self.stat_count = zeros(torch.int64)
        # Per-env means (``_accumulate_per_env``): masked and finite only.
        self.env_sum = zeros(torch.float64) if per_env else None
        self.env_count = zeros(torch.int64) if per_env else None

    def add(self, values: Tensor, mask: Tensor) -> None:
        m = mask.unsqueeze(-1)
        zero = torch.zeros((), dtype=values.dtype, device=values.device)
        self.stat_sum += torch.where(m, values, zero)
        self.stat_sumsq += torch.where(m, values.square(), zero)
        self.stat_count += m.to(torch.int64)
        if self.env_sum is not None:
            valid = m & torch.isfinite(values)
            self.env_sum += torch.where(valid, values, zero)
            self.env_count += valid.to(torch.int64)


class BlockAccumulator:
    """Every per-env sum of ``evaluate_checkpoint``'s rollout loop, on the device.

    Per step: ``begin_step`` (the loop's ``step_active``), ``record_action`` (the
    action metrics, before ``env.step``) and ``record_transition`` (the step's
    outcome and the tracking metrics after it). The masks are the loop's:
    ``step_active`` for return, survival, events and action metrics, ``step_active
    & ~done`` for tracking metrics, and the loop's validity chains for the
    acceleration and jerk terms. ``keep_after_done`` is not supported: an env
    stops counting after its first done, as in the board protocol.
    """

    def __init__(
        self,
        num_envs: int,
        termination_terms: Sequence[str],
        *,
        device: torch.device | str,
        dt: float,
    ) -> None:
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.dt = float(dt)
        n, dev = self.num_envs, self.device
        self.active = torch.ones(n, dtype=torch.bool, device=dev)
        self.step_active = self.active.clone()
        self.return_sum = torch.zeros(n, dtype=torch.float32, device=dev)
        self.survival_steps = torch.zeros(n, dtype=torch.float32, device=dev)
        self.done_events = torch.zeros(n, dtype=torch.float32, device=dev)
        self.terminated_events = torch.zeros(n, dtype=torch.float32, device=dev)
        self.truncated_events = torch.zeros(n, dtype=torch.float32, device=dev)
        self.valid_transitions = torch.zeros(n, dtype=torch.int64, device=dev)
        self.final_root_drift = torch.zeros(n, dtype=torch.float64, device=dev)
        self.termination_terms = list(termination_terms)
        self.termination_hits = {
            name: torch.zeros(n, dtype=torch.bool, device=dev)
            for name in self.termination_terms
        }
        self._sums: dict[tuple[str, ...], _Sums] = {}
        self._names: list[str] = []  # first-seen order, as the loop's dicts
        self._prev_action: Tensor | None = None
        self._prev_prev_action: Tensor | None = None
        self._prev_body_lin_vel: tuple[Tensor, Tensor] | None = None
        self._prev_velocity_valid = torch.zeros(n, dtype=torch.bool, device=dev)
        self._prev_actual_acc: Tensor | None = None
        self._prev_acc_valid = torch.zeros(n, dtype=torch.bool, device=dev)

    # ------------------------------------------------------------ per step
    def any_active(self) -> bool:
        """One host read: whether any env still counts."""
        return bool(self.active.any())

    def begin_step(self) -> Tensor:
        self.step_active = self.active.clone()
        return self.step_active

    def _add(
        self, metrics: Mapping[str, Tensor], mask: Tensor, *, per_env: bool = True
    ) -> None:
        names = tuple(metrics)
        if not names:
            return
        values = torch.stack(
            [
                v.detach().reshape(self.num_envs, -1).mean(dim=-1)
                for v in metrics.values()
            ],
            -1,
        ).to(torch.float64)
        sums = self._sums.get(names)
        if sums is None:
            seen = [name for name in names if name in self._names]
            if seen:
                raise RuntimeError(
                    f"metrics {seen} arrived in another set; their sums would split."
                )
            sums = self._sums[names] = _Sums(
                self.num_envs, len(names), self.device, per_env
            )
            self._names.extend(names)
        sums.add(values, mask.to(self.device))

    def record_action(self, action: Tensor) -> None:
        """The action metrics of the loop: L2, first and second difference."""
        active = self.step_active
        action_2d = action.detach().reshape(self.num_envs, -1)
        self._add({"action_l2": torch.linalg.vector_norm(action_2d, dim=-1)}, active)
        if self._prev_action is not None:
            delta = torch.linalg.vector_norm(action_2d - self._prev_action, dim=-1)
            self._add({"action_delta_l2": delta}, active)
            if self.dt > 0.0:
                self._add({"action_rate_l2": delta / self.dt}, active, per_env=False)
            if self._prev_prev_action is not None:
                jerk = torch.linalg.vector_norm(
                    action_2d - 2.0 * self._prev_action + self._prev_prev_action, dim=-1
                )
                self._add({"action_jerk_l2": jerk}, active)
        self._prev_prev_action = self._prev_action
        self._prev_action = action_2d.clone()

    def record_transition(
        self,
        *,
        reward: Tensor,
        done: Tensor,
        terminated: Tensor,
        truncated: Tensor,
        terms: Mapping[str, Tensor],
        tracking: Mapping[str, Tensor],
        body_lin_vel: tuple[Tensor, Tensor] | None,
    ) -> None:
        """The step's outcome and tracking metrics; then ends the done envs."""
        n, dev = self.num_envs, self.device
        step_active = self.step_active
        done_any = (done.bool() | terminated.bool() | truncated.bool()).to(dev)
        terminal = done_any & step_active
        for name in self.termination_terms:
            self.termination_hits[name] |= (
                terms[name].reshape(-1)[:n].to(dev, torch.bool) & terminal
            )
        live = step_active.float()
        # A select, not ``reward * live``: a done env keeps stepping in its
        # auto-reset replay until the group's last env is done, and NaN * 0 is
        # NaN. NDM Gym masks the same way (``torch.where(alive, r, 0.0)``).
        self.return_sum += torch.where(
            step_active, reward.to(dev).float(), torch.zeros((), device=dev)
        )
        self.survival_steps += live
        self.done_events += terminal.float()
        self.terminated_events += (terminated.to(dev).bool() & step_active).float()
        self.truncated_events += (truncated.to(dev).bool() & step_active).float()

        metric_mask = step_active & ~done_any
        self.valid_transitions += metric_mask.to(torch.int64)
        self._add(dict(tracking), metric_mask)
        drift = tracking.get("root_pos_xyz_error_m")
        if drift is not None:
            now = drift.detach().reshape(n, -1)[:, 0].double().to(dev)
            self.final_root_drift = torch.where(
                metric_mask & torch.isfinite(now), now, self.final_root_drift
            )
        if body_lin_vel is not None and self.dt > 0.0:
            if self._prev_body_lin_vel is not None:
                actual, ref = body_lin_vel
                prev_actual, prev_ref = self._prev_body_lin_vel
                actual_acc = (actual - prev_actual) / self.dt
                ref_acc = (ref - prev_ref) / self.dt
                acc_mask = metric_mask & self._prev_velocity_valid
                self._add(
                    {
                        "tracking_acceleration_distance_mps2": torch.linalg.vector_norm(
                            actual_acc - ref_acc, dim=-1
                        ).mean(dim=-1),
                        "body_acc_mps2": torch.linalg.vector_norm(
                            actual_acc, dim=-1
                        ).mean(dim=-1),
                    },
                    acc_mask,
                )
                if self._prev_actual_acc is not None:
                    jerk = torch.linalg.vector_norm(
                        (actual_acc - self._prev_actual_acc) / self.dt, dim=-1
                    ).mean(dim=-1)
                    self._add({"body_jerk_mps3": jerk}, acc_mask & self._prev_acc_valid)
                self._prev_actual_acc = actual_acc.clone()
                self._prev_acc_valid = (
                    self._prev_velocity_valid & step_active & ~done_any
                )
            self._prev_body_lin_vel = (body_lin_vel[0].clone(), body_lin_vel[1].clone())
            self._prev_velocity_valid = step_active & ~done_any
        self.active &= ~done_any

    # ------------------------------------------------------------ reduction
    def _columns(self) -> dict[str, tuple[_Sums, int]]:
        where: dict[str, tuple[_Sums, int]] = {}
        for names, sums in self._sums.items():
            for column, name in enumerate(names):
                where[name] = (sums, column)
        return where

    def block_result(
        self,
        block: slice,
        *,
        tracking_failure_terms: Sequence[str],
        trajectory_ranks: Sequence[int],
        motion_names: Sequence[str],
        max_steps: int,
    ) -> dict[str, Any]:
        """One block's summary fields, with ``evaluate_checkpoint``'s formulas.

        Returns ``aggregate``, ``metrics``, ``successful_metrics``,
        ``per_environment``, ``max_steps``, ``steps_run`` and ``stop_reason``; the
        caller adds ``metadata``. ``env_id`` is the env's index in its block, so a
        block reads as a single run of its C clips.
        """
        cpu = lambda t: t[block].detach().cpu()  # noqa: E731
        return_sum, survival = cpu(self.return_sum), cpu(self.survival_steps)
        done_events, terminated_events = (
            cpu(self.done_events),
            cpu(self.terminated_events),
        )
        truncated_events = cpu(self.truncated_events)
        hits = {name: cpu(value) for name, value in self.termination_hits.items()}
        clips = int(return_sum.numel())
        if len(trajectory_ranks) != clips or len(motion_names) != clips:
            raise ValueError(
                "one trajectory rank and one motion name per env of the block."
            )
        columns = self._columns()
        env_sums = {
            name: (cpu(sums.env_sum)[:, col], cpu(sums.env_count)[:, col])
            for name, (sums, col) in columns.items()
            if sums.env_sum is not None
        }
        stat_sums = {
            name: (
                float(cpu(sums.stat_sum)[:, col].sum()),
                float(cpu(sums.stat_sumsq)[:, col].sum()),
                int(cpu(sums.stat_count)[:, col].sum()),
            )
            for name, (sums, col) in columns.items()
        }
        final_drift = cpu(self.final_root_drift)

        active_mask = survival > 0
        evaluated = int(active_mask.sum())
        tracking_failure = torch.zeros(clips, dtype=torch.bool)
        for name in tracking_failure_terms:
            tracking_failure |= hits[name]
        tracking_success = active_mask & ~tracking_failure
        reference_finished = hits.get(
            "reference_finished", torch.zeros(clips, dtype=torch.bool)
        )
        completed = tracking_success & reference_finished

        def mean_std(values: Tensor) -> tuple[float, float]:
            values = values.float()[active_mask]
            if values.numel() == 0:
                return float("nan"), float("nan")
            std = float(values.std(unbiased=False)) if values.numel() > 1 else 0.0
            return float(values.mean()), std

        def rate(flags: Tensor) -> float:
            return (
                float(flags[active_mask].float().mean())
                if evaluated > 0
                else float("nan")
            )

        def term_rate(name: str) -> float:
            if evaluated == 0 or name not in hits:
                return float("nan")
            return float(hits[name][active_mask].float().mean())

        names = [name for name in self._names if name in env_sums]
        successful_metrics: dict[str, dict[str, float | int]] = {}
        for name in names:
            sums, counts = env_sums[name]
            selected = completed & (counts > 0)
            count = int(counts[selected].sum())
            successful_metrics[name] = {
                "mean": float(sums[selected].sum() / count)
                if count > 0
                else float("nan"),
                "count": count,
                "num_successful_envs": int(selected.sum()),
            }
        metrics: dict[str, dict[str, float | int]] = {}
        for name, (total, total_sq, count) in sorted(stat_sums.items()):
            if count <= 0:
                continue
            mean = total / count
            metrics[name] = {
                "mean": mean,
                "std": max(0.0, total_sq / count - mean * mean) ** 0.5,
                "count": count,
            }

        return_mean, return_std = mean_std(return_sum)
        survival_mean, survival_std = mean_std(survival)
        steps_run = int(survival.max()) if clips else 0
        aggregate = {
            "return_sum_mean": return_mean,
            "return_sum_std": return_std,
            "survival_steps_mean": survival_mean,
            "survival_steps_std": survival_std,
            "done_rate": rate(done_events > 0),
            "terminated_rate": rate(terminated_events > 0),
            "truncated_rate": rate(truncated_events > 0),
            "done_events_per_env": float(done_events[active_mask].mean())
            if evaluated
            else float("nan"),
            "tracking_success_rate": rate(tracking_success),
            "tracking_failure_rate": rate(tracking_failure),
            "completed_tracking_success_rate": rate(completed),
            "completed_requested_horizon_rate": rate(done_events == 0),
            "reference_finished_rate": term_rate("reference_finished"),
            "time_out_rate": term_rate("time_out"),
            "termination_term_rates": {
                name: term_rate(name) for name in self.termination_terms
            },
            "termination_cause_env_counts": {
                name: int(hits[name][active_mask].sum())
                for name in self.termination_terms
            },
            "steps_executed": steps_run,
            "valid_transition_count": int(cpu(self.valid_transitions).sum()),
            "num_evaluated_envs": evaluated,
            "planner_publish_count": 0,
        }
        per_environment = []
        for env in range(clips):
            if not bool(active_mask[env]):
                continue
            tracked = {}
            counted = {}
            for name in names:
                sums, counts = env_sums[name]
                if int(counts[env]) > 0:
                    tracked[name] = float(sums[env]) / int(counts[env])
                    counted[name] = int(counts[env])
            tracked["root_pos_xyz_error_final_m"] = float(final_drift[env])
            per_environment.append(
                {
                    "env_id": env,
                    "trajectory_rank": int(trajectory_ranks[env]),
                    "motion_name": str(motion_names[env]),
                    "return_sum": float(return_sum[env]),
                    "survival_steps": int(survival[env]),
                    "done": bool(done_events[env] > 0),
                    "terminated": bool(terminated_events[env] > 0),
                    "truncated": bool(truncated_events[env] > 0),
                    "tracking_success": bool(tracking_success[env]),
                    "completed_tracking_success": bool(completed[env]),
                    "termination_terms": [
                        name for name in self.termination_terms if bool(hits[name][env])
                    ],
                    "tracking_metrics": tracked,
                    "tracking_metric_counts": counted,
                }
            )
        return {
            "reset_logs": None,
            "aggregate": aggregate,
            "metrics": metrics,
            "successful_metrics": successful_metrics,
            "max_steps": int(max_steps),
            "steps_run": steps_run,
            "stop_reason": "max_steps"
            if steps_run == int(max_steps)
            else "all_envs_done",
            "per_environment": per_environment,
        }


def _flat(
    td: Any,
    key: tuple[str, ...],
    num_envs: int,
    default: float | bool,
    device: torch.device,
) -> Tensor:
    """``_optional_flat_tensor`` of the evaluator, left on the device."""
    value = td.get(key, None)
    if not isinstance(value, Tensor):
        return torch.full((num_envs,), default, device=device)
    flat = value.detach().reshape(-1)
    if flat.numel() == 1 and num_envs > 1:
        flat = flat.expand(num_envs)
    if flat.numel() < num_envs:
        raise RuntimeError(
            f"expected at least {num_envs} values for {key}, got {flat.numel()}."
        )
    return flat[:num_envs]


def rollout(
    env: Any,
    td: Any,
    policy: Callable[[Any], Any],
    acc: BlockAccumulator,
    *,
    steps: int,
    read_terms: Callable[[], Mapping[str, Tensor]],
    tracking: Callable[[], tuple[Mapping[str, Tensor], tuple[Tensor, Tensor] | None]],
    check_every: int = 8,
) -> int:
    """``evaluate_checkpoint``'s rollout loop over every block at once.

    ``td`` is the observation after ``env.reset()``. The policy and ``env.step``
    run under ``torch.inference_mode`` and the bookkeeping outside it, as in the
    single-checkpoint loop. The loop stops once no env counts (checked every
    ``check_every`` steps) or after ``steps``. Every sum of values selects with
    the env's mask (``torch.where``) and every count adds a masked flag, so an env
    that no longer counts adds nothing, even a non-finite value from its
    auto-reset replay; the steps after the last env is done therefore change no
    sum. Returns the steps taken.
    """
    from torchrl.envs.utils import step_mdp  # noqa: PLC0415

    n, dev = acc.num_envs, acc.device
    taken = 0
    for step in range(int(steps)):
        if step % max(1, int(check_every)) == 0 and not acc.any_active():
            break
        acc.begin_step()
        with torch.inference_mode():
            td = policy(td)
        action = td.get("action")
        if action is None:
            raise RuntimeError("Policy did not write an 'action' tensor.")
        acc.record_action(action)
        with torch.inference_mode():
            td_step = env.step(td)
        metrics, body_lin_vel = tracking()
        acc.record_transition(
            reward=_flat(td_step, ("next", "reward"), n, 0.0, dev),
            done=_flat(td_step, ("next", "done"), n, False, dev),
            terminated=_flat(td_step, ("next", "terminated"), n, False, dev),
            truncated=_flat(td_step, ("next", "truncated"), n, False, dev),
            terms=read_terms(),
            tracking=metrics,
            body_lin_vel=body_lin_vel,
        )
        td = step_mdp(
            td_step, exclude_reward=True, exclude_done=False, exclude_action=True
        )
        taken = step + 1
    return taken
