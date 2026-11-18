"""Imagined BPTT rollouts with SuperMimic reward and value targets."""

from __future__ import annotations

import torch
from torch import Tensor

from imitationlab.algorithms.common.returns import lambda_returns
from imitationlab.algorithms.common.world_model.metrics import DEVICE_COUNTS
from imitationlab.algorithms.common.world_model.task import (
    history_proprio,
    task_proprio,
)


def score_function_surrogate(
    actions: Tensor,
    mus: Tensor,
    rewards: Tensor,
    values: Tensor,
    notdone: Tensor,
    alive: Tensor,
    std: float,
    gamma: float,
    lam: float,
    *,
    normalize: bool = True,
    actor_steps: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Likelihood-ratio surrogate of an imagined rollout, (B,).

    ``actions`` (B, H, A) were sampled as ``mu + std * eps``; ``mus`` carry the
    actor's gradient, everything else is treated as constant. Its gradient is
    ``sum_t gamma^t c_t A_t grad log N(a_t; mu_t, std^2)`` with ``c_t`` =
    ``alive`` (B, H), the probability the imagined episode is still running
    when step t is taken, and ``A_t`` = TD(lambda) target - V(s_t) from
    ``values`` (B, H + 1). ``actor_steps`` limits the sum to the first N steps
    (the actor-horizon ablation). Returns the surrogate and the advantages.
    """
    H = rewards.shape[1]
    with torch.no_grad():
        targets = lambda_returns(rewards, values, notdone, gamma, lam)
        adv = targets - values[:, :H]
        if normalize:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        disc = gamma ** torch.arange(H, device=rewards.device, dtype=rewards.dtype)
        w = disc * alive
        if actor_steps is not None and actor_steps < H:
            w = w * (torch.arange(H, device=w.device) < actor_steps)
    logp = -((actions.detach() - mus).square().sum(-1)) / (2 * std * std)
    return (w * adv * logp).sum(1), adv


class ImaginedRollouts:
    """Differentiable rollouts through the learned transition model."""

    def _imagine_step(
        self,
        state: Tensor,
        prop: Tensor,
        prev_action: Tensor,
        ctx: Tensor,
        rank: Tensor,
        frame: Tensor,
        eps: Tensor,
        detach_action: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """One imagined control step: act, predict, score, extend the history.

        ``eps`` (B, A) is the pre-scaled action noise (zeros for the
        deterministic actor); ``detach_action`` cuts the gradient between the
        actor's output and the model (score-function estimator). Returns the
        executed action and the actor's mean separately.
        """
        t, wm = self.task, self.config.wm
        obs = t.policy_obs_from_parts(prop, state, rank, frame)
        mu = self._act(self.actor, obs, wm.bf16)
        action = mu + eps
        if detach_action:
            action = action.detach()
        inc = self.dynamics(t.wm_input(state, action, ctx), bf16=wm.bf16)
        nxt = self._next_state(state, inc)
        r, d = t.reward_and_termination(nxt, action, prev_action, ctx, rank, frame)
        new_prop = task_proprio(
            t, nxt, action, ctx, rank, frame + 1, getattr(t, "sensor_noise", True)
        )
        prop = torch.cat((prop[:, 1:], new_prop.unsqueeze(1)), 1)
        return action, mu, nxt, r, d, prop

    def _step_fn(self):
        fn = getattr(self, "_compiled_step", None)
        if fn is None:
            fn = self._imagine_step
            if self.config.wm.compile:
                fn = torch.compile(self._imagine_step, dynamic=False)
            self._compiled_step = fn
        return fn

    def _filter_imagination_starts(
        self,
        tau: Tensor,
        env: Tensor,
        horizon: int,
        *,
        resample: bool,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, Tensor]:
        """Apply a task's optional ``valid_imagination_start`` capability.

        Eight rejection rounds keep the usual case small. If valid rows are
        scarce, enumerate the eligible ring once rather than loop forever
        or fail because random proposals missed them. Both paths sample
        uniformly over valid (row, env) pairs. Explicit probe starts fail
        validation instead of silently changing the requested experiment.
        """
        rp = self.replay
        assert rp is not None
        predicate = self.task.valid_imagination_start
        lo, hi = rp.oldest + self.task.history - 1, rp.steps - 1
        for attempt in range(8):
            slot = rp._slot(tau)
            valid = predicate(rp.rank[slot, env], rp.frame[slot, env], horizon)
            if valid.all():
                return tau, env
            if not resample:
                msg = f"Explicit imagination starts are invalid for horizon {horizon}."
                raise ValueError(msg)
            if attempt < 7:
                next_tau = torch.randint(
                    lo, hi + 1, tau.shape, device=self.device, generator=generator
                )
                next_env = torch.randint(
                    0,
                    self._num_envs,
                    env.shape,
                    device=self.device,
                    generator=generator,
                )
                tau = torch.where(valid, tau, next_tau)
                env = torch.where(valid, env, next_env)
        rows = torch.arange(lo, hi + 1, device=self.device)
        slot = rp._slot(rows)
        candidates = predicate(rp.rank[slot], rp.frame[slot], horizon).nonzero()
        if candidates.shape[0] == 0:
            msg = (
                f"No valid imagination starts for horizon {horizon} in replay rows {lo}..{hi}; "
                "collect eligible reference frames or reduce the requested horizon."
            )
            raise RuntimeError(msg)
        pick = torch.randint(
            0, candidates.shape[0], tau.shape, device=self.device, generator=generator
        )
        chosen = candidates[pick]
        return torch.where(valid, tau, rows[chosen[:, 0]]), torch.where(
            valid, env, chosen[:, 1]
        )

    def _device_starts(
        self, batch: int, horizon: int, generator: torch.Generator | None
    ) -> tuple[Tensor, Tensor]:
        """``wm.no_sync``: uniform valid starts with no host read. Of 2 * ``batch``
        uniform candidates, the first ``batch`` valid ones stay, which a stable sort
        finds. Their distribution is that of ``_filter_imagination_starts``, but the
        random stream differs. An invalid candidate survives only when fewer than
        ``batch`` are valid; ``_flush`` raises if one did."""
        rp = self.replay
        assert rp is not None
        if rp.steps_t is None:
            lo, hi = rp.oldest + self.task.history - 1, rp.steps - 1
            tau = torch.randint(
                lo, hi + 1, (2 * batch,), device=self.device, generator=generator
            )
        else:  # ``wm.update_graphs``: bounds from the device head, the same draw
            tau = rp.uniform_rows(2 * batch, self.task.history - 1, 0, generator)
        env = torch.randint(
            0, self._num_envs, (2 * batch,), device=self.device, generator=generator
        )
        slot = rp._slot(tau)
        invalid = ~self.task.valid_imagination_start(
            rp.rank[slot, env], rp.frame[slot, env], horizon
        )
        idx = torch.sort(invalid.to(torch.uint8), stable=True).indices[:batch]
        self._device_counts[DEVICE_COUNTS.index("imagine/invalid_starts")] += invalid[
            idx
        ].sum()
        return tau[idx], env[idx]

    def _imagine(
        self,
        batch: int,
        *,
        grad: bool,
        estimator: str | None = None,
        horizon: int | None = None,
        starts: tuple[Tensor, Tensor] | None = None,
        std: float | None = None,
        generator: torch.Generator | None = None,
        start_perturb: float = 0.0,
    ) -> dict[str, Tensor]:
        """Imagined H-step rollouts from replay states.

        ``estimator`` selects what ``objective`` differentiates to: ``pathwise``
        (deterministic actor, gradient through the model), ``reparam`` (actions
        ``mu + std eps``, gradient through the sample and the model) or
        ``score_function`` (same samples, gradient through ``log pi`` only;
        the model is traversed without gradient). None: the configured
        ``actor_estimator`` when ``grad``, else pathwise. ``horizon`` and
        ``starts`` (replay rows ``tau`` and env indices) override the config
        and the uniform start sampling (gradient probes); ``generator`` draws
        the action noise. ``start_perturb`` > 0 kicks the start states
        (``wm.imag_start_perturb``, which the actor update passes).
        """
        wm = self.config.wm
        t = self.task
        assert self.replay is not None
        H = int(horizon or wm.horizon)
        est = estimator or (wm.actor_estimator if grad else "pathwise")
        if est not in ("pathwise", "reparam", "score_function"):
            msg = f"unknown estimator {est!r}"
            raise ValueError(msg)
        sf = est == "score_function"
        sigma = float(wm.sf_action_std if std is None else std)
        h = t.history
        checked = getattr(t, "valid_imagination_start", None) is not None
        if starts is None and checked and wm.no_sync:
            tau, env = self._device_starts(batch, H, generator)
        else:
            if starts is None:
                if self.replay.steps_t is None:
                    lo = self.replay.oldest + h - 1
                    hi = self.replay.steps - 1
                    tau = torch.randint(
                        lo, hi + 1, (batch,), device=self.device, generator=generator
                    )
                else:  # ``wm.update_graphs``: bounds from the device head, the same draw
                    tau = self.replay.uniform_rows(batch, h - 1, 0, generator)
                env = torch.randint(
                    0, self._num_envs, (batch,), device=self.device, generator=generator
                )
            else:
                tau, env = starts
                batch = int(tau.shape[0])
            if checked:
                tau, env = self._filter_imagination_starts(
                    tau,
                    env,
                    H,
                    resample=starts is None,
                    generator=generator,
                )
        start = self.replay.history(tau, env, h)
        ctx, rank, frame = start["context"], start["rank"], start["frame"]
        noise = getattr(t, "sensor_noise", True)
        prop = history_proprio(t, start, noise)
        prev_action = start["hist_actions"][:, -1]
        state = start["hist_states"][:, -1]
        if start_perturb > 0:
            # After ``prop``: the history stays the recorded one, as in NDM Gym.
            state = t.perturb_start(state, start_perturb, generator)  # type: ignore[attr-defined]
        if wm.reference_window:
            # After ``prop``, which read the history's frames by bank rank. From here
            # ``rank`` is the rollout's window slot; the window spans the start frame,
            # H steps and the goal's look-ahead. ``_update_actor_critic`` closes it.
            lib = t.library  # type: ignore[attr-defined]
            if lib.win_active:
                lib.close_window()
            look = max(
                int(getattr(t, "goal_frames", 0) or 0),
                int(getattr(getattr(t, "skill", None), "frames", 0) or 0),
            )
            rank = lib.open_window(rank, frame, H + 1 + look)
        if grad:
            # Every step then sees the same requires_grad pattern; otherwise the
            # compiled step recompiles between step 0 and the rest.
            state, prop, prev_action = (
                x.detach().contiguous().requires_grad_(True)
                for x in (state, prop, prev_action)
            )
        zero_eps = torch.zeros(batch, t.action_dim, device=self.device)
        alive = torch.ones(batch, device=self.device)
        ret = torch.zeros(batch, device=self.device)
        states, rewards, dones, cont = [state], [], [], []
        actions, mus, alives = [], [], []
        step_fn = self._step_fn()
        actor_h = wm.actor_horizon if 0 < wm.actor_horizon < H else H
        actor_objective = None
        for step in range(H):
            if est == "pathwise":
                eps = zero_eps
            else:
                eps = sigma * torch.randn(
                    batch, t.action_dim, device=self.device, generator=generator
                )
            alives.append(alive)
            action, mu, nxt, r, d, prop = step_fn(
                state, prop, prev_action, ctx, rank, frame, eps, sf
            )
            actions.append(action)
            mus.append(mu)
            end = t.clip_end(rank, frame).float()
            if not wm.imagine_termination:
                # Ablation: the imagined episode survives every tracking
                # failure; only the clip end stops it. The soft termination
                # is still logged (``dones``) but carries no gradient and
                # no return cut, for the actor and the critic alike.
                d = d.detach() * 0
            elif not wm.termination_gradient:
                d = d.detach()  # ablation: the cut stays, the gradient does not
            ret = ret + (wm.gamma**step) * alive * r
            alive = alive * (1 - d) * (1 - end)
            states.append(nxt)
            rewards.append(r)
            dones.append(d)
            cont.append(1 - end)
            state, prev_action = nxt, action
            frame = frame + 1
            if not grad:
                state, prev_action, prop = (
                    state.detach(),
                    prev_action.detach(),
                    prop.detach(),
                )
            elif sf or (step + 1 == actor_h and actor_h < H):
                # Score function: the model is never differentiated, so the
                # path is cut after every step. Pathwise with a shorter actor
                # horizon: the actor's objective ends here (bootstrapped with
                # V); the remaining steps only feed the critic's targets.
                # Re-wrapping keeps the compiled step's requires_grad pattern
                # unchanged (no recompile).
                if not sf and step + 1 == actor_h:
                    if wm.critic_bootstrap:
                        v_a = self.critic(
                            self._critic_obs_fn()(state, ctx, rank, frame), bf16=wm.bf16
                        )
                        actor_objective = ret + (wm.gamma**actor_h) * alive * v_a
                    else:
                        actor_objective = ret
                state, prop, prev_action = (
                    x.detach().contiguous().requires_grad_(True)
                    for x in (state, prop, prev_action)
                )
        if wm.critic_bootstrap:
            v_last = self.critic(
                self._critic_obs_fn()(state, ctx, rank, frame), bf16=wm.bf16
            )
            objective = ret + (wm.gamma**H) * alive * v_last
        else:
            objective = ret
        if actor_objective is not None:
            objective = actor_objective
        out = {
            "objective": objective,
            "J": objective.detach(),
            "states": torch.stack(states, 1),
            "rewards": torch.stack(rewards, 1),
            "dones": torch.stack(dones, 1),
            "cont": torch.stack(cont, 1),
            "alive_H": alive,
            "actions": torch.stack(actions, 1),
            "mus": torch.stack(mus, 1),
            "alives": torch.stack(alives, 1),
            "ctx": ctx,
            "rank": rank,
            "frame0": start["frame"],
        }
        if sf and grad:
            with torch.no_grad():
                values = (
                    self._critic_values(out)
                    if wm.critic_bootstrap
                    else torch.zeros(batch, H + 1, device=self.device)
                )
                notdone = (1 - out["dones"].detach()) * out["cont"]
            surrogate, adv = score_function_surrogate(
                out["actions"],
                out["mus"],
                out["rewards"].detach(),
                values,
                notdone,
                out["alives"],
                sigma,
                wm.gamma,
                wm.gae_lambda,
                normalize=wm.sf_normalize_advantage,
                actor_steps=actor_h if actor_h < H else None,
            )
            out["objective"] = surrogate
            out["advantages"] = adv
        return out
