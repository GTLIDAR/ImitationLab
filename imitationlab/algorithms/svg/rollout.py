"""Stochastic value gradients on replay windows with inferred transition noise."""

from __future__ import annotations

import torch
from torch import Tensor

from imitationlab.algorithms.common.returns import lambda_returns
from imitationlab.algorithms.common.world_model.task import (
    history_proprio,
    task_proprio,
)
from imitationlab.algorithms.svg.rebar import (
    clipped_score_surrogate,
    linearized_return,
    masked_center,
    shuffle_rows,
)


def real_window_masks(
    same: Tensor, terminated: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Episode bookkeeping of recorded windows.

    ``same`` (B, H): transition t stays in one episode; ``terminated`` (B, H):
    the env terminated (not timed out) on transition t. Returns ``inep``
    (B, H + 1), 1 while state t belongs to the window's first episode, and
    ``term`` / ``trunc`` (B, H), 1 on the transition that ends that episode by
    a termination / any other reset (timeout).
    """
    H = same.shape[1]
    s = same.float()
    inep = torch.cat((torch.ones_like(s[:, :1]), torch.cumprod(s, 1)), 1)
    end = inep[:, :H] * (1 - s)
    term = end * terminated.float()
    return inep, term, end - term


def real_window_targets(
    rewards: Tensor,
    env_reward: Tensor,
    values: Tensor,
    inep: Tensor,
    term: Tensor,
    trunc: Tensor,
    gamma: float,
    lam: float,
) -> Tensor:
    """TD(lambda) targets (B, H) over recorded windows (``real_window_masks``).

    Transitions inside the episode use ``rewards``; a terminating transition
    contributes the env's reward and no continuation; a timed-out transition
    is unknown (its next state is the reset row), so its state's value V(s_t)
    stands in for it. Entries past the episode are 0 and must not be fit.
    """
    H = rewards.shape[1]
    r = inep[:, 1:] * rewards + term * env_reward + trunc * values[:, :H]
    return lambda_returns(r, values, inep[:, 1:], gamma, lam)


class RealWindowUpdates:
    """Recorded-window steps and targets for the shared world-model learner."""

    def _real_reward(
        self,
        state: Tensor,
        action: Tensor,
        previous: Tensor,
        context: Tensor,
        rank: Tensor,
        frame: Tensor,
    ) -> Tensor:
        """Score valid recorded transitions with the learner's reward.

        Optional contact shaping uses the same task reward as imagination.
        Episode masks still come from the recording, and terminal transitions
        retain their measured environment reward because their stored next
        state is a reset row. The simulator's reward is never changed.
        """
        task = self.task
        reward = (
            task.reward_and_termination
            if self.config.wm.contact_reward > 0
            else task.reward
        )
        return reward(state, action, previous, context, rank, frame)[0]

    def _real_step(
        self,
        state: Tensor,
        prop: Tensor,
        prev_action: Tensor,
        ctx: Tensor,
        rank: Tensor,
        frame: Tensor,
        eta: Tensor,
        xi: Tensor,
        prop_noise: Tensor,
        goal_noise: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """One re-simulated recorded step: the current actor on the clean
        proprio history plus the collector's sensor noise of this row
        (``prop_noise``, ``goal_noise``) and the recorded action noise ``eta``,
        the model's output plus the recorded residual ``xi``. At the
        collecting policy this is the recorded transition."""
        t, wm = self.task, self.config.wm
        obs = t.policy_obs_from_parts(prop + prop_noise, state, rank, frame, goal_noise)
        action = self.actor(obs, bf16=wm.bf16) + eta
        prediction = self.dynamics(t.wm_input(state, action, ctx), bf16=wm.bf16)
        nxt = self._next_state(state, prediction + xi)
        r = self._real_reward(nxt, action, prev_action, ctx, rank, frame)
        new_prop = task_proprio(t, nxt, action, ctx, rank, frame + 1, False)
        prop = torch.cat((prop[:, 1:], new_prop.unsqueeze(1)), 1)
        return action, nxt, r, prop, obs

    def _real_step_fn(self):
        fn = getattr(self, "_compiled_real_step", None)
        if fn is None:
            fn = self._real_step
            if self.config.wm.compile:
                fn = torch.compile(self._real_step, dynamic=False)
            self._compiled_real_step = fn
        return fn

    def _real_rollout(
        self,
        batch: int,
        *,
        grad: bool,
        track_noise: bool = False,
        zero_noise: bool = False,
        starts: tuple[Tensor, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        """H-step recorded windows (``actor_source`` / ``critic_source`` real).

        With ``grad`` the window is re-simulated under the current actor with
        the recorded noise held fixed (SVG noise inference): the action noise
        eta_t = a_t - mu_old(o_t) of the collector, and the model residual
        xi_t = target(s_t, s_{t+1}) - f(s_t, a_t) in the model's output
        coordinates. The target uses plain increments or semi-implicit
        corrections according to the integrator. The
        forward pass equals the recording at the collecting policy and moves
        with the actor; the gradient runs through the model's Jacobians.
        Without ``grad`` only the recorded states and their analytic rewards
        are produced (critic data).
        """
        wm = self.config.wm
        t = self.task
        rp = self.replay
        assert rp is not None
        H, h = wm.horizon, t.history
        hi = rp.steps - 1 - H
        lo = max(rp.oldest + h - 1, hi - wm.real_window_recent_steps + 1)
        if hi < lo:
            msg = f"replay holds {rp.steps} steps; real windows of {H} need more."
            raise RuntimeError(msg)
        if starts is None:
            tau = rp.uniform_rows(batch, h - 1, H, recent=wm.real_window_recent_steps)
            env = torch.randint(0, self._num_envs, (batch,), device=self.device)
        else:
            tau, env = starts
        win = rp.window(tau, env, H + 1)
        es = win["episode_step"]
        same = es[:, 1:] == es[:, :-1] + 1
        inep, term, trunc = real_window_masks(same, win["terminated"][:, :H])
        env_reward = win["env_reward"][:, :H]
        ctx, rank, frame0 = win["context"][:, 0], win["rank"][:, 0], win["frame"][:, 0]
        steps = torch.arange(H + 1, device=self.device)
        frames = frame0.unsqueeze(1) + steps  # (B, H + 1), as imagination counts them
        if grad:
            with torch.no_grad():
                B, S = win["state"].shape[0], win["state"].shape[-1]
                s_now = win["state"][:, :H].reshape(B * H, S)
                s_next = win["state"][:, 1:].reshape(B * H, S)
                a_rec = win["action"][:, :H].reshape(B * H, -1)
                predicted = self.dynamics(
                    t.wm_input(s_now, a_rec, ctx.repeat_interleave(H, 0)), bf16=wm.bf16
                )
                target = (
                    t.structured_targets(s_now, s_next)  # type: ignore[attr-defined]
                    if self._structured
                    else t.increment(s_now, s_next)
                )
                xi = (target - predicted).reshape(B, H, -1) * same.unsqueeze(-1)
                eta = win["action"][:, :H] - win["policy_action"][:, :H]
            eta_recorded = eta
            if zero_noise:
                eta = torch.zeros_like(eta)  # the policy's mean path, residuals kept
            if track_noise:
                # d objective / d eta_t is the SVG action gradient of step t.
                eta = eta.detach().contiguous().requires_grad_(True)
            start = rp.history(tau, env, h)
            # Clean history; each step adds the collector's keyed noise of its row.
            prop = history_proprio(t, start, False)
            state, prev_action = win["state"][:, 0], start["hist_actions"][:, -1]
            # One requires_grad pattern for every compiled step (see _imagine).
            state, prop, prev_action = (
                x.detach().contiguous().requires_grad_(True)
                for x in (state, prop, prev_action)
            )
            step_fn = self._real_step_fn()
            states, rewards, actions, actor_obs = [state], [], [], []
            for i in range(H):
                with torch.no_grad():
                    key = (self._noise_seed, env, tau + i)
                    if getattr(t, "proprio_at", None) is not None:
                        prop_noise = t.keyed_proprio_noise(
                            *key, episode_step=start["episode_step"] + i
                        )
                    else:
                        prop_noise = t.keyed_proprio_noise(*key)
                    goal_noise = t.keyed_goal_noise(*key)
                prev_action, state, r, prop, obs_i = step_fn(
                    state,
                    prop,
                    prev_action,
                    ctx,
                    rank,
                    frames[:, i],
                    eta[:, i],
                    xi[:, i],
                    prop_noise,
                    goal_noise,
                )
                actor_obs.append(obs_i)
                states.append(state)
                rewards.append(r)
                actions.append(prev_action)
            states_t, rewards_t = torch.stack(states, 1), torch.stack(rewards, 1)
            with torch.no_grad():
                # How far the re-simulated window leaves the recording (task
                # state starts with the pelvis position), and its first action
                # the recorded one (zero at the collecting policy).
                dev = (states_t[..., :3] - win["state"][..., :3]).norm(dim=-1)
                dev_mm = 1000 * (dev * inep).sum() / inep.sum().clamp_min(1.0)
                act_dev0 = (actions[0] - win["action"][:, 0]).norm(dim=-1).mean()
        else:
            states_t = win["state"]
            B, S = states_t.shape[0], states_t.shape[-1]
            r = self._real_reward(
                states_t[:, 1:].reshape(B * H, S),
                win["action"][:, :H].reshape(B * H, -1),
                win["prev_action"][:, :H].reshape(B * H, -1),
                ctx.repeat_interleave(H, 0),
                rank.repeat_interleave(H, 0),
                frames[:, :H].reshape(-1),
            )
            rewards_t = r.reshape(B, H)
        out = {
            "states": states_t,
            "rewards": rewards_t,
            "env_reward": env_reward,
            "inep": inep,
            "term": term,
            "trunc": trunc,
            "ctx": ctx,
            "rank": rank,
            "frame0": frame0,
        }
        if grad:
            if track_noise:
                out["actions"] = torch.stack(actions, 1)
                out["eta"] = eta
                out["eta_recorded"] = eta_recorded
                out["actor_obs"] = torch.stack(actor_obs, 1).detach()
                out["recorded_states"] = win["state"]
                out["recorded_action"] = win["action"][:, :H]
                out["tau"], out["env"] = tau, env
            out["pelvis_dev_mm"] = dev_mm
            out["action_dev0"] = act_dev0
            gam = wm.gamma ** steps.float()
            objective = (gam[:H] * (inep[:, 1:] * rewards_t + term * env_reward)).sum(1)
            if wm.critic_bootstrap:
                values = self._critic_values(out)
                weights = torch.cat(
                    (gam[:H] * trunc, (gam[H] * inep[:, H]).unsqueeze(1)), 1
                )
                objective = objective + (weights * values).sum(1)
            out["objective"] = objective
        return out

    @torch.no_grad()
    def _recorded_actor_obs(self, tau: Tensor, env: Tensor, H: int) -> Tensor:
        """Actor inputs (B, H, D) of the recorded rows, with the collector's keyed
        sensor noise: the observations the recorded actions were sampled from."""
        rp, t = self.replay, self.task
        assert rp is not None
        rows = []
        for i in range(H):
            hist = rp.history(tau + i, env, t.history)
            kwargs = (
                {"episode_step": hist["episode_step"]}
                if getattr(t, "proprio_at", None) is not None
                else {}
            )
            rows.append(
                t.policy_obs(
                    hist["hist_states"],
                    hist["hist_actions"],
                    hist["context"],
                    hist["rank"],
                    hist["frame"],
                    noise_key=(self._noise_seed, env, tau + i),
                    **kwargs,
                )
            )
        return torch.stack(rows, 1)

    def _rebar_gradient(self, metrics: dict[str, list]) -> dict[str, Tensor]:
        """Fill the actor's ``.grad`` with mean-path SVG(H) plus the averaged
        score correction (``svg/rebar.py``); returns the rollout."""
        wm = self.config.wm
        assert self._rebar is not None
        params = [p for p in self.actor.net.parameters() if p.requires_grad]
        out = self._real_rollout(
            wm.imagine_batch, grad=True, track_noise=True, zero_noise=True
        )
        B, H = out["eta"].shape[:2]
        *svg, d_eta = torch.autograd.grad(
            -out["objective"].mean(), [*params, out["eta"]], allow_unused=True
        )
        svg_flat = torch.cat(
            [
                (g if g is not None else torch.zeros_like(p)).reshape(-1)
                for g, p in zip(svg, params)
            ]
        )
        disc = wm.gamma ** torch.arange(H, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            slopes = (
                -d_eta * B / disc.view(1, H, 1)
            )  # d J / d eta in the units of the advantage at t
            eta = out["eta_recorded"]
            action = out["recorded_action"]
            rec = self._real_rollout(B, grad=False, starts=(out["tau"], out["env"]))
            values = self._critic_values(rec)
            targets = real_window_targets(
                rec["rewards"],
                rec["env_reward"],
                values,
                rec["inep"],
                rec["term"],
                rec["trunc"],
                wm.gamma,
                wm.gae_lambda,
            )
            # The env clamps: on a clipped row a - mu is not the Gaussian draw.
            unclipped = ~action.abs().ge(wm.action_clip).any(-1)
            mask = rec["inep"][:, :H] * (1 - rec["trunc"]) * unclipped.float()
            adv = masked_center(targets - values[:, :H], mask)
            lin = linearized_return(slopes, eta, mask, wm.gamma * wm.gae_lambda)
            rho = adv - lin
            if wm.rebar_shuffle:
                rho = shuffle_rows(rho, mask > 0)
            obs = self._recorded_actor_obs(out["tau"], out["env"], H)
        mu = self.actor(obs.reshape(B * H, -1), bf16=wm.bf16).reshape(B, H, -1)
        mu_old = action - eta
        surrogate, clip_frac = clipped_score_surrogate(
            mu, mu_old, action, rho, disc * mask, wm.action_noise_std, wm.rebar_clip_eps
        )
        corr = torch.autograd.grad(-surrogate, params, allow_unused=True)
        corr_flat = torch.cat(
            [
                (g if g is not None else torch.zeros_like(p)).reshape(-1)
                for g, p in zip(corr, params)
            ]
        )
        applied, scale = self._rebar.update(corr_flat.detach(), svg_flat.detach())
        total = svg_flat + wm.rebar_weight * applied
        offset = 0
        for p in params:
            n = p.numel()
            p.grad = total[offset : offset + n].view_as(p).clone()
            offset += n
        with torch.no_grad():
            keep = mask > 0
            a_k, l_k = adv[keep], lin[keep]
            ac, lc = a_k - a_k.mean(), l_k - l_k.mean()
            metrics["rebar/corr_adv_noise_effect"].append(
                (ac * lc).mean() / (ac.std() * lc.std()).clamp_min(1e-12)
            )
            metrics["rebar/residual_var_ratio"].append(
                rho[keep].var() / a_k.var().clamp_min(1e-12)
            )
            metrics["rebar/adv_std"].append(a_k.std())
            metrics["rebar/noise_effect_std"].append(l_k.std())
            metrics["rebar/svg_norm"].append(svg_flat.norm())
            metrics["rebar/correction_norm"].append(corr_flat.norm())
            metrics["rebar/applied_norm"].append(applied.norm())
            metrics["rebar/cap_scale"].append(scale)
            metrics["rebar/cos_applied_svg"].append(
                (applied @ svg_flat)
                / (applied.norm() * svg_flat.norm()).clamp_min(1e-12)
            )
            metrics["rebar/ratio_clip_frac"].append(clip_frac)
            metrics["rebar/action_clip_frac"].append(1 - unclipped.float().mean())
            metrics["rebar/mu_dev"].append(
                (mu.detach() - mu_old).norm(dim=-1)[keep].mean()
            )
        return out

    def _update_critic_real(
        self, out: dict[str, Tensor], metrics: dict[str, list]
    ) -> None:
        wm = self.config.wm
        with torch.no_grad():
            states = out["states"].detach()
            B, H1 = states.shape[:2]
            H = H1 - 1
            cobs = self._critic_obs_seq({**out, "states": states})
            fit = (out["inep"][:, :H] * (1 - out["trunc"])).reshape(-1) > 0
            self.critic.obs_norm.update(
                cobs.reshape(B, H1, -1)[:, :H].reshape(B * H, -1)[fit]
            )
            values = self.critic(cobs, bf16=wm.bf16).reshape(B, H1)
            targets = real_window_targets(
                out["rewards"].detach(),
                out["env_reward"],
                values,
                out["inep"],
                out["term"],
                out["trunc"],
                wm.gamma,
                wm.gae_lambda,
            )
            cobs = cobs.reshape(B, H1, -1)[:, :H].reshape(B * H, -1)[fit]
            targets = targets.reshape(-1)[fit]
        self._fit_critic(cobs, targets, metrics)
