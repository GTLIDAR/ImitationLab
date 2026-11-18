"""Return-based reset sampling of ``imitation-g1-v1``, independent of the simulator runtime.

Reimplemented from the MIT-licensed ``zita-ch/supermimic_vibed`` at
``0525a23`` (``env.py`` and ``reset_sampling.py``). This deliberately does
not alter ``ReturnBasedResetSampler``: this sampler estimates raw
discounted returns with a truncated-episode bootstrap, and its probability
memory is an exponential average updated on each sampling call.

One boundary correction is intentional. The replica creates a singleton
last-frame bin for clips of length ``50*k + 1`` and can start on the final
frame, then index into the following clip. We retain that bin and its
probability, but select the penultimate frame. Its returns are still
credited to the sampled bin. Clips must have at least two frames.

``fail_share`` adds NDM Gym's ``--fail_reset`` (``ndm_gym/sim/rigid/reset_stats.py``
at ``05ea9f3``, itself ported from BlinkMimic and InstantTrack). It is off by
default, and the MIT notices are in ``third_party/licenses/ndm_gym.txt``. Each bin
keeps the terminating share of the episodes that ended in it, and a share of the
resets goes one bin before those failure points. The update precedes the same
step's reset, so a failure reaches the very next draw under
``reference_prefetch_mode=next`` or ``off``; the task rejects ``next_and_reset``.

The ``*_masked`` methods are the same model over all environments, selected by
a reset mask, for the captured step (``envs/graph_step/g1.py``). They follow
NDM Gym's ``--reset_masked`` (``bin_ema_w`` and ``sample_frames(N, gate)``): the
memory advances only on a step where an episode ended, as one ``sample`` call
per resetting step does. Draws come from caller-supplied uniforms, and the CDF
and return sums are int64 fixed point, so every process gets the same numbers.

``enable_focus`` adds NDM Gym's evaluation-guided focus (``--eprobe_share``,
``mix_focus``, ``set_focus`` and ``use_static_focus`` in ``ndm_gym/sim/rigid``
at ``05ea9f3``, from BlinkMimic): a share of the distribution drawn from a focus
over bins, after the failure share. The focus and its share are fixed device
tensors from the start, written in place by ``set_focus``, so a captured reset
follows them. Before the first focus the share is 0, which leaves every draw bit
for bit as it was; without ``enable_focus`` no focus code runs.

``exclude`` removes clips from the sampled resets (``env.reset_exclude_names_file``):
their bins take probability zero in every term of the distribution, the softmax
included, so the kept bins keep the 0.5 / 0.4 / 0.1 mixture among themselves. A
masked product alone would not do: an unvisited bin holds ``bin_G = 0`` and would
take most of the softmax share. Without ``exclude`` no draw changes.
"""

from __future__ import annotations

import torch
from torch import Tensor

BIN_FRAMES = 50
MAX_STEP_REWARD = 0.1
#: Checkpoint key of the reset statistics; absent unless `persist_reset_stats`.
RESET_STATS_KEY = "reset_stats"


class G1ResetSampler:
    """Replica reset distribution and batched episode-return bookkeeping.

    ``sample`` returns trajectory ranks, local frame indices, and sampled bin
    IDs. Pass the bin IDs to ``begin_episodes`` to preserve return credit when
    the singleton-bin boundary correction applies. Explicit resets may omit
    them. ``accumulate`` receives all environments' dt-scaled step rewards;
    ``end_episodes`` receives a terminal flag for each selected environment,
    excluding timeouts and reference ends from that flag.
    """

    def __init__(
        self,
        lengths: Tensor,
        *,
        num_envs: int,
        discount: float = 0.99,
        generator: torch.Generator | None = None,
        fail_share: float = 0.0,
    ) -> None:
        if lengths.ndim != 1 or lengths.numel() == 0:
            raise ValueError("lengths must be a non-empty one-dimensional tensor.")
        if (
            lengths.is_floating_point()
            or lengths.is_complex()
            or lengths.dtype == torch.bool
        ):
            raise ValueError("lengths must contain integer frame counts.")
        if bool(torch.any(lengths < 2)):
            raise ValueError("Reset clips require at least two frames.")
        if int(num_envs) <= 0:
            raise ValueError("num_envs must be positive.")
        if not 0.0 < float(discount) < 1.0:
            raise ValueError("discount must lie strictly between zero and one.")
        if not 0.0 <= float(fail_share) < 1.0:
            raise ValueError("fail_share must lie in [0, 1).")

        self.lengths = lengths.to(dtype=torch.long)
        self.trajectory_lengths = self.lengths
        self.device = lengths.device
        self.num_envs = int(num_envs)
        self.discount = float(discount)
        self.generator = generator
        self.G_max = MAX_STEP_REWARD / (1.0 - self.discount)
        self.fail_share = float(fail_share)

        counts = (self.lengths + BIN_FRAMES - 1) // BIN_FRAMES
        self.bin_offsets = torch.cumsum(counts, 0) - counts
        self.num_bins = int(counts.sum())
        self.bin_rank = torch.repeat_interleave(
            torch.arange(len(self.lengths), device=self.device), counts
        )
        self.bin_start = (
            torch.arange(self.num_bins, device=self.device)
            - self.bin_offsets[self.bin_rank]
        ) * BIN_FRAMES
        # Half-open frame range; the last recorded frame is never a normal
        # start, because a transition requires a following reference frame.
        self.bin_end = torch.minimum(
            self.bin_start + BIN_FRAMES, self.lengths[self.bin_rank] - 1
        )
        self.bins = torch.stack((self.bin_rank, self.bin_start, self.bin_end), -1)
        self.bin_G = torch.zeros(self.num_bins, device=self.device)
        self.bin_w_avg = torch.full_like(self.bin_G, 1.0 / self.num_bins)

        self.start_bin = torch.full(
            (self.num_envs,), -1, device=self.device, dtype=torch.long
        )
        self.ep_step = torch.zeros_like(self.start_bin)
        self.ep_disc_ret = torch.zeros(self.num_envs, device=self.device)
        self.ep_ret = torch.zeros_like(self.ep_disc_ret)
        self.last_return = torch.zeros_like(self.ep_disc_ret)
        if self.fail_share > 0.0:
            # Each bin's previous bin in its clip; a clip's first bin is its own.
            previous = torch.arange(self.num_bins, device=self.device) - 1
            same_clip = (previous >= 0) & (
                self.bin_rank[previous.clamp_min(0)] == self.bin_rank
            )
            self.bin_prev = torch.where(same_clip, previous, previous + 1)
            self.bin_fail = torch.zeros_like(self.bin_G)
            self.start_frame = torch.full_like(self.start_bin, -1)
        #: The evaluation-guided focus (``enable_focus``); None while off.
        self.focus_q: Tensor | None = None
        self.focus_share: Tensor | None = None
        #: Bins that may be drawn (``exclude``); None while every bin may.
        self.bin_keep: Tensor | None = None

    # ------------------------------------------------------------------
    # Excluded clips.
    # ------------------------------------------------------------------

    def exclude(self, ranks: Tensor) -> None:
        """Never start an episode in the clips ``ranks``.

        Call before any draw or CUDA-graph capture. The probability memory
        restarts uniform over the kept bins.
        """
        ranks = torch.as_tensor(ranks, device=self.device, dtype=torch.long)
        if ranks.numel() == 0:
            return
        if bool(((ranks < 0) | (ranks >= self.lengths.numel())).any()):
            raise ValueError("exclude: a rank outside the bank.")
        drop = torch.zeros(self.lengths.numel(), device=self.device, dtype=torch.bool)
        drop[ranks] = True
        keep = ~drop[self.bin_rank]
        if not bool(keep.any()):
            raise ValueError("exclude: no clip would remain.")
        self.bin_keep = keep
        self._bin_uniform = keep.to(self.bin_G.dtype) / keep.sum()
        self.bin_w_avg.copy_(self._bin_uniform)

    def _soft(self) -> Tensor:
        logits = -25.0 * self.bin_G / self.G_max
        if self.bin_keep is not None:
            logits = logits.masked_fill(~self.bin_keep, float("-inf"))
        return torch.softmax(logits, dim=0)

    def _base(self, soft: Tensor) -> Tensor:
        """0.5 uniform + 0.4 softmax + 0.1 memory, over the kept bins."""
        if self.bin_keep is None:
            return (
                0.5 / self.num_bins
                + 0.4 * soft
                + 0.1 * self.bin_w_avg / self.bin_w_avg.sum()
            )
        return (
            0.5 * self._bin_uniform
            + 0.4 * soft
            + 0.1 * self.bin_w_avg / self.bin_w_avg.sum()
        )

    def _kept(self, probs: Tensor) -> Tensor:
        """Zero the excluded bins after the failure and focus shares, renormalized."""
        if self.bin_keep is None:
            return probs
        probs = probs * self.bin_keep
        return probs / probs.sum()

    # ------------------------------------------------------------------
    # Evaluation-guided focus (NDM Gym's ``--eprobe_share``).
    # ------------------------------------------------------------------

    def enable_focus(self) -> None:
        """Hold a focus over bins in fixed device tensors, share 0 until set.

        A CUDA graph that captured a draw reads these tensors, so call this
        before any capture; ``set_focus`` then writes them in place. Calling it
        again keeps the tensors and their values.
        """
        if self.focus_q is not None:
            return
        self.focus_q = torch.full(
            (self.num_bins,), 1.0 / self.num_bins, device=self.device
        )
        self.focus_share = torch.zeros((), device=self.device)

    def set_focus(self, q: Tensor, share: float) -> None:
        """Draw ``share`` of the resets from the distribution ``q`` over bins.

        In place and without a host read of the device (``fill_`` takes the
        Python float), so a captured reset follows it at its next replay.
        """
        if self.focus_q is None:
            raise RuntimeError("set_focus needs enable_focus first.")
        if q.shape != self.focus_q.shape:
            raise ValueError(
                f"focus over {tuple(q.shape)} bins; the sampler has {self.num_bins}."
            )
        if not 0.0 <= float(share) < 1.0:
            raise ValueError(f"focus share must lie in [0, 1), got {share}.")
        self.focus_q.copy_(q)
        self.focus_share.fill_(float(share))

    def _mix_focus(self, probs: Tensor) -> Tensor:
        """``(1 - s) probs + s q``: NDM Gym's ``mix_focus``. At s = 0 this is
        ``probs`` bit for bit (1 * p + 0 * q = p for finite, non-negative q)."""
        share = self.focus_share
        return (1 - share) * probs + share * self.focus_q

    def state_dict(self) -> dict[str, Tensor]:
        """The statistics a later walltime segment resumes from.

        Episodes in flight are not kept: every segment starts with a full reset,
        and those partial returns would have no transition to continue from.
        """
        state = {
            "lengths": self.lengths,
            "bin_G": self.bin_G,
            "bin_w_avg": self.bin_w_avg,
        }
        if self.fail_share > 0.0:
            state["bin_fail"] = self.bin_fail
        return {name: value.detach().clone() for name, value in state.items()}

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        """Restore ``state_dict``; a bank with other clip lengths is refused.

        A segment that turns ``fail_share`` on starts ``bin_fail`` at zero, which
        is what no recorded failure means; one that turns it off ignores it.
        """
        if not torch.equal(state["lengths"].to(self.device), self.lengths):
            raise ValueError("Reset statistics belong to a bank with other clips.")
        self.bin_G.copy_(state["bin_G"])
        self.bin_w_avg.copy_(state["bin_w_avg"])
        if self.fail_share > 0.0 and "bin_fail" in state:
            self.bin_fail.copy_(state["bin_fail"])

    def _mix_failures(self, probs: Tensor) -> Tensor:
        """Move ``fail_share`` of ``probs`` to the bins before recent failures.

        Each bin's termination rate goes to its previous bin. The distribution
        is unchanged while no bin has one.
        """
        fail = torch.zeros_like(self.bin_fail).index_add_(
            0, self.bin_prev, self.bin_fail
        )
        fail = fail / fail.sum().clamp_min(1e-6)
        share = self.fail_share
        return (1 - share) * probs + share * torch.where(fail.sum() > 0, fail, probs)

    def sampling_probabilities(self) -> Tensor:
        """Inspect the current distribution without advancing its memory."""
        probs = self._base(self._soft())
        if self.fail_share > 0.0:
            probs = self._mix_failures(probs)
        if self.focus_q is not None:
            probs = self._mix_focus(probs)
        return self._kept(probs)

    def sample(self, count: int) -> tuple[Tensor, Tensor, Tensor]:
        """Draw ranks, local frames and bin IDs; advance memory once per call."""
        count = int(count)
        if count < 0:
            raise ValueError("count must be non-negative.")
        if count == 0:
            empty = torch.empty(0, device=self.device, dtype=torch.long)
            return empty, empty.clone(), empty.clone()
        soft = self._soft()
        self.bin_w_avg.mul_(0.999).add_(0.001 * soft)
        probs = self._base(soft)
        if self.fail_share > 0.0:
            probs = self._mix_failures(probs)
        if self.focus_q is not None:
            probs = self._mix_focus(probs)
        probs = self._kept(probs)
        bins = torch.multinomial(
            probs, count, replacement=True, generator=self.generator
        )
        ranks, start, end = self.bins[bins].unbind(-1)
        frame = (
            start
            + (
                torch.rand(count, device=self.device, generator=self.generator)
                * (end - start).float()
            ).long()
        )
        frame = torch.minimum(frame, self.lengths[ranks] - 2)
        return ranks, frame, bins

    def bin_ids(self, ranks: Tensor, frames: Tensor) -> Tensor:
        """Map valid explicit reset positions to their ordinary 50-frame bins."""
        return self.bin_offsets[ranks] + frames // BIN_FRAMES

    def begin_episodes(
        self,
        env_ids: Tensor,
        ranks: Tensor,
        frames: Tensor,
        *,
        bins: Tensor | None = None,
    ) -> None:
        """Record fresh starts after a reset; discard earlier partial returns."""
        self.start_bin[env_ids] = self.bin_ids(ranks, frames) if bins is None else bins
        # `index_fill_`, not `x[ids] = 0`: a scalar index write uploads the
        # scalar and waits for the device.
        self.ep_step.index_fill_(0, env_ids, 0)
        self.ep_disc_ret.index_fill_(0, env_ids, 0.0)
        self.ep_ret.index_fill_(0, env_ids, 0.0)
        if self.fail_share > 0.0:
            self.start_frame[env_ids] = frames

    def accumulate(self, reward: Tensor) -> None:
        """Accumulate one transition for every active episode, on the device."""
        active = self.start_bin >= 0
        self.ep_disc_ret.add_(
            torch.where(active, self.discount ** self.ep_step.float() * reward, 0.0)
        )
        self.ep_ret.add_(torch.where(active, reward, 0.0))
        self.ep_step.add_(active.long())

    def end_episodes(
        self, env_ids: Tensor, terminated: Tensor, *, completed: Tensor | None = None
    ) -> Tensor:
        """Update bins from the selected finished episodes, then mark them idle.

        Truncation bootstraps the empirical mean reward to an infinite discounted
        tail. Duplicate bin observations are averaged before applying the fixed
        0.8 old / 0.2 new EMA, including the first observation of a bin. Returns
        are not clipped or normalized before updating ``bin_G``.

        ``completed`` is an optional mask over ``env_ids``. Rows outside it leave
        every statistic unchanged, as if they had not been passed, so the caller
        needs no host-side selection; their entries in the result are not returns.
        """
        steps = self.ep_step[env_ids].float()
        boot = torch.where(
            terminated,
            0.0,
            (self.discount**steps)
            * (self.ep_ret[env_ids] / steps.clamp_min(1.0))
            / (1.0 - self.discount),
        )
        returns = self.ep_disc_ret[env_ids] + boot
        bins = self.start_bin[env_ids]
        # An explicit reset before its first transition carries no evidence.
        # Non-finite episode returns are omitted instead of poisoning every
        # future multinomial draw; finite rows keep the exact replica update.
        valid = (bins >= 0) & (steps > 0) & torch.isfinite(returns)
        if completed is not None:
            valid = valid & completed
        safe_bins = bins.clamp_min(0)
        sums = torch.zeros_like(self.bin_G)
        counts = torch.zeros_like(self.bin_G)
        sums.index_add_(0, safe_bins, torch.where(valid, returns, 0.0))
        counts.index_add_(0, safe_bins, valid.to(counts.dtype))
        average = sums / counts.clamp_min(1.0)
        self.bin_G.copy_(
            torch.where(counts > 0, 0.8 * self.bin_G + 0.2 * average, self.bin_G)
        )
        if self.fail_share > 0.0:
            ended = (bins >= 0) & (steps > 0)
            if completed is not None:
                ended = ended & completed
            self._update_failures(env_ids, terminated, bins, ended)
        if completed is None:
            self.last_return[env_ids] = returns
            self.start_bin.index_fill_(0, env_ids, -1)
        else:
            self.last_return[env_ids] = torch.where(
                completed, returns, self.last_return[env_ids]
            )
            self.start_bin[env_ids] = torch.where(completed, -1, bins)
        return returns

    def _update_failures(
        self, env_ids: Tensor, terminated: Tensor, start_bins: Tensor, ended: Tensor
    ) -> None:
        """Fade every bin, then move each touched bin toward its termination share.

        An episode ends in the bin of its last pre-step frame: a failure judged
        at ``f + 1`` is charged to ``f``, and the cursor already holds ``f + 1``
        here. The share is of ended episodes, so timeouts and clip ends lower it.
        ``bin_G`` is keyed by the start bin; this statistic is keyed by the end.
        """
        rank = self.bin_rank[start_bins.clamp_min(0)]
        last = self.start_frame[env_ids] + self.ep_step[env_ids] - 1
        last = torch.minimum(last.clamp_min(0), self.lengths[rank] - 1)
        end_bins = self.bin_ids(rank, last)
        weight = ended.to(self.bin_fail.dtype)
        sums = torch.zeros_like(self.bin_fail).index_add_(
            0, end_bins, terminated.to(weight.dtype) * weight
        )
        counts = torch.zeros_like(self.bin_fail).index_add_(0, end_bins, weight)
        faded = self.bin_fail * 0.999
        updated = torch.where(
            counts > 0, 0.9 * faded + (1 - 0.9) * sums / counts.clamp_min(1.0), faded
        )
        self.bin_fail.copy_(torch.where(ended.any(), updated, self.bin_fail))

    # ------------------------------------------------------------------
    # Masked, static-shape forms. Every environment is computed and the
    # rows of ``mask`` are selected; state changes in place, so a CUDA graph
    # that captured these calls keeps reading the live tensors.
    # ------------------------------------------------------------------

    #: Fixed-point scale of the sampling CDF (bin weights are 0.5 / bins or more).
    CDF_SCALE = float(2**40)
    #: Fixed-point scale of the per-bin return sums (|returns| stay far below 2**20).
    RETURN_SCALE = float(2**32)

    def _all_rows(self) -> Tensor:
        rows = getattr(self, "_rows", None)
        if rows is None:
            rows = torch.arange(self.num_envs, device=self.device)
            self._rows = rows
            # Rows that add nothing go to distinct bins, which keeps ~N zero-weight
            # atomics off one address.
            self._spread = rows % self.num_bins
        return rows

    def sample_masked(
        self, u_bin: Tensor, u_frame: Tensor, gate: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """``sample`` for every row of ``u_bin`` (float64) and ``u_frame`` (float32).

        ``gate`` is a device bool: the memory advances only where it holds, which
        is ``mask.any()`` for the step's reset mask. The draw is the inverse CDF of
        the same distribution ``sample`` passes to ``multinomial``.
        """
        soft = self._soft()
        torch.where(
            gate,
            self.bin_w_avg * 0.999 + 0.001 * soft,
            self.bin_w_avg,
            out=self.bin_w_avg,
        )
        probs = self._base(soft)
        if self.fail_share > 0.0:
            probs = self._mix_failures(probs)
        if self.focus_q is not None:
            probs = self._mix_focus(probs)
        probs = self._kept(probs)
        fixed = torch.round(probs.to(torch.float64) * self.CDF_SCALE).to(torch.int64)
        cdf = torch.cumsum(fixed, dim=0)
        target = torch.floor(u_bin.to(torch.float64) * cdf[-1].to(torch.float64)).to(
            torch.int64
        )
        bins = torch.searchsorted(cdf, target, right=True).clamp_max(self.num_bins - 1)
        ranks, start, end = self.bins.index_select(0, bins).unbind(-1)
        frame = start + (u_frame.to(torch.float32) * (end - start).float()).long()
        frame = torch.minimum(frame, self.lengths.index_select(0, ranks) - 2)
        return ranks, frame, bins

    def begin_episodes_masked(
        self, mask: Tensor, ranks: Tensor, frames: Tensor, bins: Tensor
    ) -> None:
        """``begin_episodes`` for the rows of ``mask`` (all-env tensors)."""
        torch.where(mask, bins, self.start_bin, out=self.start_bin)
        self.ep_step.masked_fill_(mask, 0)
        self.ep_disc_ret.masked_fill_(mask, 0.0)
        self.ep_ret.masked_fill_(mask, 0.0)
        if self.fail_share > 0.0:
            torch.where(mask, frames, self.start_frame, out=self.start_frame)

    def end_episodes_masked(self, mask: Tensor, terminated: Tensor) -> None:
        """``end_episodes`` for the rows of ``mask`` that completed a transition.

        Rows of ``mask`` with no transition (``ep_step == 0``) change nothing,
        as the eager reset's ``completed`` selection leaves them. The per-bin mean
        is exact to ``2**-32`` per return; the eager form sums in float32.
        """
        rows = self._all_rows()
        steps = self.ep_step.float()
        boot = torch.where(
            terminated,
            0.0,
            (self.discount**steps)
            * (self.ep_ret / steps.clamp_min(1.0))
            / (1.0 - self.discount),
        )
        returns = self.ep_disc_ret + boot
        bins = self.start_bin
        completed = mask & (self.ep_step > 0)
        valid = (bins >= 0) & completed & torch.isfinite(returns)
        safe_bins = torch.where(valid, bins, self._spread)
        fixed = torch.round(
            torch.where(valid, returns, 0.0).to(torch.float64) * self.RETURN_SCALE
        ).to(torch.int64)
        sums = torch.zeros(
            self.num_bins, device=self.device, dtype=torch.int64
        ).index_add_(0, safe_bins, fixed)
        counts = torch.zeros(
            self.num_bins, device=self.device, dtype=torch.int64
        ).index_add_(0, safe_bins, valid.to(torch.int64))
        average = (sums.to(torch.float64) / self.RETURN_SCALE / counts.clamp_min(1)).to(
            self.bin_G.dtype
        )
        torch.where(
            counts > 0, 0.8 * self.bin_G + 0.2 * average, self.bin_G, out=self.bin_G
        )
        if self.fail_share > 0.0:
            ended = (bins >= 0) & completed
            self._update_failures(rows, terminated, bins, ended)
        torch.where(completed, returns, self.last_return, out=self.last_return)
        self.start_bin.masked_fill_(completed, -1)


def install_reset_stats_persistence(agent, env) -> None:
    """Carry the env's reset statistics in the agent's resume checkpoints.

    Does nothing unless ``env.cfg.persist_reset_stats`` is on, so the agent and
    its checkpoints then stay exactly as they were. On, each full checkpoint holds
    ``RESET_STATS_KEY``, in the file itself so it cannot part from the weights, and
    a resume restores it. A resume from a checkpoint without the key is refused:
    statistics that restart at every segment are the behaviour the option exists
    to remove, and on a chain that would otherwise go unnoticed.

    ``_checkpoint_payload`` is WMBPTT's single source for its full checkpoints,
    periodic or final; an agent without it is refused rather than left unsaved.
    """
    if not getattr(env.cfg, "persist_reset_stats", False):
        return
    payload = getattr(agent, "_checkpoint_payload", None)
    if payload is None:
        raise ValueError(
            f"env.persist_reset_stats needs an agent with _checkpoint_payload; "
            f"{type(agent).__name__} has none."
        )
    sampler = env.g1_reset_sampler
    load_model = agent.load_model

    def payload_with_reset_stats(frames: int) -> dict:
        checkpoint = payload(frames)
        checkpoint[RESET_STATS_KEY] = sampler.state_dict()
        return checkpoint

    def load_model_with_reset_stats(path, *args, **kwargs) -> None:
        load_model(path, *args, **kwargs)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if RESET_STATS_KEY not in checkpoint:
            raise ValueError(
                f"{path} holds no reset statistics; resume it with "
                "env.persist_reset_stats=false."
            )
        sampler.load_state_dict(checkpoint[RESET_STATS_KEY])
        print(
            f"[INFO] Restored reset statistics of {sampler.num_bins} bins from "
            f"{path}: sum bin_G {float(sampler.bin_G.sum()):.6f}.",
            flush=True,
        )

    agent._checkpoint_payload = payload_with_reset_stats
    agent.load_model = load_model_with_reset_stats
