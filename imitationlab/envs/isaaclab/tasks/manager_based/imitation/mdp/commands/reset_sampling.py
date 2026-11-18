"""Reset samplers owned by the imitation environment.

The official SONIC sampler remains in :mod:`imitationlab.data_tools.common.reset_sampling`.
This module contains repo-specific compositions around that sampler so they do
not change the reproduced SONIC behavior.
"""

from __future__ import annotations

import torch

from imitationlab.data_tools.common.reset_sampling import SonicAdaptiveResetSampler


class RandomTrajectoryAdaptiveResetSampler:
    """Mix uniform-trajectory starts with SONIC adaptive starts.

    The random branch first chooses a trajectory uniformly, then chooses a
    local frame uniformly from the leading ``random_start_fraction`` of that
    trajectory.  The adaptive branch delegates unchanged to ``adaptive``.

    This is deliberately different from SONIC's ``uniform_sampling_rate``:
    that setting mixes probability over trajectory-frame bins before applying
    sequence-length weights, while this class makes an explicit per-reset
    branch choice and gives every trajectory equal probability on the random
    branch.
    """

    def __init__(
        self,
        lengths: torch.Tensor,
        *,
        adaptive: SonicAdaptiveResetSampler,
        random_sampling_ratio: float = 0.8,
        random_start_fraction: float = 0.5,
        generator: torch.Generator | None = None,
    ) -> None:
        if lengths.ndim != 1 or lengths.numel() == 0:
            raise ValueError("lengths must be a non-empty one-dimensional tensor.")
        if torch.any(lengths <= 0):
            raise ValueError("All trajectory lengths must be positive.")
        if not 0.0 <= float(random_sampling_ratio) <= 1.0:
            raise ValueError("random_sampling_ratio must be in [0, 1].")
        if not 0.0 < float(random_start_fraction) <= 1.0:
            raise ValueError("random_start_fraction must be in (0, 1].")

        self.lengths = lengths.to(dtype=torch.long)
        self.adaptive = adaptive
        self.random_sampling_ratio = float(random_sampling_ratio)
        self.random_start_fraction = float(random_start_fraction)
        self.generator = generator

    def sampling_probabilities(self) -> torch.Tensor:
        """Return the current adaptive-branch probability snapshot."""
        return self.adaptive.sampling_probabilities()

    def sample(
        self,
        batch_size: int,
        *,
        probabilities: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample trajectory ranks and local frames for ``batch_size`` resets."""
        count = int(batch_size)
        if count < 0:
            raise ValueError("batch_size must be non-negative.")
        if count == 0:
            empty = torch.empty(0, device=self.lengths.device, dtype=torch.long)
            return empty, empty.clone()
        if self.random_sampling_ratio <= 0.0:
            return self.adaptive.sample(count, probabilities=probabilities)

        device = self.lengths.device
        random_mask = (
            torch.rand(count, device=device, generator=self.generator)
            < self.random_sampling_ratio
        )
        if self.random_sampling_ratio >= 1.0:
            random_mask.fill_(True)

        # Sampling full candidate batches avoids a device synchronization just
        # to discover the stochastic branch counts. At reset-batch scale this
        # is cheaper than pulling ``random_mask.sum()`` back to Python.
        adaptive_ranks, adaptive_steps = self.adaptive.sample(
            count,
            probabilities=probabilities,
        )
        random_ranks = torch.randint(
            int(self.lengths.numel()),
            (count,),
            device=device,
            generator=self.generator,
        )
        selected_lengths = self.lengths.index_select(0, random_ranks)
        # ceil(fraction * length) makes the half-open range non-empty even for
        # a one-frame trajectory and defines 50% of odd lengths without
        # silently discarding their middle frame.
        random_span = torch.ceil(
            selected_lengths.to(dtype=torch.float64) * self.random_start_fraction
        ).to(dtype=torch.long)
        random_steps = torch.floor(
            torch.rand(
                count,
                device=device,
                dtype=torch.float64,
                generator=self.generator,
            )
            * random_span
        ).to(dtype=torch.long)
        return (
            torch.where(random_mask, random_ranks, adaptive_ranks),
            torch.where(random_mask, random_steps, adaptive_steps),
        )


class ReturnBasedResetSampler:
    """SuperMimic's return-based reset sampling (arXiv 2610.28480, App. D.1).

    The motion timeline is cut into ``bin_size``-frame bins (SONIC's layout,
    reused from :class:`SonicAdaptiveResetSampler`). Each bin keeps a running
    estimate of the normalized discounted return earned from it, and a reset
    draws its bin from a three-way mixture:

    - ``uniform_ratio`` uniform over bins;
    - ``softmax_ratio`` from ``softmax(-temperature * G_b)``, so low-return
      bins dominate;
    - ``memory_ratio`` in proportion to each bin's past softmax weight, so a
      motion that was hard keeps some practice after it is learned.

    The paper leaves three things open; this class fixes them as follows:

    - ``G_b`` is the discounted return of episodes that START in bin ``b``,
      divided by the largest return attainable from that start
      (``step_reward_max * (1 - discount**H) / (1 - discount)`` with
      ``H = min(frames left in the motion, max_episode_steps)``), clamped to
      [0, 1], and folded into the bin with an exponential moving average of
      rate ``ema``. The first episode from a bin sets it directly.
    - Unvisited bins hold ``G_b = 0``, so the softmax branch covers them before
      it revisits bins it has already scored.
    - "Past softmax weight" is the running maximum of each bin's softmax
      probability.

    Frames are uniform inside the drawn bin, with no SONIC lead-in.

    Per-environment state: :meth:`begin_episodes` records the start bin and
    the attainable return; :meth:`accumulate` adds one step of reward;
    :meth:`end_episodes` folds the finished returns into the bins. All three
    stay on the device and never synchronize with the host.
    """

    def __init__(
        self,
        lengths: torch.Tensor,
        *,
        num_envs: int,
        max_episode_steps: int,
        bin_size: int = 50,
        uniform_ratio: float = 0.5,
        softmax_ratio: float = 0.4,
        memory_ratio: float = 0.1,
        temperature: float = 25.0,
        discount: float = 0.99,
        ema: float = 0.2,
        generator: torch.Generator | None = None,
    ) -> None:
        ratios = (float(uniform_ratio), float(softmax_ratio), float(memory_ratio))
        if any(r < 0.0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError(
                "uniform_ratio, softmax_ratio and memory_ratio must be "
                f"non-negative and sum to 1; got {ratios}."
            )
        if float(temperature) <= 0.0:
            raise ValueError("temperature must be positive.")
        if not 0.0 < float(discount) < 1.0:
            raise ValueError("discount must be in (0, 1).")
        if not 0.0 < float(ema) <= 1.0:
            raise ValueError("ema must be in (0, 1].")
        if int(num_envs) <= 0 or int(max_episode_steps) <= 0:
            raise ValueError("num_envs and max_episode_steps must be positive.")

        self.layout = SonicAdaptiveResetSampler(
            lengths,
            bin_size=int(bin_size),
            sequence_length_agnostic=False,
            pre_failure_sample_window=0,
            generator=generator,
        )
        self.device = self.layout.device
        self.lengths = self.layout.trajectory_lengths
        self.num_bins = self.layout.num_bins
        self.bins = self.layout.bins
        self.generator = generator
        self.uniform_ratio, self.softmax_ratio, self.memory_ratio = ratios
        self.temperature = float(temperature)
        self.discount = float(discount)
        self.ema = float(ema)
        self.max_episode_steps = int(max_episode_steps)
        self.step_reward_max: float | None = None

        self.normalized_return = torch.zeros(
            self.num_bins, device=self.device, dtype=torch.float32
        )
        self.visited = torch.zeros(self.num_bins, device=self.device, dtype=torch.bool)
        self.memory = torch.zeros_like(self.normalized_return)

        n = int(num_envs)
        self.env_bin = torch.full((n,), -1, device=self.device, dtype=torch.long)
        self.env_discounted = torch.zeros(n, device=self.device, dtype=torch.float32)
        self.env_weight = torch.ones_like(self.env_discounted)
        self.env_undiscounted = torch.zeros_like(self.env_discounted)
        self.env_discounted_max = torch.ones_like(self.env_discounted)
        self.env_undiscounted_max = torch.ones_like(self.env_discounted)
        self.last_normalized_return = torch.zeros_like(self.env_discounted)
        self.last_normalized_undiscounted = torch.zeros_like(self.env_discounted)

    def set_step_reward_max(self, value: float) -> None:
        """Largest per-step reward (dt-scaled sum of positive term weights)."""
        if float(value) <= 0.0:
            raise ValueError("step_reward_max must be positive.")
        self.step_reward_max = float(value)

    def sampling_probabilities(self) -> torch.Tensor:
        """Current bin distribution; also advances the softmax-weight memory."""
        softmax = torch.softmax(-self.temperature * self.normalized_return, dim=0)
        torch.maximum(self.memory, softmax, out=self.memory)
        probabilities = self.softmax_ratio * softmax + self.uniform_ratio / float(
            self.num_bins
        )
        memory_sum = self.memory.sum()
        probabilities = probabilities + self.memory_ratio * torch.where(
            memory_sum > 0.0, self.memory / memory_sum.clamp_min(1e-30), softmax
        )
        return probabilities / probabilities.sum()

    def sample(
        self,
        count: int,
        *,
        probabilities: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw ``count`` (trajectory rank, local frame) reset points."""
        count = int(count)
        if count < 0:
            raise ValueError("count must be >= 0.")
        if count == 0:
            empty = torch.empty(0, device=self.device, dtype=torch.long)
            return empty, empty.clone()
        if probabilities is None:
            probabilities = self.sampling_probabilities()
        bin_ids = torch.multinomial(
            probabilities, count, replacement=True, generator=self.generator
        )
        drawn = self.bins.index_select(0, bin_ids)
        starts, ends = drawn[:, 1], drawn[:, 2]
        offsets = (
            (
                torch.rand(count, device=self.device, generator=self.generator)
                * (ends - starts)
            )
            .floor()
            .to(torch.long)
        )
        return drawn[:, 0], starts + offsets

    def begin_episodes(
        self,
        env_ids: torch.Tensor,
        trajectory_ranks: torch.Tensor,
        frame_steps: torch.Tensor,
    ) -> None:
        """Start return accounting for environments reset onto these points."""
        if self.step_reward_max is None:
            raise RuntimeError("set_step_reward_max must be called before resets.")
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        ranks = torch.as_tensor(trajectory_ranks, device=self.device, dtype=torch.long)
        steps = torch.as_tensor(frame_steps, device=self.device, dtype=torch.long)
        lengths = self.lengths.index_select(0, ranks)
        # Same bin lookup as the SONIC layout, without its host-synchronizing
        # range check: the ranks and frames come from this sampler or the
        # trajectory manager, both already in range.
        local = torch.minimum(steps.clamp_min(0), lengths - 1)
        self.env_bin[env_ids] = self.layout.first_bin_ids.index_select(
            0, ranks
        ) + torch.div(local, self.layout.bin_size, rounding_mode="floor")
        remaining = (lengths - steps).clamp_min(1)
        horizon = remaining.clamp_max(self.max_episode_steps).to(torch.float32)
        self.env_discounted_max[env_ids] = (
            self.step_reward_max
            * (1.0 - self.discount**horizon)
            / (1.0 - self.discount)
        )
        self.env_undiscounted_max[env_ids] = self.step_reward_max * horizon
        self.env_discounted[env_ids] = 0.0
        self.env_undiscounted[env_ids] = 0.0
        self.env_weight[env_ids] = 1.0

    def accumulate(self, rewards: torch.Tensor) -> None:
        """Add one control step of (dt-scaled) reward for every environment.

        A non-finite reward (a physics blow-up in one environment) counts as
        zero: a single NaN would otherwise reach the bin estimates, the softmax
        and ``torch.multinomial``, whose device assert kills the run (job
        6478126, 2026-10-22).
        """
        rewards = torch.nan_to_num(
            rewards.reshape(-1).to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        self.env_discounted.add_(self.env_weight * rewards)
        self.env_undiscounted.add_(rewards)
        self.env_weight.mul_(self.discount)

    def end_episodes(self, env_ids: torch.Tensor) -> None:
        """Fold the finished episodes' returns into their start bins."""
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        bins = self.env_bin.index_select(0, env_ids)
        started = bins >= 0
        normalized = torch.nan_to_num(
            self.env_discounted.index_select(0, env_ids)
            / self.env_discounted_max.index_select(0, env_ids),
            nan=0.0,
        ).clamp(0.0, 1.0)
        self.last_normalized_return[env_ids] = normalized
        self.last_normalized_undiscounted[env_ids] = torch.nan_to_num(
            self.env_undiscounted.index_select(0, env_ids)
            / self.env_undiscounted_max.index_select(0, env_ids),
            nan=0.0,
        ).clamp(0.0, 1.0)
        # Several environments may finish in one bin on the same step; their
        # mean is the sample, so the update is order-free and needs no host
        # sync to deduplicate.
        safe_bins = torch.where(started, bins, torch.zeros_like(bins))
        weight = started.to(torch.float32)
        sums = torch.zeros_like(self.normalized_return).index_add_(
            0, safe_bins, normalized * weight
        )
        counts = torch.zeros_like(self.normalized_return).index_add_(
            0, safe_bins, weight
        )
        touched = counts > 0
        sample = sums / counts.clamp_min(1.0)
        updated = torch.where(
            self.visited,
            (1.0 - self.ema) * self.normalized_return + self.ema * sample,
            sample,
        )
        self.normalized_return = torch.where(touched, updated, self.normalized_return)
        self.visited |= touched
        self.env_bin[env_ids] = -1

    # ------------------------------------------------------------------
    # Masked, static-shape forms for the graph step. Every environment is
    # computed and the resetting ones are selected; state is written in place.
    # Draws come from caller-supplied uniforms (the counter-based RNG), and
    # the softmax is built from multi-block reductions instead of the
    # single-block row softmax. The CDF and the bin sums are int64 fixed
    # point: integer scans and atomics are exact, so these forms give the same
    # result in every process (a float scan or float atomics do not). Same
    # model as the methods above, different random stream and rounding.
    # ------------------------------------------------------------------

    #: Fixed-point scale of the sampling CDF (bin weights are ~1e-6 or more).
    CDF_SCALE = float(2**40)
    #: Fixed-point scale of the per-bin return sums (returns are in [0, 1]).
    RETURN_SCALE = float(2**32)

    def sampling_cdf(self) -> torch.Tensor:
        """int64 fixed-point CDF of the bin distribution; advances the memory."""
        logits = -self.temperature * self.normalized_return
        weights = torch.exp(logits - logits.max())
        softmax = weights / weights.sum()
        torch.maximum(self.memory, softmax, out=self.memory)
        memory_sum = self.memory.sum()
        probabilities = (
            self.softmax_ratio * softmax
            + self.uniform_ratio / float(self.num_bins)
            + self.memory_ratio
            * torch.where(
                memory_sum > 0.0, self.memory / memory_sum.clamp_min(1e-30), softmax
            )
        )
        fixed = torch.round(probabilities.to(torch.float64) * self.CDF_SCALE).to(
            torch.int64
        )
        return torch.cumsum(fixed, dim=0)

    def sample_from_uniforms(
        self, u_bin: torch.Tensor, u_frame: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(trajectory rank, local frame) per row of ``u_bin`` (float64) / ``u_frame``."""
        cdf = self.sampling_cdf()
        target = torch.floor(u_bin.to(torch.float64) * cdf[-1].to(torch.float64)).to(
            torch.int64
        )
        bin_ids = torch.searchsorted(cdf, target, right=True).clamp_max(
            self.num_bins - 1
        )
        drawn = self.bins.index_select(0, bin_ids)
        starts, ends = drawn[:, 1], drawn[:, 2]
        offsets = (u_frame.to(torch.float32) * (ends - starts)).floor().to(torch.long)
        return drawn[:, 0], starts + offsets

    def begin_episodes_masked(
        self,
        mask: torch.Tensor,
        trajectory_ranks: torch.Tensor,
        frame_steps: torch.Tensor,
    ) -> None:
        """:meth:`begin_episodes` for the rows of ``mask`` (all-env tensors)."""
        if self.step_reward_max is None:
            raise RuntimeError("set_step_reward_max must be called before resets.")
        lengths = self.lengths.index_select(0, trajectory_ranks)
        local = torch.minimum(frame_steps.clamp_min(0), lengths - 1)
        env_bin = self.layout.first_bin_ids.index_select(
            0, trajectory_ranks
        ) + torch.div(local, self.layout.bin_size, rounding_mode="floor")
        horizon = (
            (lengths - frame_steps)
            .clamp_min(1)
            .clamp_max(self.max_episode_steps)
            .to(torch.float32)
        )
        discounted_max = (
            self.step_reward_max
            * (1.0 - self.discount**horizon)
            / (1.0 - self.discount)
        )
        torch.where(mask, env_bin, self.env_bin, out=self.env_bin)
        torch.where(
            mask, discounted_max, self.env_discounted_max, out=self.env_discounted_max
        )
        torch.where(
            mask,
            self.step_reward_max * horizon,
            self.env_undiscounted_max,
            out=self.env_undiscounted_max,
        )
        self.env_discounted.masked_fill_(mask, 0.0)
        self.env_undiscounted.masked_fill_(mask, 0.0)
        self.env_weight.masked_fill_(mask, 1.0)

    def end_episodes_masked(self, mask: torch.Tensor) -> None:
        """:meth:`end_episodes` for the rows of ``mask``, in place."""
        started = (self.env_bin >= 0) & mask
        normalized = torch.nan_to_num(
            self.env_discounted / self.env_discounted_max, nan=0.0
        ).clamp(0.0, 1.0)
        undiscounted = torch.nan_to_num(
            self.env_undiscounted / self.env_undiscounted_max, nan=0.0
        ).clamp(0.0, 1.0)
        torch.where(
            mask,
            normalized,
            self.last_normalized_return,
            out=self.last_normalized_return,
        )
        torch.where(
            mask,
            undiscounted,
            self.last_normalized_undiscounted,
            out=self.last_normalized_undiscounted,
        )
        # Rows that did not start an episode here add zero; spreading them over
        # distinct bins keeps ~N zero-weight atomics off one address.
        spread = getattr(self, "_spread_bins", None)
        if spread is None or spread.shape != self.env_bin.shape:
            spread = torch.arange(
                self.env_bin.numel(), device=self.device, dtype=torch.long
            ) % int(self.num_bins)
            self._spread_bins = spread
        safe_bins = torch.where(started, self.env_bin, spread)
        fixed = torch.round(normalized.to(torch.float64) * self.RETURN_SCALE).to(
            torch.int64
        ) * started.to(torch.int64)
        sums = torch.zeros(
            self.num_bins, device=self.device, dtype=torch.int64
        ).index_add_(0, safe_bins, fixed)
        counts = torch.zeros(
            self.num_bins, device=self.device, dtype=torch.int64
        ).index_add_(0, safe_bins, started.to(torch.int64))
        touched = counts > 0
        sample = (sums.to(torch.float64) / self.RETURN_SCALE / counts.clamp_min(1)).to(
            torch.float32
        )
        updated = torch.where(
            self.visited,
            (1.0 - self.ema) * self.normalized_return + self.ema * sample,
            sample,
        )
        torch.where(
            touched, updated, self.normalized_return, out=self.normalized_return
        )
        self.visited |= touched
        self.env_bin.masked_fill_(mask, -1)

    def diagnostics(self) -> dict[str, torch.Tensor]:
        """Device scalars describing the bin estimates (no host sync)."""
        visited = self.visited.to(torch.float32)
        return {
            "visited_bin_fraction": visited.mean(),
            "visited_bin_return_mean": (self.normalized_return * visited).sum()
            / visited.sum().clamp_min(1.0),
        }


__all__ = ["RandomTrajectoryAdaptiveResetSampler", "ReturnBasedResetSampler"]
