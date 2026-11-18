"""Masked forms of the return-based reset sampler match the index forms."""

from __future__ import annotations

import torch

from imitationlab.envs.isaaclab.tasks.manager_based.imitation.mdp.commands.reset_sampling import (
    ReturnBasedResetSampler,
)


def _pair(num_envs: int = 6) -> tuple[ReturnBasedResetSampler, ReturnBasedResetSampler]:
    def make() -> ReturnBasedResetSampler:
        sampler = ReturnBasedResetSampler(
            torch.tensor([120, 55, 300]),
            num_envs=num_envs,
            max_episode_steps=500,
            bin_size=50,
            discount=0.95,
            ema=0.3,
        )
        sampler.set_step_reward_max(0.1)
        return sampler

    return make(), make()


def test_masked_episode_accounting_matches_index_accounting() -> None:
    eager, masked = _pair()
    ranks = torch.tensor([0, 1, 2, 2, 0, 1])
    steps = torch.tensor([10, 52, 0, 260, 110, 3])
    ids = torch.tensor([0, 2, 3, 5])
    mask = torch.zeros(6, dtype=torch.bool)
    mask[ids] = True
    eager.begin_episodes(ids, ranks[ids], steps[ids])
    masked.begin_episodes_masked(mask, ranks, steps)
    for name in ("env_bin", "env_discounted_max", "env_undiscounted_max"):
        torch.testing.assert_close(getattr(masked, name), getattr(eager, name))
    rewards = torch.tensor([0.1, 0.05, 0.02, 0.08, 0.0, 0.03])
    for _ in range(5):
        eager.accumulate(rewards)
        masked.accumulate(rewards)
    end_ids = torch.tensor([0, 3, 5])
    end_mask = torch.zeros(6, dtype=torch.bool)
    end_mask[end_ids] = True
    eager.end_episodes(end_ids)
    masked.end_episodes_masked(end_mask)
    torch.testing.assert_close(
        masked.last_normalized_return[end_mask], eager.last_normalized_return[end_mask]
    )
    # Fixed-point bin sums: equal to 2**-32 per sample.
    torch.testing.assert_close(
        masked.normalized_return, eager.normalized_return, atol=1e-8, rtol=0.0
    )
    assert torch.equal(masked.visited, eager.visited)
    assert torch.equal(masked.env_bin, eager.env_bin)


def test_inverse_cdf_draws_follow_the_bin_distribution() -> None:
    sampler, _ = _pair()
    sampler.normalized_return.copy_(torch.linspace(0.0, 1.0, int(sampler.num_bins)))
    sampler.visited.fill_(True)
    reference = sampler.sampling_probabilities()
    sampler.memory.zero_()
    n = 400_000
    u_bin = torch.rand(
        n, dtype=torch.float64, generator=torch.Generator().manual_seed(0)
    )
    u_frame = torch.rand(n, generator=torch.Generator().manual_seed(1))
    ranks, frames = sampler.sample_from_uniforms(u_bin, u_frame)
    lengths = sampler.lengths.index_select(0, ranks)
    assert bool((frames >= 0).all()) and bool((frames < lengths).all())
    local = frames
    bins = sampler.layout.first_bin_ids.index_select(0, ranks) + torch.div(
        local, sampler.layout.bin_size, rounding_mode="floor"
    )
    freq = torch.bincount(bins, minlength=int(sampler.num_bins)).double() / n
    assert float((freq - reference.double()).abs().max()) < 4e-3


def test_cdf_is_integer_and_reproducible() -> None:
    sampler, other = _pair()
    sampler.normalized_return.uniform_(generator=torch.Generator().manual_seed(3))
    other.normalized_return.copy_(sampler.normalized_return)
    cdf = sampler.sampling_cdf()
    assert cdf.dtype == torch.int64
    assert torch.equal(cdf, other.sampling_cdf())
    assert bool((cdf.diff() > 0).all())
