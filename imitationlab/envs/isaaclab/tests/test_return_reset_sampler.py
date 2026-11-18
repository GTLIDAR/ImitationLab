"""SuperMimic's return-based reset sampler (arXiv 2610.28480, App. D.1)."""

from __future__ import annotations

import math

import pytest
import torch

from imitationlab.envs.isaaclab.tasks.manager_based.imitation.mdp.commands.reset_sampling import (
    ReturnBasedResetSampler,
)


def _sampler(**kwargs) -> ReturnBasedResetSampler:
    sampler = ReturnBasedResetSampler(
        torch.tensor([120, 55]),
        num_envs=4,
        max_episode_steps=500,
        bin_size=50,
        **kwargs,
    )
    sampler.set_step_reward_max(0.1)
    return sampler


def test_bins_follow_the_sonic_layout() -> None:
    sampler = _sampler()
    assert sampler.bins.tolist() == [
        [0, 0, 50],
        [0, 50, 100],
        [0, 100, 120],
        [1, 0, 50],
        [1, 50, 55],
    ]


def test_unvisited_bins_are_sampled_uniformly() -> None:
    probabilities = _sampler().sampling_probabilities()
    torch.testing.assert_close(probabilities, torch.full((5,), 0.2))


def test_episode_return_is_normalized_by_the_attainable_return() -> None:
    sampler = _sampler(discount=0.9)
    sampler.begin_episodes(
        torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([10, 52])
    )
    for _ in range(2):
        sampler.accumulate(torch.tensor([0.1, 0.05, 0.0, 0.0]))
    sampler.end_episodes(torch.tensor([0, 1]))

    # env 0 starts in bin 0 with 110 frames left; env 1 in bin 4 with 3 left.
    max_0 = 0.1 * (1 - 0.9**110) / (1 - 0.9)
    max_1 = 0.1 * (1 - 0.9**3) / (1 - 0.9)
    got_0 = (0.1 + 0.9 * 0.1) / max_0
    got_1 = (0.05 + 0.9 * 0.05) / max_1
    torch.testing.assert_close(
        sampler.last_normalized_return[:2], torch.tensor([got_0, got_1])
    )
    torch.testing.assert_close(
        sampler.last_normalized_undiscounted[:2],
        torch.tensor([0.2 / (0.1 * 110), 0.1 / (0.1 * 3)]),
    )
    assert sampler.visited.tolist() == [True, False, False, False, True]
    torch.testing.assert_close(
        sampler.normalized_return[[0, 4]], torch.tensor([got_0, got_1])
    )
    assert sampler.env_bin[:2].tolist() == [-1, -1]


def test_low_return_bins_dominate_the_softmax_branch() -> None:
    sampler = _sampler(temperature=25.0)
    sampler.visited.fill_(True)
    sampler.normalized_return = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0])
    probabilities = sampler.sampling_probabilities()
    # 50% uniform (0.1 each) plus 40% softmax and 10% memory almost all on bin 2.
    hard = 0.1 + 0.5 * (1.0 / (1.0 + 4.0 * math.exp(-25.0)))
    assert probabilities[2].item() == pytest.approx(hard, rel=1e-6)
    assert probabilities.sum().item() == pytest.approx(1.0)


def test_memory_keeps_a_learned_bin_in_the_mix() -> None:
    sampler = _sampler(temperature=25.0)
    sampler.visited.fill_(True)
    sampler.normalized_return = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0])
    sampler.sampling_probabilities()
    # Bin 2 is learned and bin 0 becomes the hard one.
    sampler.normalized_return = torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0])
    probabilities = sampler.sampling_probabilities()
    # Bin 2 keeps its 10% memory share, split with bin 0's new peak.
    assert probabilities[2].item() == pytest.approx(0.1 + 0.05, rel=1e-4)


def test_repeat_episodes_move_the_estimate_by_the_ema_rate() -> None:
    sampler = _sampler(discount=0.5, ema=0.25)
    for reward in (0.1, 0.0):
        sampler.begin_episodes(torch.tensor([0]), torch.tensor([0]), torch.tensor([0]))
        sampler.accumulate(torch.tensor([reward, 0.0, 0.0, 0.0]))
        sampler.end_episodes(torch.tensor([0]))
    first = 0.1 / (0.1 * (1 - 0.5**120) / 0.5)
    assert sampler.normalized_return[0].item() == pytest.approx(0.75 * first)


def test_samples_land_inside_the_drawn_bin() -> None:
    generator = torch.Generator().manual_seed(0)
    sampler = ReturnBasedResetSampler(
        torch.tensor([120, 55]),
        num_envs=1,
        max_episode_steps=500,
        bin_size=50,
        generator=generator,
    )
    ranks, steps = sampler.sample(4000)
    lengths = torch.tensor([120, 55]).index_select(0, ranks)
    assert torch.all((steps >= 0) & (steps < lengths))
    # Uniform over 5 bins: trajectory 0 owns 3 of them.
    assert (ranks == 0).float().mean().item() == pytest.approx(0.6, abs=0.03)


def test_ratios_must_sum_to_one() -> None:
    with pytest.raises(ValueError):
        _sampler(uniform_ratio=0.5, softmax_ratio=0.5, memory_ratio=0.1)


def test_non_finite_rewards_do_not_poison_the_distribution() -> None:
    sampler = _sampler()
    sampler.begin_episodes(
        torch.tensor([0, 1, 2]), torch.tensor([0, 0, 1]), torch.tensor([0, 60, 0])
    )
    sampler.accumulate(torch.tensor([float("nan"), float("inf"), float("-inf"), 0.0]))
    sampler.accumulate(torch.tensor([0.1, 0.1, 0.1, 0.0]))
    sampler.end_episodes(torch.tensor([0, 1, 2]))
    assert torch.isfinite(sampler.normalized_return).all()
    probabilities = sampler.sampling_probabilities()
    assert torch.isfinite(probabilities).all()
    assert probabilities.sum().item() == pytest.approx(1.0)
    sampler.sample(16)  # torch.multinomial asserts on a NaN distribution
