"""Masked forms of the G1 reset sampler against its index forms."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

_PATH = Path(__file__).parent.parent / "contracts/g1_reset_sampling.py"
_SPEC = importlib.util.spec_from_file_location("g1_reset_sampling_masked", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
G1ResetSampler = _MODULE.G1ResetSampler

NUM_ENVS = 8


def _pair(fail_share: float = 0.0):
    def make():
        return G1ResetSampler(
            torch.tensor([120, 55, 301]),
            num_envs=NUM_ENVS,
            discount=0.9,
            fail_share=fail_share,
        )

    return make(), make()


def _mask(ids) -> torch.Tensor:
    mask = torch.zeros(NUM_ENVS, dtype=torch.bool)
    mask[torch.tensor(ids)] = True
    return mask


def _run_episodes(eager, masked, *, terminated, end_ids, never_started=()):
    """Same starts, rewards and ends through both forms."""
    ranks = torch.tensor([0, 1, 2, 2, 0, 1, 2, 0])
    frames = torch.tensor([10, 52, 0, 260, 110, 3, 149, 60])
    bins = eager.bin_ids(ranks, frames)
    started = [i for i in range(NUM_ENVS) if i not in never_started]
    ids = torch.tensor(started)
    eager.begin_episodes(ids, ranks[ids], frames[ids], bins=bins[ids])
    masked.begin_episodes_masked(_mask(started), ranks, frames, bins)
    for name in ("start_bin", "ep_step", "ep_disc_ret", "ep_ret"):
        assert torch.equal(getattr(masked, name), getattr(eager, name)), name
    rewards = torch.tensor([0.1, 0.05, 0.02, 0.08, 0.0, 0.03, 0.09, 0.07])
    for step in range(5):
        eager.accumulate(rewards * (step + 1))
        masked.accumulate(rewards * (step + 1))
    end = torch.tensor(end_ids)
    completed = eager.ep_step[end] > 0
    eager.end_episodes(end, terminated[end], completed=completed)
    masked.end_episodes_masked(_mask(end_ids), terminated)


@pytest.mark.parametrize("fail_share", [0.0, 0.3])
def test_masked_episode_ends_match_the_index_form(fail_share) -> None:
    eager, masked = _pair(fail_share)
    terminated = torch.tensor([True, False, True, False, True, False, False, True])
    # Envs 2 and 6 start in one bin of clip 2, so the average has two rows.
    _run_episodes(eager, masked, terminated=terminated, end_ids=[0, 2, 3, 6])
    # Fixed-point sums: exact to 2**-32 per return, then the same EMA.
    torch.testing.assert_close(masked.bin_G, eager.bin_G, atol=1e-6, rtol=0.0)
    assert bool((eager.bin_G != 0).sum() >= 3)
    for name in ("start_bin", "last_return", "ep_step", "ep_disc_ret", "ep_ret"):
        assert torch.equal(getattr(masked, name), getattr(eager, name)), name
    if fail_share:
        # Integer-valued sums: identical, not just close.
        assert torch.equal(masked.bin_fail, eager.bin_fail)
        assert float(masked.bin_fail.sum()) > 0


def test_a_masked_row_without_a_transition_changes_nothing() -> None:
    eager, masked = _pair(0.3)
    terminated = torch.ones(NUM_ENVS, dtype=torch.bool)
    _run_episodes(
        eager, masked, terminated=terminated, end_ids=[1, 5], never_started=(5,)
    )
    torch.testing.assert_close(masked.bin_G, eager.bin_G, atol=1e-6, rtol=0.0)
    assert torch.equal(masked.bin_fail, eager.bin_fail)
    # Env 5 never started: its bin stays idle and nothing was credited for it.
    assert int(masked.start_bin[5]) == -1 and float(masked.last_return[5]) == 0.0


def test_non_finite_returns_are_skipped_in_both_forms() -> None:
    eager, masked = _pair()
    for sampler in (eager, masked):
        zeros = torch.zeros(NUM_ENVS, dtype=torch.long)
        sampler.begin_episodes(torch.arange(NUM_ENVS), zeros, zeros)
        reward = torch.full((NUM_ENVS,), 0.05)
        reward[3] = float("nan")
        sampler.accumulate(reward)
    terminated = torch.zeros(NUM_ENVS, dtype=torch.bool)
    eager.end_episodes(torch.arange(NUM_ENVS), terminated)
    masked.end_episodes_masked(torch.ones(NUM_ENVS, dtype=torch.bool), terminated)
    assert bool(torch.isfinite(masked.bin_G).all())
    torch.testing.assert_close(masked.bin_G, eager.bin_G, atol=1e-6, rtol=0.0)


def test_memory_advances_only_on_a_gated_step_and_as_sample_does() -> None:
    eager, masked = _pair()
    eager.bin_G.copy_(torch.linspace(-1.0, 5.0, eager.num_bins))
    masked.bin_G.copy_(eager.bin_G)
    u = torch.full((NUM_ENVS,), 0.5, dtype=torch.float64)
    before = masked.bin_w_avg.clone()
    masked.sample_masked(u, u.float(), torch.tensor(False))
    assert torch.equal(masked.bin_w_avg, before)
    eager.sample(3)
    masked.sample_masked(u, u.float(), torch.tensor(True))
    assert torch.equal(masked.bin_w_avg, eager.bin_w_avg)
    assert not torch.equal(masked.bin_w_avg, before)


@pytest.mark.parametrize("fail_share", [0.0, 0.3])
def test_inverse_cdf_draws_follow_the_sampling_distribution(fail_share) -> None:
    sampler, _ = _pair(fail_share)
    sampler.bin_G.copy_(torch.linspace(-2.0, 8.0, sampler.num_bins))
    if fail_share:
        sampler.bin_fail[3] = 0.5
    n = 400_000
    big = G1ResetSampler(
        sampler.lengths, num_envs=n, discount=0.9, fail_share=fail_share
    )
    big.bin_G.copy_(sampler.bin_G)
    if fail_share:
        big.bin_fail.copy_(sampler.bin_fail)
    u_bin = torch.rand(
        n, dtype=torch.float64, generator=torch.Generator().manual_seed(0)
    )
    u_frame = torch.rand(n, generator=torch.Generator().manual_seed(1))
    ranks, frames, bins = big.sample_masked(u_bin, u_frame, torch.tensor(True))
    # The memory advanced once, as `sample` would, before the draw.
    reference = big.sampling_probabilities().double()
    freq = torch.bincount(bins, minlength=big.num_bins).double() / n
    assert float((freq - reference).abs().max()) < 4e-3
    lengths = big.lengths[ranks]
    start, end = big.bins[bins, 1], big.bins[bins, 2]
    # Clip 2 (301 frames) ends in a singleton bin, drawn as its last frame - 1.
    singleton = start == end
    assert bool(singleton.any())
    assert bool(((frames >= start) & (frames < end))[~singleton].all())
    assert bool((frames[singleton] == lengths[singleton] - 2).all())
    assert torch.equal(big.bins[bins, 0], ranks)


def test_the_cdf_is_integer_and_the_draw_reproducible() -> None:
    a, b = _pair()
    for sampler in (a, b):
        sampler.bin_G.copy_(torch.linspace(0.0, 3.0, sampler.num_bins))
    u = torch.rand(
        NUM_ENVS, dtype=torch.float64, generator=torch.Generator().manual_seed(4)
    )
    out_a = a.sample_masked(u, u.float(), torch.tensor(True))
    out_b = b.sample_masked(u, u.float(), torch.tensor(True))
    for x, y in zip(out_a, out_b):
        assert torch.equal(x, y)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("fail_share", [0.0, 0.3])
def test_masked_bookkeeping_replays_as_a_graph_bit_for_bit_without_a_sync(
    fail_share,
) -> None:
    """One step's sampler work, eager and as a CUDA graph, over random masks."""
    device = torch.device("cuda")
    n = 64
    lengths = torch.tensor([120, 55, 301], device=device)

    def make():
        return G1ResetSampler(lengths, num_envs=n, discount=0.9, fail_share=fail_share)

    eager, graphed = make(), make()
    reward = torch.zeros(n, device=device)
    mask = torch.zeros(n, device=device, dtype=torch.bool)
    terminated = torch.zeros_like(mask)
    u_bin = torch.zeros(n, device=device, dtype=torch.float64)
    u_frame = torch.zeros(n, device=device)

    def segment(sampler):
        sampler.accumulate(reward)
        sampler.end_episodes_masked(mask, terminated)
        ranks, frames, bins = sampler.sample_masked(u_bin, u_frame, mask.any())
        sampler.begin_episodes_masked(mask, ranks, frames, bins)

    generator = torch.Generator(device=device).manual_seed(7)

    def inputs(p_reset):
        reward.copy_(torch.rand(n, device=device, generator=generator) * 0.1)
        mask.copy_(torch.rand(n, device=device, generator=generator) < p_reset)
        terminated.copy_(
            mask & (torch.rand(n, device=device, generator=generator) < 0.5)
        )
        u_bin.copy_(
            torch.rand(n, device=device, generator=generator, dtype=torch.float64)
        )
        u_frame.copy_(torch.rand(n, device=device, generator=generator))

    # Every env starts an episode, eagerly in both, before the capture.
    inputs(1.1)
    segment(eager)
    segment(graphed)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        segment(graphed)
    torch.cuda.synchronize()
    names = [k for k, v in vars(eager).items() if isinstance(v, torch.Tensor)]
    for step in range(40):
        inputs(0.0 if step % 7 == 3 else 0.2)
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            segment(eager)
            graph.replay()
        finally:
            torch.cuda.set_sync_debug_mode("default")
        for name in names:
            assert torch.equal(getattr(graphed, name), getattr(eager, name)), (
                step,
                name,
            )
    assert float(eager.bin_G.abs().sum()) > 0
