"""Independent values for the G1 replica reset protocol."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_PATH = Path(__file__).parent.parent / "contracts/g1_reset_sampling.py"
_SPEC = importlib.util.spec_from_file_location("g1_reset_sampling", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
G1ResetSampler = _MODULE.G1ResetSampler
RESET_STATS_KEY = _MODULE.RESET_STATS_KEY
install_reset_stats_persistence = _MODULE.install_reset_stats_persistence


def _sampler(lengths=(120, 55), num_envs=4, discount=0.9, fail_share=0.0):
    return G1ResetSampler(
        torch.tensor(lengths),
        num_envs=num_envs,
        discount=discount,
        generator=torch.Generator().manual_seed(53),
        fail_share=fail_share,
    )


def test_bin_layout_and_initial_probabilities() -> None:
    sampler = _sampler()
    assert sampler.bins.tolist() == [
        [0, 0, 50],
        [0, 50, 100],
        [0, 100, 119],
        [1, 0, 50],
        [1, 50, 54],
    ]
    torch.testing.assert_close(sampler.bin_G, torch.zeros(5))
    torch.testing.assert_close(sampler.bin_w_avg, torch.full((5,), 0.2))
    torch.testing.assert_close(sampler.sampling_probabilities(), torch.full((5,), 0.2))


def test_probability_memory_is_an_average_advanced_by_sampling_only() -> None:
    sampler = _sampler(lengths=(100,), discount=0.9)
    # Gmax = 1, so these raw returns give softmax(log(3), 0) = (0.75, 0.25).
    sampler.bin_G[:] = torch.tensor([-torch.log(torch.tensor(3.0)) / 25.0, 0.0])
    initial_memory = sampler.bin_w_avg.clone()
    torch.testing.assert_close(
        sampler.sampling_probabilities(), torch.tensor([0.6, 0.4])
    )
    torch.testing.assert_close(sampler.bin_w_avg, initial_memory)
    sampler.sample(20)
    torch.testing.assert_close(sampler.bin_w_avg, torch.tensor([0.50025, 0.49975]))
    torch.testing.assert_close(
        sampler.sampling_probabilities(), torch.tensor([0.600025, 0.399975])
    )
    # A second call updates exactly once, independent of its batch size.
    sampler.sample(1)
    expected = 0.999 * torch.tensor([0.50025, 0.49975]) + 0.001 * torch.tensor(
        [0.75, 0.25]
    )
    torch.testing.assert_close(sampler.bin_w_avg, expected)


def test_terminal_and_truncated_returns_have_distinct_bootstraps() -> None:
    sampler = _sampler(discount=0.5)
    ids = torch.tensor([0, 1])
    sampler.begin_episodes(ids, torch.tensor([0, 1]), torch.tensor([10, 52]))
    sampler.accumulate(torch.tensor([0.1, 0.1, 12.0, 12.0]))
    sampler.accumulate(torch.tensor([0.2, 0.2, 12.0, 12.0]))
    got = sampler.end_episodes(ids, torch.tensor([True, False]))
    # G = .1 + .5*.2 = .2. Truncation adds .5^2 * mean(.1,.2)/(1-.5) = .075.
    torch.testing.assert_close(got, torch.tensor([0.2, 0.275]))
    # First observation also applies .2 EMA; no special first-visit override.
    torch.testing.assert_close(sampler.bin_G, torch.tensor([0.04, 0, 0, 0, 0.055]))
    assert sampler.start_bin.tolist() == [-1, -1, -1, -1]
    assert sampler.ep_step.tolist() == [2, 2, 0, 0]


def test_duplicate_bins_average_before_ema_and_keep_negative_returns() -> None:
    sampler = _sampler(discount=0.5)
    ids = torch.tensor([0, 1, 2])
    sampler.begin_episodes(ids, torch.tensor([0, 0, 1]), torch.tensor([0, 17, 0]))
    sampler.bin_G[0] = 2.0
    sampler.accumulate(torch.tensor([1.0, 3.0, -1.0, 0.0]))
    sampler.end_episodes(ids, torch.tensor([True, True, True]))
    torch.testing.assert_close(sampler.bin_G, torch.tensor([2.0, 0.0, 0.0, -0.2, 0.0]))


@pytest.mark.parametrize("length", [2, 50, 51, 100, 101])
def test_every_sample_has_a_valid_next_frame_including_singleton_bins(
    length: int,
) -> None:
    sampler = _sampler(lengths=(length,), num_envs=1)
    ranks, frames, bins = sampler.sample(4096)
    assert torch.all(ranks == 0)
    assert torch.all((frames >= 0) & (frames + 1 < length))
    starts = sampler.bin_start[bins]
    ends = sampler.bin_end[bins]
    ordinary = starts < ends
    assert torch.all(
        (frames[ordinary] >= starts[ordinary]) & (frames[ordinary] < ends[ordinary])
    )
    if length % 50 == 1:
        singleton = bins == sampler.num_bins - 1
        assert singleton.any()
        assert torch.all(frames[singleton] == length - 2)
        # The corrected position lies in the preceding bin, but credit belongs
        # to the bin actually sampled, preserving its learning signal.
        sampler.begin_episodes(
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([length - 2]),
            bins=torch.tensor([sampler.num_bins - 1]),
        )
        sampler.accumulate(torch.tensor([1.0]))
        sampler.end_episodes(torch.tensor([0]), torch.tensor([True]))
        assert sampler.bin_G[-1] == pytest.approx(0.2)
        assert sampler.bin_G[:-1].sum() == 0


def test_draws_match_a_manually_computed_distribution() -> None:
    sampler = _sampler(lengths=(100,), discount=0.9)
    sampler.bin_G[:] = torch.tensor([-torch.log(torch.tensor(3.0)) / 25.0, 0.0])
    _, frames, bins = sampler.sample(20000)
    assert (bins == 0).float().mean().item() == pytest.approx(0.600025, abs=0.015)
    assert frames[bins == 0].float().mean().item() == pytest.approx(24.5, abs=0.6)
    assert frames[bins == 1].float().mean().item() == pytest.approx(74.0, abs=0.6)


def test_asynchronous_reset_leaves_other_episode_returns_intact() -> None:
    sampler = _sampler(discount=0.5)
    sampler.begin_episodes(
        torch.tensor([0, 1]), torch.tensor([0, 0]), torch.tensor([0, 60])
    )
    sampler.accumulate(torch.tensor([0.1, 0.2, 0.0, 0.0]))
    sampler.end_episodes(torch.tensor([0]), torch.tensor([True]))
    sampler.begin_episodes(torch.tensor([0]), torch.tensor([1]), torch.tensor([5]))
    sampler.accumulate(torch.tensor([0.3, 0.4, 0.0, 0.0]))
    torch.testing.assert_close(sampler.ep_disc_ret[:2], torch.tensor([0.3, 0.4]))
    torch.testing.assert_close(sampler.ep_ret[:2], torch.tensor([0.3, 0.6]))
    assert sampler.ep_step[:2].tolist() == [1, 2]


def test_zero_draw_and_zero_length_episode_do_not_update_memory_or_bins() -> None:
    sampler = _sampler()
    before = sampler.bin_w_avg.clone()
    assert all(x.numel() == 0 for x in sampler.sample(0))
    torch.testing.assert_close(sampler.bin_w_avg, before)
    sampler.begin_episodes(torch.tensor([0]), torch.tensor([0]), torch.tensor([0]))
    sampler.end_episodes(torch.tensor([0]), torch.tensor([False]))
    torch.testing.assert_close(sampler.bin_G, torch.zeros_like(sampler.bin_G))


def test_nonfinite_episode_cannot_poison_finite_bin_updates() -> None:
    sampler = _sampler()
    ids = torch.tensor([0, 1, 2])
    sampler.begin_episodes(ids, torch.tensor([0, 0, 1]), torch.tensor([0, 0, 0]))
    sampler.accumulate(torch.tensor([float("nan"), 0.1, float("inf"), 0.0]))
    sampler.end_episodes(ids, torch.tensor([True, True, False]))
    torch.testing.assert_close(sampler.bin_G, torch.tensor([0.02, 0, 0, 0, 0]))
    assert torch.isfinite(sampler.sampling_probabilities()).all()


def _tensors(sampler) -> dict[str, torch.Tensor]:
    return {k: v for k, v in vars(sampler).items() if isinstance(v, torch.Tensor)}


def _assert_same_state(a, b) -> None:
    left, right = _tensors(a), _tensors(b)
    assert left.keys() == right.keys()
    for name in left:
        assert torch.equal(left[name], right[name]), name


def _mid_episode_pair():
    pair = _sampler(num_envs=5), _sampler(num_envs=5)
    ids = torch.tensor([0, 1, 2, 3])
    for sampler in pair:
        sampler.begin_episodes(
            ids, torch.tensor([0, 0, 1, 0]), torch.tensor([3, 70, 10, 110])
        )
        sampler.last_return[:] = torch.tensor([9.0, 8.0, 7.0, 6.0, 5.0])
        sampler.accumulate(torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]))
    return pair


@pytest.mark.parametrize(
    "mask",
    [
        [True, True, True, True],
        [True, False, True, False],
        [False, False, False, False],
    ],
)
def test_masked_credit_equals_selecting_the_completed_rows(mask) -> None:
    selected, masked = _mid_episode_pair()
    ids = torch.tensor([0, 1, 2, 3])
    terminated = torch.tensor([True, False, True, False])
    completed = torch.tensor(mask)
    if completed.any():
        selected.end_episodes(ids[completed], terminated[completed])
    masked.end_episodes(ids, terminated, completed=completed)
    _assert_same_state(selected, masked)


def test_reset_bookkeeping_writes_equal_scalar_index_writes() -> None:
    """``index_fill_`` replaced ``x[ids] = scalar``; the bits must not change."""
    sampler = _sampler(num_envs=5)
    ids = torch.tensor([3, 0])
    for name in ("start_bin", "ep_step", "ep_disc_ret", "ep_ret"):
        getattr(sampler, name).add_(7)
    expected = {k: v.clone() for k, v in _tensors(sampler).items()}
    expected["start_bin"][ids] = sampler.bin_ids(
        torch.tensor([1, 0]), torch.tensor([3, 60])
    )
    expected["ep_step"][ids] = 0
    expected["ep_disc_ret"][ids] = 0.0
    expected["ep_ret"][ids] = 0.0
    sampler.begin_episodes(ids, torch.tensor([1, 0]), torch.tensor([3, 60]))
    for name, tensor in _tensors(sampler).items():
        assert tensor.dtype == expected[name].dtype, name
        assert torch.equal(tensor, expected[name]), name
    sampler.ep_step.fill_(1)
    sampler.end_episodes(ids, torch.tensor([True, True]))
    assert sampler.start_bin[ids].tolist() == [-1, -1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_reset_bookkeeping_never_waits_for_the_device() -> None:
    device = torch.device("cuda")
    sampler = G1ResetSampler(
        torch.tensor([120, 55], device=device), num_envs=4, discount=0.9
    )
    ids = torch.arange(4, device=device)
    reward = torch.full((4,), 0.1, device=device)
    terminated = torch.tensor([True, False, True, False], device=device)
    completed = torch.tensor([True, True, False, True], device=device)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        ranks, frames, bins = sampler.sample(4)
        sampler.begin_episodes(ids, ranks, frames, bins=bins)
        sampler.accumulate(reward)
        sampler.end_episodes(ids, terminated, completed=completed)
        sampler.end_episodes(ids, terminated)
    finally:
        torch.cuda.set_sync_debug_mode("default")


def _one_episode(sampler, rank, frame, steps, terminated, *, bins=None):
    """Run one episode on env 0 and end it, as the env does."""
    ids = torch.tensor([0])
    sampler.begin_episodes(
        ids,
        torch.tensor([rank]),
        torch.tensor([frame]),
        bins=None if bins is None else torch.tensor([bins]),
    )
    reward = torch.zeros(sampler.num_envs)
    for _ in range(steps):
        sampler.accumulate(reward)
    sampler.end_episodes(ids, torch.tensor([terminated]))


def test_share_zero_holds_no_failure_state_and_draws_the_base_formula() -> None:
    sampler = _sampler()
    assert not {"bin_fail", "bin_prev", "start_frame"} & vars(sampler).keys()
    sampler.bin_G[:] = torch.tensor([0.3, -0.2, 0.0, 0.1, 0.4])
    memory = sampler.bin_w_avg.clone()
    soft = torch.softmax(-25.0 * sampler.bin_G / sampler.G_max, dim=0)
    memory = memory.mul(0.999).add(0.001 * soft)
    probs = 0.5 / 5 + 0.4 * soft + 0.1 * memory / memory.sum()
    generator = torch.Generator().manual_seed(53)
    expected = torch.multinomial(probs, 256, replacement=True, generator=generator)
    _, _, bins = sampler.sample(256)
    assert torch.equal(bins, expected)
    assert torch.equal(sampler.bin_w_avg, memory)
    torch.rand(256, generator=generator)  # the frame draw
    assert torch.equal(sampler.generator.get_state(), generator.get_state())


def test_previous_bins_stop_at_the_start_of_each_clip() -> None:
    sampler = _sampler(fail_share=0.3)
    assert sampler.bin_prev.tolist() == [0, 0, 1, 3, 3]
    lengths = torch.randint(
        120, 9000, (40,), generator=torch.Generator().manual_seed(1)
    )
    many = G1ResetSampler(lengths, num_envs=1, fail_share=0.3)
    first = many.bin_offsets
    assert torch.equal(many.bin_prev[first], first)
    others = torch.ones(many.num_bins, dtype=torch.bool)
    others[first] = False
    ids = torch.arange(many.num_bins)
    assert torch.equal(many.bin_prev[others], ids[others] - 1)
    assert (many.bin_prev == ids).sum() == len(lengths)


@pytest.mark.parametrize(("steps", "failing_bin"), [(10, 0), (11, 1)])
def test_a_failure_is_charged_to_the_pre_step_frame(steps, failing_bin) -> None:
    """Cursor 50 means the last pre-step frame was 49, still local bin 0."""
    sampler = _sampler(fail_share=0.3)
    _one_episode(sampler, 0, 40, steps, True)
    expected = torch.zeros(5)
    expected[failing_bin] = 1 - 0.9
    torch.testing.assert_close(sampler.bin_fail, expected)


def test_a_singleton_start_credits_returns_and_failures_to_different_bins() -> None:
    sampler = _sampler(lengths=(101,), num_envs=1, fail_share=0.3)
    assert sampler.num_bins == 3
    # Sampling bin 2 selects the penultimate frame, which lies in bin 1.
    sampler.begin_episodes(
        torch.tensor([0]), torch.tensor([0]), torch.tensor([99]), bins=torch.tensor([2])
    )
    sampler.accumulate(torch.tensor([1.0]))
    sampler.end_episodes(torch.tensor([0]), torch.tensor([True]))
    assert sampler.bin_G.nonzero().flatten().tolist() == [2]
    assert sampler.bin_fail.nonzero().flatten().tolist() == [1]


def test_an_episode_past_its_clip_end_stays_in_its_clip() -> None:
    sampler = _sampler(lengths=(60, 60), fail_share=0.3)
    _one_episode(sampler, 0, 55, 20, True)
    assert sampler.bin_fail.nonzero().flatten().tolist() == [1]


def test_mix_moves_the_share_to_the_bin_before_the_failure() -> None:
    sampler = _sampler(fail_share=0.3)
    base = _sampler().sampling_probabilities()
    torch.testing.assert_close(sampler.sampling_probabilities(), base)
    _one_episode(sampler, 0, 60, 5, True)  # ends in bin 1, so bin 0 gets the mass
    expected = 0.7 * base
    expected[0] += 0.3
    # The episode's zero return raises no bin, so `base` still applies.
    torch.testing.assert_close(sampler.sampling_probabilities(), expected)


def test_first_bin_failures_stay_on_the_first_bin() -> None:
    sampler = _sampler(fail_share=0.3)
    _one_episode(sampler, 1, 10, 3, True)
    probs = sampler.sampling_probabilities()
    assert probs.argmax() == 3
    assert probs[3] == pytest.approx(0.7 * 0.2 + 0.3)


def test_a_failure_reaches_the_very_next_draw() -> None:
    """End before sample, as `_reset_idx` orders them: nothing is one step stale."""
    sampler = _sampler(fail_share=0.9)
    _one_episode(sampler, 0, 60, 5, True)
    _, _, bins = sampler.sample(4096)
    assert (bins == 0).float().mean() > 0.85


def test_sampling_memory_advances_on_the_softmax_alone() -> None:
    plain, mixed = _sampler(), _sampler(fail_share=0.5)
    _one_episode(plain, 0, 60, 5, True)
    _one_episode(mixed, 0, 60, 5, True)
    plain.sample(8)
    mixed.sample(8)
    assert torch.equal(plain.bin_w_avg, mixed.bin_w_avg)
    assert torch.equal(plain.bin_G, mixed.bin_G)


def test_failure_statistics_equal_the_reset_stats_formula() -> None:
    """The source formulas, restated, over rounds with duplicate bins."""

    def bin_ema(stat, bins, x, old):
        sums = torch.zeros_like(stat).index_add_(0, bins, x)
        cnt = torch.zeros_like(stat).index_add_(0, bins, torch.ones_like(x))
        return torch.where(
            cnt > 0, old * stat + (1 - old) * sums / cnt.clamp(min=1), stat
        )

    def mix_failures(probs, bin_fail, bin_prev, share):
        fail = torch.zeros_like(bin_fail).index_add_(0, bin_prev, bin_fail)
        fail = fail / fail.sum().clamp(min=1e-6)
        return (1 - share) * probs + share * torch.where(fail.sum() > 0, fail, probs)

    generator = torch.Generator().manual_seed(7)
    envs = 16
    sampler = _sampler(lengths=(320, 180, 75), num_envs=envs, fail_share=0.3)
    expected = torch.zeros(sampler.num_bins)
    ids = torch.arange(envs)
    for _ in range(30):
        ranks = torch.randint(0, 3, (envs,), generator=generator)
        frames = (
            torch.rand(envs, generator=generator)
            * (sampler.lengths[ranks] - 40).float()
        ).long()
        steps = torch.randint(1, 30, (envs,), generator=generator)
        terminated = torch.rand(envs, generator=generator) < 0.5
        sampler.begin_episodes(ids, ranks, frames)
        sampler.ep_step[:] = steps
        sampler.end_episodes(ids, terminated)
        expected.mul_(0.999)
        expected = bin_ema(
            expected,
            sampler.bin_ids(ranks, frames + steps - 1),
            terminated.float(),
            0.9,
        )
        assert torch.equal(sampler.bin_fail, expected)
    base = torch.full((sampler.num_bins,), 1.0 / sampler.num_bins)
    torch.testing.assert_close(
        sampler._mix_failures(base),
        mix_failures(base, expected, sampler.bin_prev, 0.3),
        rtol=0,
        atol=0,
    )


def test_only_a_step_that_ends_an_episode_fades_the_statistic() -> None:
    sampler = _sampler(fail_share=0.3)
    _one_episode(sampler, 0, 60, 5, True)
    before = sampler.bin_fail.clone()
    ids = torch.tensor([1])
    sampler.begin_episodes(ids, torch.tensor([0]), torch.tensor([0]))
    sampler.end_episodes(ids, torch.tensor([True]), completed=torch.tensor([False]))
    assert torch.equal(sampler.bin_fail, before)
    sampler.end_episodes(ids, torch.tensor([True]))  # zero steps: no evidence
    assert torch.equal(sampler.bin_fail, before)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_a_diverged_episode_leaves_returns_but_counts_as_a_failure(value) -> None:
    """Only `bin_G` omits a non-finite return; the episode still ended in its bin."""
    sampler = _sampler(fail_share=0.3)
    ids = torch.tensor([0])
    sampler.begin_episodes(ids, torch.tensor([0]), torch.tensor([60]))
    sampler.accumulate(torch.tensor([value, 0.0, 0.0, 0.0]))
    sampler.end_episodes(ids, torch.tensor([True]))
    assert torch.equal(sampler.bin_G, torch.zeros_like(sampler.bin_G))
    torch.testing.assert_close(sampler.bin_fail[1], torch.tensor(1 - 0.9))


def test_a_timeout_lowers_its_bin_and_fades_the_others() -> None:
    sampler = _sampler(fail_share=0.3)
    _one_episode(sampler, 0, 60, 5, True)
    _one_episode(sampler, 0, 0, 5, False)
    first = 1 - 0.9
    torch.testing.assert_close(sampler.bin_fail[1], torch.tensor(first * 0.999))
    _one_episode(sampler, 0, 60, 5, False)
    torch.testing.assert_close(
        sampler.bin_fail[1], torch.tensor(first * 0.999 * 0.999 * 0.9)
    )


@pytest.mark.parametrize(
    "mask",
    [
        [True, True, True, True],
        [True, False, True, False],
        [False, False, False, False],
    ],
)
def test_masked_credit_equals_selection_with_failure_statistics(mask) -> None:
    pair = _sampler(num_envs=5, fail_share=0.3), _sampler(num_envs=5, fail_share=0.3)
    ids = torch.tensor([0, 1, 2, 3])
    for sampler in pair:
        sampler.begin_episodes(
            ids, torch.tensor([0, 0, 1, 0]), torch.tensor([3, 70, 10, 110])
        )
        sampler.accumulate(torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]))
    selected, masked = pair
    terminated = torch.tensor([True, False, True, False])
    completed = torch.tensor(mask)
    if completed.any():
        selected.end_episodes(ids[completed], terminated[completed])
    masked.end_episodes(ids, terminated, completed=completed)
    _assert_same_state(selected, masked)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_failure_statistics_never_wait_for_the_device() -> None:
    device = torch.device("cuda")
    sampler = G1ResetSampler(
        torch.tensor([120, 55], device=device),
        num_envs=4,
        discount=0.9,
        fail_share=0.3,
    )
    ids = torch.arange(4, device=device)
    reward = torch.full((4,), 0.1, device=device)
    terminated = torch.tensor([True, False, True, False], device=device)
    completed = torch.tensor([True, True, False, True], device=device)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        ranks, frames, bins = sampler.sample(4)
        sampler.begin_episodes(ids, ranks, frames, bins=bins)
        sampler.accumulate(reward)
        sampler.end_episodes(ids, terminated, completed=completed)
        sampler.sample(4)
    finally:
        torch.cuda.set_sync_debug_mode("default")


def _trained(fail_share=0.3):
    sampler = _sampler(fail_share=fail_share)
    _one_episode(sampler, 0, 60, 5, True)
    _one_episode(sampler, 1, 10, 7, False)
    sampler.bin_G[:] = torch.tensor([0.3, -0.2, 0.0, 0.1, 0.4])
    sampler.sample(16)
    return sampler


def test_a_resumed_sampler_draws_what_the_saved_one_would() -> None:
    saved = _trained()
    resumed = _sampler(fail_share=0.3)
    resumed.load_state_dict(saved.state_dict())
    for name in ("bin_G", "bin_w_avg", "bin_fail"):
        assert torch.equal(getattr(resumed, name), getattr(saved, name)), name
    assert torch.equal(resumed.sampling_probabilities(), saved.sampling_probabilities())
    saved.generator.manual_seed(11)
    resumed.generator.manual_seed(11)
    for a, b in zip(saved.sample(256), resumed.sample(256)):
        assert torch.equal(a, b)
    assert torch.equal(resumed.bin_w_avg, saved.bin_w_avg)


def test_saved_statistics_do_not_alias_the_live_sampler() -> None:
    sampler = _trained()
    state = sampler.state_dict()
    frozen = state["bin_G"].clone()
    sampler.bin_G.add_(1.0)
    assert torch.equal(state["bin_G"], frozen)


def test_episodes_in_flight_do_not_cross_a_segment() -> None:
    saved = _sampler()
    saved.begin_episodes(torch.tensor([0]), torch.tensor([0]), torch.tensor([10]))
    saved.accumulate(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    state = saved.state_dict()
    assert set(state) == {"lengths", "bin_G", "bin_w_avg"}
    resumed = _sampler()
    resumed.load_state_dict(state)
    assert resumed.start_bin.tolist() == [-1] * 4
    assert resumed.ep_step.tolist() == [0] * 4


def test_statistics_of_another_bank_are_refused() -> None:
    with pytest.raises(ValueError, match="other clips"):
        _sampler(lengths=(120, 56)).load_state_dict(_trained().state_dict())


def test_failure_share_may_change_between_segments() -> None:
    on = _sampler(fail_share=0.3)
    on.load_state_dict(_trained(fail_share=0.0).state_dict())
    assert not on.bin_fail.any()
    off = _sampler()
    off.load_state_dict(_trained().state_dict())
    assert "bin_fail" not in vars(off)


class _Agent:
    """The checkpoint seam of WMBPTT, and nothing else of it."""

    def __init__(self) -> None:
        self.loaded = None

    def _checkpoint_payload(self, frames: int) -> dict:
        return {"cumulative_env_frames": frames}

    def save_model(self, path: Path, frames: int) -> None:
        torch.save(self._checkpoint_payload(frames), path)

    def load_model(self, path: Path) -> None:
        self.loaded = path


def _env(sampler, persist: bool):
    return SimpleNamespace(
        cfg=SimpleNamespace(persist_reset_stats=persist), g1_reset_sampler=sampler
    )


def test_persistence_off_leaves_the_agent_and_its_checkpoints_alone(tmp_path) -> None:
    agent = _Agent()
    install_reset_stats_persistence(agent, _env(_trained(), False))
    assert "_checkpoint_payload" not in vars(agent)
    assert "load_model" not in vars(agent)
    agent.save_model(tmp_path / "off.pt", 3)
    assert torch.load(tmp_path / "off.pt", weights_only=False) == {
        "cumulative_env_frames": 3
    }


def test_an_env_cfg_without_the_field_counts_as_off() -> None:
    agent = _Agent()
    env = SimpleNamespace(cfg=SimpleNamespace())
    install_reset_stats_persistence(agent, env)
    assert "load_model" not in vars(agent)


def test_statistics_travel_inside_the_checkpoint_across_a_segment(tmp_path) -> None:
    saved = _trained()
    first = _Agent()
    install_reset_stats_persistence(first, _env(saved, True))
    path = tmp_path / "model_step_3.pt"
    first.save_model(path, 3)

    resumed = _sampler(fail_share=0.3)
    second = _Agent()
    load = second.load_model
    install_reset_stats_persistence(second, _env(resumed, True))
    second.load_model(path)
    assert second.load_model is not load and second.loaded == path
    for name in ("bin_G", "bin_w_avg", "bin_fail"):
        assert torch.equal(getattr(resumed, name), getattr(saved, name)), name


def test_a_resume_without_saved_statistics_is_refused(tmp_path) -> None:
    path = tmp_path / "model_step_3.pt"
    _Agent().save_model(path, 3)
    agent = _Agent()
    install_reset_stats_persistence(agent, _env(_sampler(), True))
    with pytest.raises(ValueError, match="persist_reset_stats=false"):
        agent.load_model(path)


def test_an_agent_without_the_seam_is_refused() -> None:
    with pytest.raises(ValueError, match="_checkpoint_payload"):
        install_reset_stats_persistence(SimpleNamespace(), _env(_sampler(), True))


@pytest.mark.parametrize("share", [-0.1, 1.0, float("nan")])
def test_invalid_failure_share_fails_before_sampling(share: float) -> None:
    with pytest.raises(ValueError, match="fail_share"):
        _sampler(fail_share=share)


@pytest.mark.parametrize("lengths", [[], [1], [0, 100], [-2], [1.5, 4.0]])
def test_invalid_lengths_fail_before_sampling(lengths) -> None:
    with pytest.raises(ValueError):
        G1ResetSampler(torch.tensor(lengths), num_envs=1)


@pytest.mark.parametrize("discount", [0, 1, -1, float("nan")])
def test_invalid_discounts_fail_before_sampling(discount: float) -> None:
    with pytest.raises(ValueError):
        _sampler(discount=discount)


# ----------------------------------------------------------------------------
# Excluded clips (`env.reset_exclude_names_file`).
# ----------------------------------------------------------------------------


def test_excluded_clips_take_no_probability_in_any_term() -> None:
    """Bins (0,1,2 | 3,4 | 5,6): clip 1 out leaves five bins sharing 0.5 / 0.4 / 0.1."""
    sampler = _sampler(lengths=(120, 55, 80))
    sampler.exclude(torch.tensor([1]))
    assert sampler.bin_keep.tolist() == [True, True, True, False, False, True, True]
    kept = torch.tensor([0.2, 0.2, 0.2, 0.0, 0.0, 0.2, 0.2])
    torch.testing.assert_close(sampler.bin_w_avg, kept)
    torch.testing.assert_close(sampler.sampling_probabilities(), kept)
    # A kept bin with a high return loses softmax share to the other kept bins
    # only. Without the mask inside the softmax the two excluded bins, at
    # bin_G = 0, would take 2/6 of it.
    sampler.bin_G[0] = (
        float(torch.log(torch.tensor(4.0))) / 25.0 * -1.0
    )  # exp(-25 G) = 4
    probs = sampler.sampling_probabilities()
    soft = torch.tensor([4.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]) / 8.0
    torch.testing.assert_close(probs, 0.5 * kept + 0.4 * soft + 0.1 * kept)
    assert float(probs.sum()) == pytest.approx(1.0)


def test_excluded_clips_are_never_drawn_by_either_sampling_path() -> None:
    sampler = _sampler(lengths=(120, 55, 80), num_envs=4, fail_share=0.5)
    sampler.exclude(torch.tensor([1]))
    # Failure statistics on the excluded clip must not pull resets into it.
    sampler.bin_fail[3:5] = 1.0
    sampler.bin_fail[5] = 0.5
    assert sampler.sampling_probabilities()[3:5].tolist() == [0.0, 0.0]
    ranks, _, bins = sampler.sample(4000)
    assert not bool((ranks == 1).any())
    assert set(bins.tolist()) <= {0, 1, 2, 5, 6}
    generator = torch.Generator().manual_seed(7)
    u_bin = torch.rand(4000, generator=generator, dtype=torch.float64)
    u_bin[:3] = torch.tensor([0.0, 0.5, 1.0 - 2.0**-53], dtype=torch.float64)
    u_frame = torch.rand(4000, generator=generator)
    ranks, frames, bins = sampler.sample_masked(
        u_bin, u_frame, torch.ones((), dtype=torch.bool)
    )
    assert not bool((ranks == 1).any())
    assert set(bins.tolist()) <= {0, 1, 2, 5, 6}
    assert bool((frames >= 0).all())
    # The memory never grows on an excluded bin.
    assert sampler.bin_w_avg[3:5].tolist() == [0.0, 0.0]


def test_an_empty_exclusion_changes_no_draw() -> None:
    plain, masked = _sampler(), _sampler()
    masked.exclude(torch.zeros(0, dtype=torch.long))
    assert masked.bin_keep is None
    plain.bin_G[:] = torch.tensor([0.1, 0.0, 0.3, 0.2, 0.05])
    masked.bin_G[:] = plain.bin_G
    for a, b in zip(plain.sample(64), masked.sample(64)):
        assert torch.equal(a, b)
    assert torch.equal(plain.bin_w_avg, masked.bin_w_avg)


def test_invalid_exclusions_are_refused() -> None:
    with pytest.raises(ValueError, match="outside the bank"):
        _sampler().exclude(torch.tensor([2]))
    with pytest.raises(ValueError, match="no clip would remain"):
        _sampler().exclude(torch.tensor([0, 1]))
