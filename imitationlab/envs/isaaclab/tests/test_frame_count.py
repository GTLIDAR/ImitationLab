"""Frame-based env schedules count global frames under data-parallel training."""

from __future__ import annotations

from types import SimpleNamespace as NS

from imitationlab.envs.isaaclab.envs.frame_count import schedule_frames


def test_schedule_frames_counts_every_rank() -> None:
    single = NS(common_step_counter=10, num_envs=16384, cfg=NS())
    ranked = NS(
        common_step_counter=10, num_envs=4096, cfg=NS(data_parallel_world_size=4)
    )
    assert schedule_frames(single) == schedule_frames(ranked) == 163840
    # Envs without the field (and test doubles without a cfg) count locally.
    assert schedule_frames(NS(common_step_counter=3, num_envs=5)) == 15
