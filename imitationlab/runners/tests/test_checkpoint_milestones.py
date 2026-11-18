from types import SimpleNamespace

import pytest

from imitationlab.runners.checkpointing import Checkpointing


class Clock(Checkpointing):
    def __init__(self, schedule=(), interval=0, restored=0):
        self.config = SimpleNamespace(
            save_interval=interval, trainer=SimpleNamespace(checkpoint_seconds=schedule)
        )
        self._train_seconds_offset = restored
        self.now = restored

    def train_elapsed_seconds(self):
        return self.now

    def at(self, now, frames=1, batch=1):
        self.now = now
        return self._should_save_checkpoint(
            frames_processed=frames, frames_in_iteration=batch
        )


def test_milestones_cross_at_completed_iterations_once():
    clock = Clock((60, 120, 300, 600))
    assert not clock.at(59.9)
    assert clock.at(60.2)
    assert not clock.at(61)
    assert clock.at(301)  # One checkpoint covers two crossed milestones.
    assert not clock.at(302)
    assert clock.at(600.1)
    assert not clock.at(601)


def test_resume_does_not_resave_past_milestones():
    clock = Clock((60, 120, 300, 600), restored=120.5)
    assert not clock.at(121)
    assert clock.at(300.5)


def test_frame_checkpoints_still_work_with_or_without_milestones():
    for schedule in ((), (60,)):
        clock = Clock(schedule, interval=100)
        assert not clock.at(1, 90, 10)
        assert clock.at(2, 105, 15)
        assert not clock.at(3, 110, 5)


@pytest.mark.parametrize(
    "schedule", [(0,), (-1,), (float("nan"),), (float("inf"),), (20, 10), (10, 10)]
)
def test_invalid_milestones_rejected(schedule):
    with pytest.raises(ValueError, match="checkpoint_seconds"):
        Clock(schedule).at(100)
