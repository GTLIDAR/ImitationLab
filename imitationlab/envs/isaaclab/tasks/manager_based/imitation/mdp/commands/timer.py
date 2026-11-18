"""Command compute without the resample timer, for terms whose timer cannot fire.

``CommandTerm.compute`` checks ``(time_left <= 0).nonzero()`` every control
step, a host synchronization, to resample commands on a timer. The imitation
command terms set ``resampling_time_range`` to (1e9, 1e9) s: the reference
cursor is resampled by the environment at reset, never on a timer. Each reset
sets ``time_left`` from that range, and an episode would need ~5e10 control
steps to spend it, so the check is always empty and skipping it is exact. The
decrement is kept so ``time_left`` stays bit-identical.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from imitationlab.envs.isaaclab.envs.collect_fastpath import fastpath_enabled

if TYPE_CHECKING:
    from isaaclab.managers import CommandTerm

#: A lower resampling bound at or above this many seconds counts as "never".
NEVER_RESAMPLE_S = 1.0e8

_FAST_NEVER_RESAMPLE = fastpath_enabled("never_resample")


def compute_if_never_resampled(term: CommandTerm, dt: float) -> bool:
    """Run ``term.compute(dt)`` without the timer check if it cannot fire.

    Returns False (doing nothing) when the fast path is off or the term's
    timer is a real one; the caller then runs ``CommandTerm.compute``.
    """
    if not _FAST_NEVER_RESAMPLE:
        return False
    if float(term.cfg.resampling_time_range[0]) < NEVER_RESAMPLE_S:
        return False
    term._update_metrics()
    term.time_left -= dt
    term._update_command()
    return True
