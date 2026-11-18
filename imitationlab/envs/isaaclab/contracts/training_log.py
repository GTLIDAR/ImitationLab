"""The environment half of the training-log contract.

Every imitation environment publishes these entries in ``extras["log"]`` under
these exact names so log readers can use the same keys across recipes.
The algorithm half lives in ``imitationlab.algorithms.common.logging_utils``.
"""

from __future__ import annotations

from typing import Any

import torch

MPJPE_L_ENTRY = "Metrics/reference/mpjpe_l_mm"
MPJPE_G_ENTRY = "Metrics/reference/mpjpe_g_mm"
TIME_OUT_ENTRY = "Episode_Termination/time_out"
REFERENCE_FINISHED_ENTRY = "Episode_Termination/reference_finished"
TRACKING_FAILED_ENTRY = "Episode_Termination/tracking_failed"
REQUIRED_ENV_LOG_ENTRIES: tuple[str, ...] = (
    MPJPE_L_ENTRY,
    MPJPE_G_ENTRY,
    TIME_OUT_ENTRY,
    REFERENCE_FINISHED_ENTRY,
    TRACKING_FAILED_ENTRY,
)


def tracking_failure_term_mask(manager: Any, device: Any) -> torch.Tensor:
    """``[num_terms]`` bool: which of a termination manager's terms are failures.

    Everything but a time-out and the clip end. Built from Python values, so it
    copies from the host; build it once, outside any CUDA-graph capture.
    """

    return torch.tensor(
        [
            not (manager.get_term_cfg(name).time_out or name == "reference_finished")
            for name in manager.active_terms
        ],
        dtype=torch.bool,
        device=device,
    )


def tracking_failed_share(
    last_episode_dones: torch.Tensor, failure_terms: torch.Tensor
) -> torch.Tensor:
    """Share of environments whose latest episode ended on a failure term.

    Args:
        last_episode_dones: ``[num_envs, num_terms]`` bool, the terms that ended
            each environment's latest episode. It is the buffer behind
            ``Episode_Termination/<term>``, which makes this the same
            population share; summing those entries instead would count an
            environment twice when two failure terms fire on one step.
        failure_terms: ``[num_terms]`` bool, the terms that count as failure.
    """

    return (last_episode_dones & failure_terms).any(dim=1).float().mean()


__all__ = [
    "MPJPE_G_ENTRY",
    "MPJPE_L_ENTRY",
    "REFERENCE_FINISHED_ENTRY",
    "REQUIRED_ENV_LOG_ENTRIES",
    "TIME_OUT_ENTRY",
    "TRACKING_FAILED_ENTRY",
    "tracking_failed_share",
    "tracking_failure_term_mask",
]
