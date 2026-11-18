"""Value targets shared by imagined and real-window policy updates."""

from __future__ import annotations

import torch
from torch import Tensor


def lambda_returns(
    rewards: Tensor, values: Tensor, notdone: Tensor, gamma: float, lam: float
) -> Tensor:
    """TD(lambda) targets over a rollout.

    ``rewards`` and ``notdone`` are (B, H): the reward of step t and the
    probability the episode continues past it; ``values`` is (B, H + 1).
    R_H = V_H and R_t = r_t + gamma c_t ((1 - lam) V_{t+1} + lam R_{t+1}).
    """
    H = rewards.shape[1]
    targets = torch.zeros_like(rewards)
    nxt = values[:, H]
    for i in range(H - 1, -1, -1):
        nxt = rewards[:, i] + gamma * notdone[:, i] * (
            (1 - lam) * values[:, i + 1] + lam * nxt
        )
        targets[:, i] = nxt
    return targets
