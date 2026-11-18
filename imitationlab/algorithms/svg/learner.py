"""Stochastic value gradients on recorded simulator windows."""

from __future__ import annotations

from torch import Tensor

from imitationlab.algorithms.common.world_model.learner import WorldModelLearner


class SVG(WorldModelLearner):
    """Use inferred action noise and model residuals to replay actor updates."""

    def _actor_rollout(self) -> dict[str, Tensor]:
        wm = self.config.wm
        if wm.actor_source != "real":
            return super()._actor_rollout()
        return self._real_rollout(wm.imagine_batch, grad=True)
