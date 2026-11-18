"""SuperMimic policy updates through imagined model rollouts."""

from __future__ import annotations

from torch import Tensor

from imitationlab.algorithms.common.world_model.learner import WorldModelLearner


class SuperMimic(WorldModelLearner):
    """Optimize the actor through imagined trajectories with a value bootstrap."""

    def _actor_rollout(self) -> dict[str, Tensor]:
        wm = self.config.wm
        if wm.actor_source != "imagined":
            return super()._actor_rollout()
        return self._imagine(
            wm.imagine_batch, grad=True, start_perturb=wm.imag_start_perturb
        )
