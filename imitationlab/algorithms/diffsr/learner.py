"""Policy learning through a DiffSR dynamics model."""

from __future__ import annotations

from torch import nn

from imitationlab.algorithms.common.world_model.learner import WorldModelLearner
from imitationlab.algorithms.diffsr.training import build_dynamics


class DiffSR(WorldModelLearner):
    """Fit diffusion spectral features and use their mean readout for BPTT."""

    def _construct_dynamics(self) -> nn.Module:
        if self.config.wm.dynamics_model != "diffsr":
            return super()._construct_dynamics()
        return build_dynamics(self.task, self.config.wm)
