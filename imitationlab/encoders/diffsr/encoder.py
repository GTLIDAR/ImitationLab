"""Continuous deterministic encoder used by DiffSR skill pretraining."""

from __future__ import annotations

from imitationlab.encoders.base import HighLevelSkillEncoder


class DeterministicSkillEncoder(HighLevelSkillEncoder):
    """Continuous skill encoder: the trunk output is the latent ``z``.

    The regularizer is the mean squared latent, ``mean(z ** 2)``.
    """

    def __init__(self, **base) -> None:
        super().__init__(raw_dim=base["z_dim"], **base)

    def _latent(self, raw, **_):
        return raw, raw.pow(2).mean(), {}
