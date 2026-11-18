"""Finite scalar quantization and normalized FSQ skill codes."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from imitationlab.encoders.base import _DiscreteSkillEncoder


class FSQQuantizer(nn.Module):
    """Finite Scalar Quantization: per-dim bounded rounding. No params, no collapse."""

    def __init__(self, levels: tuple[int, ...]) -> None:
        super().__init__()
        levels_i = torch.tensor([int(level) for level in levels], dtype=torch.long)
        if levels_i.numel() == 0 or int(levels_i.min()) < 2:
            msg = f"FSQ levels must each be >= 2, got {list(levels)!r}."
            raise ValueError(msg)
        self.register_buffer("_levels", levels_i)
        # Exact Python-int product: torch int64 prod/cumprod silently wrap for
        # large level sets (e.g. SONIC's 64 dims x 32 levels = 2^320).
        self._codebook_size = math.prod(int(level) for level in levels)
        # A flat scalar code index only exists when the product fits int64;
        # otherwise forward() returns per-dim level indices instead (the same
        # pooled-code convention the grouped categorical encoder uses).
        self.flat_code_supported = self._codebook_size <= (1 << 62)
        basis = torch.ones(levels_i.numel(), dtype=torch.long)
        if self.flat_code_supported:
            basis = torch.cumprod(
                torch.cat([torch.ones(1, dtype=torch.long), levels_i[:-1]]), dim=0
            )
        self.register_buffer("_basis", basis)
        self.code_dim = int(levels_i.numel())

    @property
    def codebook_size(self) -> int:
        return self._codebook_size

    def _bound(self, z: Tensor) -> Tensor:
        levels = self._levels.to(z.dtype)
        half = (levels - 1) * 0.5
        offset = (self._levels % 2 == 0).to(z.dtype) * 0.5
        shift = torch.atanh(offset / half.clamp(min=1.0))
        return (z + shift).tanh() * half - offset

    def forward(self, z_e: Tensor) -> tuple[Tensor, Tensor]:
        bounded = self._bound(z_e)
        rounded = torch.round(bounded)
        z_q = bounded + (rounded - bounded).detach()  # straight-through
        zhat = (rounded.long() + self._levels // 2).clamp(min=0)
        zhat = torch.minimum(zhat, self._levels - 1)
        if not self.flat_code_supported:
            return z_q, zhat
        code = (zhat * self._basis).sum(dim=-1)
        return z_q, code


class FSQSkillEncoder(_DiscreteSkillEncoder):
    """FSQ-bottlenecked skill encoder with a learned code-to-latent projection.

    The trunk emits ``len(levels)`` values, :class:`FSQQuantizer` rounds them
    onto the lattice (straight-through), and ``code_to_latent`` maps the result
    to a continuous ``z_dim`` command. See :class:`SONICFSQSkillEncoder` for the
    variant that publishes the lattice values directly.
    """

    def __init__(self, *, levels: tuple[int, ...] = (8, 8, 8, 5, 5), **base) -> None:
        super().__init__(raw_dim=len(levels), **base)
        self.fsq = FSQQuantizer(levels)
        # When the flat code index does not fit int64, per-dim level indices
        # are the codes (pooled across dims by the usage metrics), so the
        # usage-metric vocabulary is the largest per-dim level count.
        self.num_codes = (
            self.fsq.codebook_size
            if self.fsq.flat_code_supported
            else max(int(level) for level in levels)
        )
        # `perplexity` is a flat-codebook statistic. Without a flat index the
        # codes are per-dimension level indices, so bincounting them pools
        # every dimension into one histogram and the number means nothing --
        # the same reason MultiCategoricalSkillEncoder opts out. SONIC's
        # 64 x 32 lattice is 2^320 and always lands here.
        self.report_flat_code_metrics = bool(self.fsq.flat_code_supported)
        self.code_to_latent = nn.Linear(self.fsq.code_dim, base["z_dim"])

    def _quantize(self, z_e, **_):
        z_q, code = self.fsq(z_e)
        return z_q, code, z_e.new_zeros(()), {}


class SONICFSQSkillEncoder(_DiscreteSkillEncoder):
    """SONIC-motivated FSQ bottleneck: the quantizer output *is* the command.

    Two differences from :class:`FSQSkillEncoder`, both required for the
    planner -> tracker interface to actually be quantized rather than merely
    trained through a quantizer:

    1. **Normalized codes.** Emits ``round(bound(z)) / (L // 2)``, so values land
       on the lattice in ``[-1, 1]``. This matches the released SONIC token
       space: every entry of ``gear_sonic``'s ``LATENT_INITIAL_MOTION_TOKEN`` is
       an exact multiple of ``1/16 = 1/(32 // 2)``.
    2. **No projection before the command boundary.** :class:`FSQSkillEncoder`
       applies ``code_to_latent = nn.Linear(code_dim, z_dim)`` inside
       ``_latent``, so the published command is a ``z_dim`` continuous vector and
       every point in R^z_dim is admissible -- the interface is not quantized,
       only the training was. Here ``code_to_latent`` is the identity and
       ``z_dim == len(levels)``, so the tracker observes the lattice values
       directly and its own first layer plays the role of SONIC's ``g1_dyn``
       input projection.

    :class:`FSQQuantizer` is reused unmodified: dividing its straight-through
    output by a constant preserves the estimator -- the forward value becomes
    ``rounded / half`` and the gradient is scaled by ``1 / half``.
    """

    def __init__(self, *, levels: tuple[int, ...] = (32,) * 64, **base) -> None:
        z_dim = int(base["z_dim"])
        if z_dim != len(levels):
            msg = (
                "sonic_fsq publishes the quantizer output directly, so z_dim must "
                f"equal the number of FSQ dimensions: z_dim={z_dim}, "
                f"len(levels)={len(levels)}."
            )
            raise ValueError(msg)
        super().__init__(raw_dim=len(levels), **base)
        self.fsq = FSQQuantizer(levels)
        # Same pooled-code convention as FSQSkillEncoder: a flat code index only
        # exists when the level product fits int64 (SONIC's 64 x 32 is 2^320).
        self.num_codes = (
            self.fsq.codebook_size
            if self.fsq.flat_code_supported
            else max(int(level) for level in levels)
        )
        # `perplexity` is a flat-codebook statistic. Without a flat index the
        # codes are per-dimension level indices, so bincounting them pools
        # every dimension into one histogram and the number means nothing --
        # the same reason MultiCategoricalSkillEncoder opts out. SONIC's
        # 64 x 32 lattice is 2^320 and always lands here.
        self.report_flat_code_metrics = bool(self.fsq.flat_code_supported)
        # The command boundary sits at the quantizer output -- no learned map.
        self.code_to_latent = nn.Identity()
        self.register_buffer(
            "_half_levels",
            torch.tensor(
                [max(int(level) // 2, 1) for level in levels], dtype=torch.float32
            ),
        )

    def _quantize(self, z_e, **_):
        z_q, code = self.fsq(z_e)
        return z_q / self._half_levels.to(z_q.dtype), code, z_e.new_zeros(()), {}
