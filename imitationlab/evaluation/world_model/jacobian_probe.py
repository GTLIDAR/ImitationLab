"""Directional derivatives of the simulator against a learned dynamics model.

Pure tensor helpers behind ``scripts/tracking/probe_model_jacobian.py``. The
script copies one simulator state into a block of environments and steps each
copy with a perturbed action. A block holds two unperturbed copies (their
disagreement is the solver's noise floor) and a +/- pair per direction and
step size, so a central difference gives the simulator's Jacobian-vector
product J v without any differentiable simulator.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

# Increment layout (``g1_task.increment``).
BLOCKS: dict[str, slice] = {
    "all": slice(0, 70),
    "root_pos": slice(0, 3),
    "root_rot": slice(3, 6),
    "root_lin_vel": slice(6, 9),
    "root_ang_vel": slice(9, 12),
    "joint_pos": slice(12, 41),
    "joint_vel": slice(41, 70),
}
PHASES = ("flight", "single", "double", "transition", "other_contact")


@dataclass(frozen=True)
class ReplicaLayout:
    """Rows of one block: 0, 1 unperturbed; then (delta d, direction k, sign)."""

    directions: int
    deltas: tuple[float, ...]

    @property
    def size(self) -> int:
        return 2 + 2 * self.directions * len(self.deltas)

    def row(self, delta_index: int, direction: int, sign: int) -> int:
        return (
            2 + 2 * (delta_index * self.directions + direction) + (0 if sign > 0 else 1)
        )

    def perturbations(self, directions: Tensor) -> Tensor:
        """Action offsets (M, size, A) from unit-coordinate ``directions`` (M, K, A)."""
        M, K, A = directions.shape
        out = directions.new_zeros(M, self.size, A)
        for d, delta in enumerate(self.deltas):
            for k in range(K):
                out[:, self.row(d, k, 1)] = delta * directions[:, k]
                out[:, self.row(d, k, -1)] = -delta * directions[:, k]
        return out

    def central_difference(self, values: Tensor, delta_index: int) -> Tensor:
        """(f(a + d v) - f(a - d v)) / (2 d) of ``values`` (M, size, ...) -> (M, K, ...)."""
        delta = self.deltas[delta_index]
        plus = torch.stack(
            [values[:, self.row(delta_index, k, 1)] for k in range(self.directions)], 1
        )
        minus = torch.stack(
            [values[:, self.row(delta_index, k, -1)] for k in range(self.directions)], 1
        )
        return (plus - minus) / (2 * delta)


def jvp_agreement(model: Tensor, sim: Tensor, eps: float = 1e-12) -> dict[str, Tensor]:
    """Per-row agreement of two (..., D) directional derivatives: ``cos``,
    ``rel_err`` ||m - s|| / ||s||, ``gain`` <m, s> / ||s||^2 (1 when the model's
    sensitivity is calibrated, under 1 when it is too small)."""
    dot = (model * sim).sum(-1)
    ms, ss = model.norm(dim=-1), sim.norm(dim=-1)
    return {
        "cos": dot / (ms * ss).clamp_min(eps),
        "rel_err": (model - sim).norm(dim=-1) / ss.clamp_min(eps),
        "gain": dot / ss.square().clamp_min(eps),
        "sim_norm": ss,
        "model_norm": ms,
    }


def contact_phase(foot: Tensor, other: Tensor, foot_block: Tensor) -> Tensor:
    """Phase index into ``PHASES`` of each block.

    ``foot`` (M, 2) bool: foot contact in the unperturbed copy; ``other`` (M,)
    bool: another body touches; ``foot_block`` (M, R, 2) bool: foot contact of
    every copy. "transition" means the perturbation changed the contact set
    within the step, the case a smooth model cannot follow.
    """
    n = foot.sum(-1)
    phase = n.clamp(max=2).clone()
    phase[(foot_block != foot.unsqueeze(1)).any(-1).any(-1)] = PHASES.index(
        "transition"
    )
    phase[other] = PHASES.index("other_contact")
    return phase


def summarize(
    values: dict[str, Tensor], select: Tensor | None = None
) -> dict[str, float]:
    """Medians and quartiles of ``jvp_agreement`` output, plus the pooled gain
    sum <m, s> / sum ||s||^2 and the number of rows."""
    out: dict[str, float] = {}
    keep = {
        k: (v[select] if select is not None else v).reshape(-1).double()
        for k, v in values.items()
    }
    n = keep["cos"].numel()
    out["n"] = float(n)
    if n == 0:
        return out
    for key in ("cos", "rel_err", "gain"):
        q = torch.quantile(
            keep[key],
            torch.tensor(
                [0.25, 0.5, 0.75], dtype=torch.float64, device=keep[key].device
            ),
        )
        out[f"{key}_p25"], out[f"{key}_median"], out[f"{key}_p75"] = (
            float(x) for x in q
        )
    s2 = keep["sim_norm"].square()
    out["gain_pooled"] = float((keep["gain"] * s2).sum() / s2.sum().clamp_min(1e-30))
    out["sim_norm_median"] = float(keep["sim_norm"].median())
    out["model_norm_median"] = float(keep["model_norm"].median())
    return out


def scalar_agreement(model: Tensor, sim: Tensor) -> dict[str, float]:
    """Pearson correlation, regression slope of sim on model, sign agreement
    and rows of two scalar directional derivatives (any shape)."""
    m, s = model.reshape(-1).double(), sim.reshape(-1).double()
    mc, sc = m - m.mean(), s - s.mean()
    corr = (mc * sc).sum() / (mc.norm() * sc.norm()).clamp_min(1e-30)
    return {
        "corr": float(corr),
        "slope_sim_on_model": float(
            (mc * sc).sum() / mc.square().sum().clamp_min(1e-30)
        ),
        "sign_agreement": float(((m > 0) == (s > 0)).double().mean()),
        "n": float(m.numel()),
    }
