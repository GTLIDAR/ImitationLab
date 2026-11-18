"""Shared encoder trunk, latent configuration, and usage diagnostics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor, nn


def _perplexity(code: Tensor, num_codes: int) -> Tensor:
    counts = torch.bincount(code.reshape(-1), minlength=num_codes).float()
    probs = counts / counts.sum().clamp(min=1.0)
    return torch.exp(-(probs * probs.clamp_min(1e-12).log()).sum())


def _effective_rank(z: Tensor) -> Tensor:
    """Participation-ratio effective rank of ``z`` [B, D] as a 0-dim tensor."""
    if z.shape[0] < 2:
        return z.new_zeros(())
    sv = torch.linalg.svdvals(z - z.mean(0, keepdim=True))
    total = sv.sum()
    if bool((total <= 1e-12).item()):
        return z.new_zeros(())
    probs = sv / total
    return torch.exp(-(probs * probs.clamp_min(1e-12).log()).sum())


class HighLevelSkillEncoder(nn.Module, ABC):
    """Shared MLP trunk over ``[state ; future_window]`` + a latent head.

    ``encode`` returns ``(z [B, z_dim], reg_loss scalar, info dict)``. ``reg_loss``
    is the latent regularizer, weighted by a single ``reg_coeff``: L2 on z
    (deterministic), KL (gaussian / categorical / gumbel), commitment (vq), 0 (fsq).
    """

    def __init__(
        self,
        *,
        state_dim: int,
        window_steps: int,
        z_dim: int,
        hidden_dims: tuple[int, ...],
        raw_dim: int,
        activation: str = "mish",
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.window_steps = int(window_steps)
        self.z_dim = int(z_dim)
        layers: list[nn.Module] = []
        prev = self.state_dim * (self.window_steps + 1)
        activation_name = str(activation).strip().lower()
        activation_types: dict[str, type[nn.Module]] = {
            "elu": nn.ELU,
            "mish": nn.Mish,
            "relu": nn.ReLU,
            "silu": nn.SiLU,
        }
        try:
            activation_type = activation_types[activation_name]
        except KeyError as error:
            msg = (
                "activation must be one of "
                f"{sorted(activation_types)}, got {activation!r}."
            )
            raise ValueError(msg) from error
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev, int(hidden)))
            if layer_norm:
                layers.append(nn.LayerNorm(int(hidden)))
            layers.append(activation_type())
            prev = int(hidden)
        layers.append(nn.Linear(prev, int(raw_dim)))
        self.net = nn.Sequential(*layers)

    def _raw(self, state: Tensor, future_window: Tensor) -> Tensor:
        if state.ndim != 2:
            msg = f"state must have shape [B, D], got {tuple(state.shape)}."
            raise ValueError(msg)
        if future_window.ndim != 3:
            msg = (
                "future_window must have shape [B, W, D], got "
                f"{tuple(future_window.shape)}."
            )
            raise ValueError(msg)
        batch_size, state_dim = state.shape
        expected_window = (batch_size, self.window_steps, state_dim)
        if (
            int(state_dim) != self.state_dim
            or tuple(future_window.shape) != expected_window
        ):
            msg = (
                f"state/future_window shape mismatch: expected state [B, {self.state_dim}] "
                f"and future_window {expected_window}, got {tuple(state.shape)} and "
                f"{tuple(future_window.shape)}."
            )
            raise ValueError(msg)
        flat_window = future_window.reshape(batch_size, self.window_steps * state_dim)
        return self.net(torch.cat([state, flat_window], dim=-1))

    @abstractmethod
    def _latent(
        self, raw: Tensor, *, deterministic: bool, step: int | None
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]: ...

    def encode(
        self,
        state: Tensor,
        future_window: Tensor,
        *,
        deterministic: bool = False,
        step: int | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        return self._latent(
            self._raw(state, future_window), deterministic=deterministic, step=step
        )

    def forward(self, state: Tensor, future_window: Tensor) -> Tensor:
        return self.encode(state, future_window, deterministic=True)[0]

    def on_after_train_step(self, step: int) -> None:
        """Called after each optimizer step (VQ overrides for dead-code revival)."""

    def _z_diversity(self, z: Tensor) -> dict[str, Tensor]:
        """Universal continuous collapse signal: rank + per-dim spread of ``z``."""
        std = z.std(dim=0, unbiased=False)
        return {
            "effective_rank": _effective_rank(z),
            "z_std_mean": std.mean(),
            "z_std_min": std.min(),
        }

    @torch.no_grad()
    def diversity_metrics(
        self, state: Tensor, future_window: Tensor
    ) -> dict[str, Tensor]:
        """Per-method diversity / collapse diagnostics (eval calls this)."""
        return self._z_diversity(self.forward(state, future_window))


class _DiscreteSkillEncoder(HighLevelSkillEncoder):
    """Shared discrete flow: trunk -> quantize -> code_to_latent -> z."""

    code_to_latent: nn.Module
    num_codes: int = 0
    report_flat_code_metrics: bool = True

    def _pre_quantize(self, raw: Tensor) -> Tensor:
        return raw

    @abstractmethod
    def _quantize(
        self, z_e: Tensor, *, deterministic: bool, step: int | None
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        """Return ``(z_q, code, reg_loss, info)``."""

    def _latent(self, raw, *, deterministic, step):
        z_e = self._pre_quantize(raw)
        z_q, code, reg, info = self._quantize(
            z_e, deterministic=deterministic, step=step
        )
        z = self.code_to_latent(z_q).reshape(*raw.shape[:-1], self.z_dim)
        if self.report_flat_code_metrics and 0 < self.num_codes <= 1 << 16:
            info = {**info, "perplexity": _perplexity(code.detach(), self.num_codes)}
        return z, reg, info

    @torch.no_grad()
    def diversity_metrics(self, state, future_window):
        raw = self._raw(state, future_window)
        z_e = self._pre_quantize(raw)
        z_q, code, _, _ = self._quantize(z_e, deterministic=True, step=None)
        z = self.code_to_latent(z_q).reshape(*raw.shape[:-1], self.z_dim)
        metrics = self._z_diversity(z)
        # Codebook collapse: effective # of codes used / fraction of the codebook.
        metrics["code_perplexity"] = _perplexity(code, self.num_codes)
        used = float(code.reshape(-1).unique().numel())
        metrics["code_usage_frac"] = z.new_tensor(used / self.num_codes)
        return metrics


@dataclass(frozen=True)
class SkillLatentSpec:
    """Latent hyperparameters; separate from the trainer config to avoid a cycle."""

    latent_mode: str = "deterministic"
    gaussian_logstd_min: float = -5.0
    gaussian_logstd_max: float = 2.0
    categorical_groups: int = 8
    categorical_categories: int = 32
    gumbel_codebook_size: int = 512
    gumbel_tau_start: float = 2.0
    gumbel_tau_end: float = 0.5
    gumbel_tau_anneal_iters: int = 2000
    gumbel_hard: bool = True
    fsq_levels: tuple[int, ...] = (8, 8, 8, 5, 5)
    # SONIC-matched token space: 64 dims x 32 levels ~= 320 bits per command,
    # i.e. gear_sonic's tokens of shape (2, 32) at num_fsq_levels=32.
    sonic_fsq_levels: tuple[int, ...] = (32,) * 64
    vq_codebook_size: int = 512
    vq_ema_decay: float = 0.99
    vq_dead_code_reset_iters: int = 0
