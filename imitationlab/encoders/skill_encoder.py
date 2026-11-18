"""Build the skill encoder specified by a recipe or checkpoint."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from imitationlab.encoders.base import (
    HighLevelSkillEncoder,
    SkillLatentSpec,
    _DiscreteSkillEncoder,
)
from imitationlab.encoders.diffsr.encoder import DeterministicSkillEncoder
from imitationlab.encoders.fsq.encoder import FSQQuantizer as FSQQuantizer
from imitationlab.encoders.fsq.encoder import FSQSkillEncoder, SONICFSQSkillEncoder

LATENT_MODES = (
    "deterministic",
    "gaussian",
    "categorical",
    "gumbel_multicat",
    "gumbel",
    "fsq",
    "sonic_fsq",
    "vq",
)


class EMAVQQuantizer(nn.Module):
    """VQ-VAE quantizer: EMA codebook + commitment loss, with dead-code revival."""

    def __init__(
        self,
        codebook_size: int,
        embed_dim: int,
        *,
        decay: float = 0.99,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.embed_dim = int(embed_dim)
        self.decay, self.eps = float(decay), float(eps)
        embedding = torch.randn(self.codebook_size, self.embed_dim) * 0.01
        self.register_buffer("embedding", embedding)
        self.register_buffer("cluster_size", torch.zeros(self.codebook_size))
        self.register_buffer("cluster_sum", embedding.clone())
        self.register_buffer("_inited", torch.tensor(False))

    @torch.no_grad()
    def _kmeans_init(self, z_e: Tensor) -> None:
        flat = z_e.reshape(-1, self.embed_dim)
        if flat.shape[0] == 0:
            return
        idx = torch.randint(0, flat.shape[0], (self.codebook_size,), device=flat.device)
        self.embedding.copy_(flat[idx])
        self.cluster_sum.copy_(self.embedding)
        self.cluster_size.fill_(1.0)
        self._inited.fill_(True)

    def _quantize(self, z_e: Tensor) -> tuple[Tensor, Tensor]:
        flat = z_e.reshape(-1, self.embed_dim)
        dist = (
            flat.pow(2).sum(-1, keepdim=True)
            - 2.0 * flat @ self.embedding.t()
            + self.embedding.pow(2).sum(-1)
        )
        code = dist.argmin(dim=-1)
        z_q = F.embedding(code, self.embedding).reshape_as(z_e)
        return z_q, code.reshape(z_e.shape[:-1])

    def forward(self, z_e: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if self.training and not bool(self._inited.item()):
            self._kmeans_init(z_e.detach())
        z_q, code = self._quantize(z_e)
        commitment = F.mse_loss(z_e, z_q.detach())
        z_q_st = z_e + (z_q - z_e).detach()  # straight-through
        if self.training:
            with torch.no_grad():
                flat = z_e.detach().reshape(-1, self.embed_dim)
                onehot = F.one_hot(code.reshape(-1), self.codebook_size).to(flat.dtype)
                self.cluster_size.mul_(self.decay).add_(
                    onehot.sum(0), alpha=1.0 - self.decay
                )
                self.cluster_sum.mul_(self.decay).add_(
                    onehot.t() @ flat, alpha=1.0 - self.decay
                )
                n = self.cluster_size.sum()
                smoothed = (
                    (self.cluster_size + self.eps)
                    / (n + self.codebook_size * self.eps)
                    * n
                )
                self.embedding.copy_(self.cluster_sum / smoothed.unsqueeze(-1))
        return z_q_st, code, commitment

    @torch.no_grad()
    def revive_dead_codes(self, z_e: Tensor, *, threshold: float = 1e-3) -> int:
        flat = z_e.detach().reshape(-1, self.embed_dim)
        dead = self.cluster_size < threshold
        n_dead = int(dead.sum().item())
        if n_dead and flat.shape[0]:
            idx = torch.randint(0, flat.shape[0], (n_dead,), device=flat.device)
            self.embedding[dead] = flat[idx]
            self.cluster_sum[dead] = self.embedding[dead]
            self.cluster_size[dead] = 1.0
        return n_dead


class GumbelQuantizer(nn.Module):
    """Categorical bottleneck via Gumbel-Softmax (annealed tau)."""

    def __init__(self, input_dim: int, codebook_size: int, embed_dim: int) -> None:
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.logit_head = nn.Linear(input_dim, self.codebook_size)
        self.codebook = nn.Embedding(self.codebook_size, embed_dim)
        nn.init.normal_(self.codebook.weight, std=0.02)

    def forward(
        self, z_e: Tensor, *, tau: float, hard: bool, deterministic: bool
    ) -> tuple[Tensor, Tensor, Tensor]:
        logits = self.logit_head(z_e)
        if deterministic:
            y = F.one_hot(logits.argmax(-1), self.codebook_size).to(logits.dtype)
        else:
            y = F.gumbel_softmax(logits, tau=max(float(tau), 1e-3), hard=hard, dim=-1)
        z_q = y @ self.codebook.weight
        log_q = F.log_softmax(logits, dim=-1)
        kl = (log_q.exp() * log_q).sum(-1).mean() + math.log(self.codebook_size)
        return z_q, logits.argmax(-1), kl


def _anneal_tau(step: int | None, start: float, end: float, iters: int) -> float:
    """Linearly anneal Gumbel temperature from ``start`` to ``end`` over ``iters``."""
    if iters <= 0 or step is None:
        return end
    progress = min(1.0, float(step) / iters)
    return start + progress * (end - start)


class GaussianSkillEncoder(HighLevelSkillEncoder):
    def __init__(
        self, *, logstd_min: float = -5.0, logstd_max: float = 2.0, **base
    ) -> None:
        super().__init__(raw_dim=2 * base["z_dim"], **base)
        self.logstd_min, self.logstd_max = float(logstd_min), float(logstd_max)

    def _latent(self, raw, *, deterministic, **_):
        mean, logstd = raw.chunk(2, dim=-1)
        logstd = logstd.clamp(self.logstd_min, self.logstd_max)
        std = logstd.exp()
        z = mean if deterministic else mean + std * torch.randn_like(std)
        kl = 0.5 * (mean.pow(2) + std.pow(2) - 1.0 - 2.0 * logstd).sum(-1).mean()
        return z, kl, {}

    @torch.no_grad()
    def diversity_metrics(self, state, future_window):
        mean, logstd = self._raw(state, future_window).chunk(2, dim=-1)
        logstd = logstd.clamp(self.logstd_min, self.logstd_max)
        kl_per_dim = 0.5 * (mean.pow(2) + (2 * logstd).exp() - 1.0 - 2 * logstd).mean(0)
        metrics = self._z_diversity(mean)
        metrics["kl_per_dim_mean"] = kl_per_dim.mean()
        # Active units: latent dims that carry information (posterior collapse -> 0).
        metrics["active_units"] = (kl_per_dim > 1e-2).float().sum()
        metrics["post_std_mean"] = logstd.exp().mean()
        return metrics


class MultiCategoricalSkillEncoder(_DiscreteSkillEncoder):
    report_flat_code_metrics = False

    def __init__(self, *, groups: int = 8, categories: int = 32, **base) -> None:
        groups, categories = int(groups), int(categories)
        if base["z_dim"] % groups != 0:
            msg = (
                "categorical latent requires z_dim divisible by groups (code dim = "
                f"z_dim // groups): z_dim={base['z_dim']}, groups={groups}."
            )
            raise ValueError(msg)
        super().__init__(raw_dim=groups * categories, **base)
        self.groups, self.categories = groups, categories
        self.num_codes = categories
        self.code_to_latent = nn.Identity()
        code_dim = base["z_dim"] // groups
        # Per-group codebook [G, C, code_dim]; ~unit-norm codes so z scale ~ one-hot.
        self.codebook = nn.Parameter(
            torch.randn(groups, categories, code_dim) / math.sqrt(code_dim)
        )

    def _pre_quantize(self, raw):
        return raw.reshape(*raw.shape[:-1], self.groups, self.categories)  # logits

    def _quantize(self, logits, *, deterministic, **_):
        dist = torch.distributions.Independent(
            torch.distributions.OneHotCategoricalStraightThrough(logits=logits), 1
        )
        if deterministic:
            idx = logits.argmax(dim=-1)
            group_index = torch.arange(self.groups, device=idx.device)
            z_q = self.codebook[group_index, idx]
        else:
            z_q = torch.einsum("...gc,gcb->...gb", dist.rsample(), self.codebook)
        probs = dist.base_dist.probs
        log_probs = torch.log_softmax(logits, dim=-1)
        kl = (probs * (log_probs + math.log(self.categories))).sum(-1).sum(-1).mean()
        return z_q, logits.argmax(dim=-1), kl, {}

    @torch.no_grad()
    def diversity_metrics(self, state, future_window):
        logits = self._pre_quantize(self._raw(state, future_window))
        z_q, codes, _, _ = self._quantize(
            logits,
            deterministic=True,
            step=None,
        )
        z = self.code_to_latent(z_q).reshape(*logits.shape[:-2], self.z_dim)

        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        conditional_entropy = -(probs * log_probs).sum(dim=-1).mean()

        marginal_probs = probs.mean(dim=0)
        marginal_entropy = (
            -(marginal_probs * marginal_probs.clamp_min(1.0e-12).log())
            .sum(dim=-1)
            .mean()
        )

        group_code_usage = (
            F.one_hot(
                codes.reshape(-1, self.groups),
                num_classes=self.categories,
            )
            .any(dim=0)
            .to(dtype=z.dtype)
            .mean(dim=-1)
        )

        return {
            **self._z_diversity(z),
            "marginal_entropy": marginal_entropy,
            "conditional_entropy": conditional_entropy,
            "group_code_usage_frac_mean": group_code_usage.mean(),
            "group_code_usage_frac_min": group_code_usage.min(),
            "group_code_usage_frac_max": group_code_usage.max(),
        }


class GumbelMultiCategoricalSkillEncoder(MultiCategoricalSkillEncoder):
    """Product codebook trained with per-group Gumbel-softmax (annealed temperature).

    Same capacity and inference as ``MultiCategoricalSkillEncoder`` (G groups x C
    categories, per-group codebook, deterministic = per-group argmax), but the
    training sampler is Gumbel-softmax instead of the hard straight-through one-hot:
    the soft, temperature-annealed relaxation keeps gradients flowing to all logits
    and codes, avoiding the premature commitment / collapse of the ST sampler.
    """

    def __init__(
        self,
        *,
        groups: int = 8,
        categories: int = 32,
        tau_start: float = 2.0,
        tau_end: float = 0.5,
        tau_anneal_iters: int = 2000,
        hard: bool = True,
        **base,
    ) -> None:
        super().__init__(groups=groups, categories=categories, **base)
        self.tau_start, self.tau_end = float(tau_start), float(tau_end)
        self.tau_anneal_iters, self.hard = int(tau_anneal_iters), bool(hard)

    def _quantize(self, logits, *, deterministic, step):
        log_probs = torch.log_softmax(logits, dim=-1)
        kl = (
            (log_probs.exp() * (log_probs + math.log(self.categories)))
            .sum(-1)
            .sum(-1)
            .mean()
        )
        if deterministic:
            idx = logits.argmax(dim=-1)
            group_index = torch.arange(self.groups, device=idx.device)
            return self.codebook[group_index, idx], idx, kl, {}
        tau = _anneal_tau(step, self.tau_start, self.tau_end, self.tau_anneal_iters)
        weights = F.gumbel_softmax(logits, tau=max(tau, 1e-3), hard=self.hard, dim=-1)
        z_q = torch.einsum("...gc,gcb->...gb", weights, self.codebook)
        return z_q, logits.argmax(dim=-1), kl, {"tau": logits.new_tensor(tau)}


class GumbelSkillEncoder(_DiscreteSkillEncoder):
    def __init__(
        self,
        *,
        codebook_size: int = 512,
        tau_start: float = 2.0,
        tau_end: float = 0.5,
        tau_anneal_iters: int = 2000,
        hard: bool = True,
        **base,
    ) -> None:
        super().__init__(raw_dim=base["z_dim"], **base)
        self.num_codes = int(codebook_size)
        self.code_to_latent = nn.Identity()
        self.gumbel = GumbelQuantizer(base["z_dim"], codebook_size, base["z_dim"])
        self.tau_start, self.tau_end = float(tau_start), float(tau_end)
        self.tau_anneal_iters, self.hard = int(tau_anneal_iters), bool(hard)

    def _quantize(self, z_e, *, deterministic, step):
        tau = _anneal_tau(step, self.tau_start, self.tau_end, self.tau_anneal_iters)
        z_q, code, kl = self.gumbel(
            z_e, tau=tau, hard=self.hard, deterministic=deterministic
        )
        return z_q, code, kl, {"tau": torch.tensor(tau, device=z_e.device)}


class VQSkillEncoder(_DiscreteSkillEncoder):
    def __init__(
        self,
        *,
        codebook_size: int = 512,
        ema_decay: float = 0.99,
        dead_code_reset_iters: int = 0,
        **base,
    ) -> None:
        super().__init__(raw_dim=base["z_dim"], **base)
        self.num_codes = int(codebook_size)
        self.code_to_latent = nn.Identity()
        self.vq = EMAVQQuantizer(codebook_size, base["z_dim"], decay=ema_decay)
        self.dead_code_reset_iters = int(dead_code_reset_iters)
        self._last_z_e: Tensor | None = None

    def _quantize(self, z_e, **_):
        self._last_z_e = z_e.detach()  # nearest-neighbor; deterministic/step unused
        z_q, code, commitment = self.vq(z_e)
        return z_q, code, commitment, {}

    def on_after_train_step(self, step: int) -> None:
        if (
            self.dead_code_reset_iters > 0
            and step > 0
            and step % self.dead_code_reset_iters == 0
            and self._last_z_e is not None
        ):
            self.vq.revive_dead_codes(self._last_z_e)

    @torch.no_grad()
    def diversity_metrics(self, state, future_window):
        metrics = super().diversity_metrics(state, future_window)
        # EMA cluster usage gives a global (not per-batch) dead-code estimate.
        metrics["dead_code_frac"] = (self.vq.cluster_size < 1e-3).float().mean()
        return metrics


def build_skill_encoder(
    *,
    state_dim: int,
    window_steps: int,
    z_dim: int,
    hidden_dims: tuple[int, ...],
    spec: SkillLatentSpec,
    activation: str = "mish",
    layer_norm: bool = True,
) -> HighLevelSkillEncoder:
    """Build the skill encoder selected by ``spec.latent_mode``.

    ``state_dim``, ``window_steps``, ``z_dim``, ``hidden_dims``, ``activation``
    and ``layer_norm`` configure the shared trunk; the mode-specific fields of
    ``spec`` configure the latent head.
    """
    base = {
        "state_dim": state_dim,
        "window_steps": window_steps,
        "z_dim": z_dim,
        "hidden_dims": hidden_dims,
        "activation": activation,
        "layer_norm": layer_norm,
    }
    mode = spec.latent_mode
    if mode == "deterministic":
        return DeterministicSkillEncoder(**base)
    if mode == "gaussian":
        return GaussianSkillEncoder(
            logstd_min=spec.gaussian_logstd_min,
            logstd_max=spec.gaussian_logstd_max,
            **base,
        )
    if mode == "categorical":
        return MultiCategoricalSkillEncoder(
            groups=spec.categorical_groups,
            categories=spec.categorical_categories,
            **base,
        )
    if mode == "gumbel_multicat":
        return GumbelMultiCategoricalSkillEncoder(
            groups=spec.categorical_groups,
            categories=spec.categorical_categories,
            tau_start=spec.gumbel_tau_start,
            tau_end=spec.gumbel_tau_end,
            tau_anneal_iters=spec.gumbel_tau_anneal_iters,
            hard=spec.gumbel_hard,
            **base,
        )
    if mode == "gumbel":
        return GumbelSkillEncoder(
            codebook_size=spec.gumbel_codebook_size,
            tau_start=spec.gumbel_tau_start,
            tau_end=spec.gumbel_tau_end,
            tau_anneal_iters=spec.gumbel_tau_anneal_iters,
            hard=spec.gumbel_hard,
            **base,
        )
    if mode == "fsq":
        return FSQSkillEncoder(levels=tuple(spec.fsq_levels), **base)
    if mode == "sonic_fsq":
        return SONICFSQSkillEncoder(levels=tuple(spec.sonic_fsq_levels), **base)
    if mode == "vq":
        return VQSkillEncoder(
            codebook_size=spec.vq_codebook_size,
            ema_decay=spec.vq_ema_decay,
            dead_code_reset_iters=spec.vq_dead_code_reset_iters,
            **base,
        )
    msg = f"unknown latent_mode {mode!r}; expected one of {LATENT_MODES}."
    raise ValueError(msg)
