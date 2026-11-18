"""Bilinear diffusion model shared by the skill encoder and world model."""

from __future__ import annotations

import copy
import math
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .networks import (
    PositionalFeature,
    ResidualMLP,
    get_noise_schedule,
)

# ---------------------------------------------------------------------------
# Empirical normalization
# ---------------------------------------------------------------------------


class EmpiricalNormalization(nn.Module):
    """Running mean/std normalization using Welford's online algorithm."""

    def __init__(self, shape: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def update(self, x: Tensor) -> None:
        """Recompute mean/var from the provided data (e.g. all effective buffer samples)."""
        x = x.reshape(-1, x.shape[-1])
        self.mean = x.mean(dim=0)
        self.var = x.var(dim=0, correction=0)
        self.count = torch.tensor(x.shape[0], dtype=torch.long, device=x.device)

    def normalize(self, x: Tensor) -> Tensor:
        if self.count < 2:
            return x
        return (x - self.mean) / (self.var.sqrt() + self.eps)

    def inv_normalize(self, x: Tensor) -> Tensor:
        return x * (self.var.sqrt() + self.eps) + self.mean


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BilinearSR(ABC, nn.Module):
    """Abstract base for bilinear spectral representation modules.

    phi(s, a) = g(a)^T F(s), with g(a) in R^{embed_dim} and
    F(s) in R^{embed_dim x feature_dim}. Subclasses implement mu and
    compute_loss for different SR objectives.

    ``phi_parameterization`` selects how phi mixes state and action:

    - ``concat``: phi = phi_net([F(s); g(a)]), the default. No bilinear
      algebra survives the MLP.
    - ``bilinear``: the factorization above with a nonlinear g.
    - ``affine``: the factorization above with g(a) = A a + b a single
      linear layer, which makes phi affine in the action. Combinations of
      actions then combine their phi exactly:
      phi(s, sum_i w_i a_i) = sum_i w_i phi(s, a_i) when sum_i w_i = 1.
      For a skill latent z this makes the learned score field affine in z,
      without guaranteeing exact clean-distribution product sampling.
    - ``affine_poe``: affine reassociated as [1; a]^T E, retaining all
      internal widths and parameters; E has action_dim + 1 fields.
    - ``affine_no_bias``: the same matrix factorization with g(a) = A a.
      The complete field is exactly linear in a: <a, A^T F(s) mu>.
    - ``identity``: phi(s, a) = a. The action IS the representation, so the
      score ``phi^T mu`` is ``<a, mu(.)>``: exactly linear in the action, no
      bias. Requires ``action_dim == feature_dim``. Paired with
      ``mu_conditioning="pair"`` on the DiffSR subclass this is the
      product-of-experts form ``<z, E(s, s')>``: each coordinate of z is the
      coefficient of one expert field ``E_k(s, s', t)``. Arbitrary linear
      combinations of z combine the fields exactly; this does not establish
      exact clean-distribution product sampling.
    - ``linear``: phi(s, a) = A a, a single bias-free linear map from the
      action to ``feature_dim``, which may be WIDER than the action. The score
      ``<A a, E(s, s')>`` stays exactly linear in a, so composition is
      unchanged, but the expert bank is no longer tied to the latent width:
      ``feature_dim`` experts are mixed by ``feature_dim`` learned rows of a.
    - ``linear_bias``: phi(s, a) = [1; A a], the same projection with the
      constant channel of ``identity_bias`` prepended. ``A`` maps to
      ``feature_dim - 1``; the first expert row is the command-independent
      base denoiser. ``A`` stays bias-free, because an affine ``A a + b``
      would put a second command-independent term in the same span.
    - ``identity_bias``: phi(s, a) = [1; a]. Requires
      ``feature_dim == action_dim + 1``. With pair-conditioned mu the first
      field is a source/target/time-dependent base denoiser independent of
      the command. Noise predictions are affine in a; this alone does not
      guarantee exact clean-distribution product sampling.
    """

    def __init__(
        self,
        obs_dim: int,
        next_obs_dim: int,
        action_dim: int,
        feature_dim: int,
        embed_dim: int,
        g_hidden_dims: tuple[int, ...],
        f_hidden_dims: tuple[int, ...] | None = None,
        phi_parameterization: str = "concat",
        use_ema_for_policy: bool = True,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.obs_dim = obs_dim
        self.next_obs_dim = next_obs_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.phi_parameterization = str(phi_parameterization).strip().lower()
        aliases = {"legacy": "bilinear", "f_matrix": "bilinear"}
        self.phi_parameterization = aliases.get(
            self.phi_parameterization, self.phi_parameterization
        )
        if self.phi_parameterization not in {
            "concat",
            "bilinear",
            "affine",
            "affine_no_bias",
            "affine_poe",
            "identity",
            "identity_bias",
            "linear",
            "linear_bias",
        }:
            msg = (
                "phi_parameterization must be one of 'concat', 'bilinear', "
                "'affine', 'affine_no_bias', 'affine_poe', 'identity', 'identity_bias', 'linear', or "
                "'linear_bias' "
                "(aliases: 'legacy', 'f_matrix'), got "
                f"{phi_parameterization!r}."
            )
            raise ValueError(msg)
        if self.phi_parameterization == "identity" and action_dim != feature_dim:
            msg = (
                "phi_parameterization='identity' sets phi(s, a) = a, so "
                f"feature_dim must equal action_dim; got feature_dim={feature_dim}, "
                f"action_dim={action_dim}."
            )
            raise ValueError(msg)
        if self.phi_parameterization == "linear" and feature_dim < 1:
            msg = (
                "phi_parameterization='linear' needs feature_dim >= 1; got "
                f"{feature_dim}."
            )
            raise ValueError(msg)
        if self.phi_parameterization == "linear_bias" and feature_dim < 2:
            msg = (
                "phi_parameterization='linear_bias' prepends a constant channel, "
                f"so feature_dim must be at least 2; got {feature_dim}."
            )
            raise ValueError(msg)
        if (
            self.phi_parameterization == "identity_bias"
            and feature_dim != action_dim + 1
        ):
            msg = (
                "phi_parameterization='identity_bias' sets phi(s, a) = [1; a], so "
                "feature_dim must equal action_dim + 1; "
                f"got feature_dim={feature_dim}, action_dim={action_dim}."
            )
            raise ValueError(msg)
        # Both factorized parameterizations need a matrix-valued F(s); they
        # differ only in g. Derived, never serialized.
        self._matrix_f = self.phi_parameterization in {
            "bilinear",
            "affine",
            "affine_no_bias",
            "affine_poe",
        }

        # --- Shared phi components ---
        if self._matrix_f:
            # Checkpoint-era parameterization: F(s) is matrix-valued and the
            # policy consumes F(s) z. Kept for backwards-compatible evaluation.
            self.state_net = ResidualMLP(
                input_dim=obs_dim,
                output_dim=embed_dim * feature_dim,
                hidden_dims=f_hidden_dims or g_hidden_dims,
                activation=nn.Mish(),
            )
        else:
            self.state_net = ResidualMLP(
                input_dim=obs_dim,
                output_dim=embed_dim,
                hidden_dims=g_hidden_dims,
                activation=nn.Mish(),
            )
        self.action_net: nn.Module
        if self.phi_parameterization in {"identity", "identity_bias"}:
            # phi never mixes state and action; the action passes through.
            self.action_net = nn.Identity()
        elif self.phi_parameterization in {"linear", "linear_bias"}:
            # phi = A a, widened to the expert count. Bias-free: `linear_bias`
            # carries its command-independent term in the constant channel,
            # and `linear` deliberately has none.
            out_dim = (
                feature_dim - 1
                if self.phi_parameterization == "linear_bias"
                else feature_dim
            )
            self.action_net = nn.Linear(action_dim, out_dim, bias=False)
        elif self.phi_parameterization in {"affine", "affine_no_bias", "affine_poe"}:
            # g(a) = A a + b. The bias is the action-independent component of
            # the transition operator; the columns of A are the patterns each
            # action coordinate weights.
            self.action_net = nn.Linear(
                action_dim,
                embed_dim,
                bias=self.phi_parameterization in {"affine", "affine_poe"},
            )
        else:
            self.action_net = ResidualMLP(
                input_dim=action_dim,
                output_dim=embed_dim,
                hidden_dims=g_hidden_dims,
                activation=nn.Mish(),
            )

        self.phi_net: ResidualMLP | None = None
        if self.phi_parameterization == "concat":
            self.phi_net = ResidualMLP(
                input_dim=embed_dim + embed_dim,
                output_dim=feature_dim,
                hidden_dims=g_hidden_dims,
                activation=nn.Mish(),
            )

        # --- EMA target state_net for policy ---
        if use_ema_for_policy:
            self.state_net_ema = copy.deepcopy(self.state_net).to(device)
            for p in self.state_net_ema.parameters():
                p.requires_grad_(False)
        else:
            self.state_net_ema = None

    def _F(self, s: Tensor, use_ema: bool = False) -> Tensor:
        """Return matrix-valued F(s) for legacy bilinear checkpoints."""
        net = (
            self.state_net_ema
            if use_ema and self.state_net_ema is not None
            else self.state_net
        )
        return net(s).reshape(-1, self.embed_dim, self.feature_dim)

    def encode_state(self, s: Tensor) -> Tensor:
        if self._matrix_f:
            return self._F(s)
        return self.state_net(s)

    def encode_action(self, a: Tensor) -> Tensor:
        return self.action_net(a)

    def forward_phi(self, s: Tensor, a: Tensor) -> Tensor:
        """phi(s, a) = (1/sqrt(E)) * g(a)^T F(s) -> (B, feature_dim).

        The 1/sqrt(embed_dim) factor keeps the inner product at unit scale,
        mirroring scaled dot-product attention. Under 'affine', g is a single
        linear layer, so phi is affine in a.
        """
        if self.phi_parameterization == "identity":
            return a
        if self.phi_parameterization in {"identity_bias", "affine_poe"}:
            return torch.cat([torch.ones_like(a[..., :1]), a], dim=-1)
        if self.phi_parameterization == "linear":
            return self.action_net(a)
        if self.phi_parameterization == "linear_bias":
            projected = self.action_net(a)
            ones = torch.ones_like(projected[..., :1])
            return torch.cat([ones, projected], dim=-1)
        if self._matrix_f:
            f_s = self._F(s)
            g_a = self.encode_action(a)
            return torch.einsum("be,bef->bf", g_a, f_s) / math.sqrt(self.embed_dim)
        s = self.state_net(s)
        a = self.action_net(a)
        assert self.phi_net is not None
        return self.phi_net(torch.concat([s, a], dim=-1))

    # -- EMA --

    @torch.no_grad()
    def update_ema(self, tau: float) -> None:
        if self.state_net_ema is None:
            return
        for p_ema, p_online in zip(
            self.state_net_ema.parameters(), self.state_net.parameters(), strict=True
        ):
            p_ema.data.lerp_(p_online.data, tau)

    # -- Policy / Q representation (shared) --

    def compute_policy_representation(
        self,
        s: Tensor,
        z: Tensor = None,
        *,
        include_raw_state: bool = True,
    ) -> Tensor:
        if self._matrix_f:
            f_s = self._F(s, use_ema=True).detach()
            component1 = torch.einsum("bef,bf->be", f_s, z)
        else:
            net = (
                self.state_net_ema if self.state_net_ema is not None else self.state_net
            )
            state_embed = net(s).detach()
            component1 = torch.concat([state_embed, z], dim=-1)
        if include_raw_state:
            return torch.concat([component1, s.detach()], dim=-1)
        return component1

    def compute_q_representation(self, s: Tensor, a: Tensor) -> Tensor:
        """Return phi(s, a) for Q-function use."""
        return self.forward_phi(s, a)

    # -- Abstract interface --

    @abstractmethod
    def compute_loss(
        self,
        s: Tensor,
        a: Tensor,
        sp: Tensor,
        r: Tensor,
    ) -> tuple[dict[str, float], Tensor, Tensor]:
        """Compute the spectral representation learning loss.

        Returns:
            metrics: Dict of scalar metrics for logging.
            main_loss: Primary SR loss (diffusion or spectral contrastive).
            aux_loss: Auxiliary loss (e.g. reward prediction), or zero tensor.
        """
        ...

    def sample(
        self,
        s: Tensor,
        a: Tensor,
        preserve_history: bool = False,
    ) -> tuple[Tensor, dict]:
        """Generate s' given (s, a). Only supported by diffusion-based SR."""
        msg = f"{type(self).__name__} does not support generative sampling."
        raise NotImplementedError(msg)

    @property
    def supports_sampling(self) -> bool:
        return False

    def update_obs_norm(self, _next_obs: Tensor) -> None:
        """Update observation normalization statistics. No-op by default."""
        del _next_obs


# ---------------------------------------------------------------------------
# DiffSR implementation
# ---------------------------------------------------------------------------


class DiffSRBilinear(BilinearSR):
    r"""Bilinear spectral representation with diffusion-based learning.

    mu(s', t) outputs a Jacobian (feature_dim x next_obs_dim) so that
    phi(s,a)^T @ mu(s',t) predicts the noise added to s' in the forward
    diffusion process.

    ``mu_conditioning`` selects what mu sees:

    - ``next``: ``mu(s', t)``, the default. mu is a function of the noisy
      target and the diffusion time only, so one mu per target serves every
      (s, a) in the batch.
    - ``pair``: ``mu(s, s', t)``, the transition pair. Each row of mu is then
      a joint expert field ``E_k(s, s', t)`` and the noise prediction is
      ``sum_k phi_k(s, a) E_k(s, s', t)``. With ``phi_parameterization=
      "identity"`` that is ``<a, E(s, s', t)>``. mu is no longer shareable
      across different sources, which the positive-pair denoising loss here
      never needs.
    """

    def __init__(
        self,
        obs_dim: int,
        next_obs_dim: int,
        action_dim: int,
        feature_dim: int,
        embed_dim: int,
        g_hidden_dims: tuple[int, ...],
        mu_hidden_dims: tuple[int, ...],
        f_hidden_dims: tuple[int, ...] | None = None,
        phi_parameterization: str = "concat",
        num_noises: int = 16,
        use_ema_for_policy: bool = True,
        x_min: float = -10.0,
        x_max: float = 10.0,
        device: str | torch.device = "cpu",
        mu_conditioning: str = "next",
    ) -> None:
        super().__init__(
            obs_dim=obs_dim,
            next_obs_dim=next_obs_dim,
            action_dim=action_dim,
            feature_dim=feature_dim,
            embed_dim=embed_dim,
            g_hidden_dims=g_hidden_dims,
            f_hidden_dims=f_hidden_dims,
            phi_parameterization=phi_parameterization,
            use_ema_for_policy=use_ema_for_policy,
            device=device,
        )
        self.mu_conditioning = str(mu_conditioning).strip().lower()
        if self.mu_conditioning not in {"next", "pair", "film"}:
            msg = (
                "mu_conditioning must be 'next', 'pair', or 'film' "
                f"(source-modulated noisy-target features), got {mu_conditioning!r}."
            )
            raise ValueError(msg)
        self.num_noises = num_noises
        self.x_min, self.x_max = x_min, x_max

        # Observation normalization for diffusion
        self.obs_norm = EmpiricalNormalization(next_obs_dim)

        # VP noise schedule
        betas, alphas, alphabars = get_noise_schedule("vp", num_noises)
        alphabars_prev = F.pad(alphabars[:-1], (1, 0), value=1.0)
        self.register_buffer("betas", betas[..., None])
        self.register_buffer("alphas", alphas[..., None])
        self.register_buffer("alphabars", alphabars[..., None])
        self.register_buffer("alphabars_prev", alphabars_prev[..., None])

        # Time embedding
        self.mlp_t = nn.Sequential(
            PositionalFeature(128),
            nn.Linear(128, 256),
            nn.Mish(),
            nn.Linear(256, 128),
        )

        # mu(s', t) -> Jacobian (feature_dim x next_obs_dim); under 'pair' the
        # source state is part of mu's input as well.
        mu_source_dim = obs_dim if self.mu_conditioning == "pair" else 0
        self.mu_net = ResidualMLP(
            input_dim=next_obs_dim + 128 + mu_source_dim,
            output_dim=feature_dim * next_obs_dim,
            hidden_dims=mu_hidden_dims,
            activation=nn.Mish(),
        )
        self.mu_modulation: nn.Module | None = None
        if self.mu_conditioning == "film":
            self.mu_modulation = ResidualMLP(
                input_dim=obs_dim,
                output_dim=2 * mu_hidden_dims[-1],
                hidden_dims=g_hidden_dims,
                activation=nn.Mish(),
            )
            # Begin near the target-only network while allowing source gradients
            # from the first step. Only s, never z, enters the modulation branch.
            nn.init.normal_(self.mu_modulation.fc2.weight, std=1e-3)
            nn.init.zeros_(self.mu_modulation.fc2.bias)

    def update_obs_norm(self, next_obs: Tensor) -> None:
        self.obs_norm.update(next_obs)

    def add_noise(self, x0: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        noise_idx = torch.randint(
            0, self.num_noises, (x0.shape[0],), device=x0.device
        ).long()
        alphabars = self.alphabars[noise_idx]
        eps = torch.randn_like(x0)
        xt = alphabars.sqrt() * x0 + (1 - alphabars).sqrt() * eps
        return xt, noise_idx, eps

    def forward_mu(self, sp: Tensor, t: Tensor, s: Tensor | None = None) -> Tensor:
        """mu(s', t) -> Jacobian (B, feature_dim, next_obs_dim).

        Under ``mu_conditioning="pair"`` the source state ``s`` is required and
        mu becomes mu(s, s', t)."""
        t_ff = self.mlp_t(t)
        if self.mu_conditioning == "pair":
            if s is None:
                msg = "mu_conditioning='pair' needs the source state s in forward_mu."
                raise ValueError(msg)
            x = torch.concat([s, sp, t_ff], dim=-1)
        else:
            x = torch.concat([sp, t_ff], dim=-1)
        if self.mu_modulation is not None:
            if s is None:
                msg = "mu_conditioning='film' needs the source state s in forward_mu."
                raise ValueError(msg)
            gamma, beta = self.mu_modulation(s).chunk(2, dim=-1)
            hidden = self.mu_net.activation(self.mu_net.blocks(self.mu_net.fc1(x)))
            x = self.mu_net.fc2((1 + gamma) * hidden + beta)
        else:
            x = self.mu_net(x)
        fields = x.reshape(-1, self.feature_dim, self.next_obs_dim)
        if self.phi_parameterization == "affine_poe":
            if s is None:
                raise ValueError("affine_poe requires source s in forward_mu.")
            # Same A, b, F and mu as affine, reassociated into [1; a]^T E.
            # Contract A with F first to avoid a (B, embed_dim, target_dim)
            # intermediate. feature_dim remains the internal factor width.
            assert isinstance(self.action_net, nn.Linear)
            assert self.action_net.bias is not None
            weights = torch.cat(
                [self.action_net.bias[:, None], self.action_net.weight], dim=1
            )
            source_fields = torch.einsum("ek,bef->bkf", weights, self._F(s))
            fields = torch.bmm(source_fields, fields) / math.sqrt(self.embed_dim)
        return fields

    def forward_eps(
        self,
        s: Tensor | None = None,
        a: Tensor | None = None,
        sp: Tensor | None = None,
        t: Tensor | None = None,
        z_phi: Tensor | None = None,
        z_mu: Tensor | None = None,
    ) -> Tensor:
        """Predict noise: eps_pred = phi(s,a)^T @ mu(s',t) -> (B, next_obs_dim)."""
        if z_phi is None:
            z_phi = self.forward_phi(s, a)
        if z_mu is None:
            z_mu = self.forward_mu(sp, t, s=s)
        return torch.bmm(z_phi.unsqueeze(1), z_mu).squeeze(1)

    def compute_loss(
        self,
        s: Tensor,
        a: Tensor,
        sp: Tensor,
        _r: Tensor,
    ) -> tuple[dict[str, float], Tensor, Tensor]:
        x0 = self.obs_norm.normalize(sp)
        xt, t, eps = self.add_noise(x0)

        z_phi = self.forward_phi(s=s, a=a)
        z_mu = self.forward_mu(sp=xt.detach(), t=t.unsqueeze(-1), s=s)
        eps_pred = self.forward_eps(z_phi=z_phi, z_mu=z_mu)
        diffusion_loss = (eps_pred - eps).pow(2).sum(-1).mean()

        metrics = {
            "info/eps_l1_norm": eps.abs().mean().item(),
            "info/eps_pred_l1_norm": eps_pred.abs().mean().item(),
            "info/x0_mean": x0.mean().item(),
            "info/x0_std": x0.std(0).mean().item(),
            "info/x0_l1_norm": x0.abs().mean().item(),
            "info/obs_norm_mean": self.obs_norm.mean.abs().mean().item(),
            "info/obs_norm_std": self.obs_norm.var.sqrt().mean().item(),
            "info/obs_norm_count": float(self.obs_norm.count.item()),
            "loss/dynamics_loss": diffusion_loss.item(),
        }
        return metrics, diffusion_loss, torch.tensor(0.0, device=s.device)

    @property
    def supports_sampling(self) -> bool:
        return True

    @torch.no_grad()
    def sample(
        self,
        s: Tensor,
        a: Tensor,
        preserve_history: bool = False,
    ) -> tuple[Tensor, dict]:
        info = {}
        shape = (s.shape[0], self.next_obs_dim)
        xt = torch.randn(shape, device=s.device)
        z_phi = self.forward_phi(s, a)

        if preserve_history:
            info["sample_history"] = [xt.clone()]

        for t in reversed(range(self.num_noises)):
            z = torch.randn_like(xt)
            timestep = torch.full((xt.shape[0],), t, dtype=torch.int64, device=s.device)
            z_mu = self.forward_mu(sp=xt, t=timestep.unsqueeze(-1), s=s)
            eps_pred = torch.bmm(z_phi.unsqueeze(1), z_mu).squeeze(1)

            sigma_t = 0
            if t > 0:
                sigma_t_sq = (
                    self.betas[timestep]
                    * (1 - self.alphabars_prev[timestep])
                    / (1 - self.alphabars[timestep])
                )
                sigma_t = sigma_t_sq.clip(1e-20).sqrt()

            xt = (
                1.0
                / self.alphas[timestep].sqrt()
                * (
                    xt
                    - self.betas[timestep]
                    / (1 - self.alphabars[timestep]).sqrt()
                    * eps_pred
                )
                + sigma_t * z
            )
            xt = xt.clip(self.x_min, self.x_max)

            if preserve_history:
                info["sample_history"].append(xt.clone())

        xt = self.obs_norm.inv_normalize(xt)
        return xt, info
