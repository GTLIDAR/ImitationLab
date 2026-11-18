from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def linear_beta_schedule(beta_min: float = 1e-4, beta_max: float = 0.02, T: int = 1000):
    return np.linspace(beta_min, beta_max, T)


def cosine_beta_schedule(s: float = 0.008, T: int = 1000):
    f = np.cos((np.arange(T + 1) / T + s) / (1 + s) * np.pi / 2.0) ** 2
    alpha_bar = f / f[0]
    beta = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return beta.clip(None, 0.999)


def vp_beta_schedule(T: int = 1000):
    t = np.arange(1, T + 1)
    b_max = 10.0
    b_min = 0.1
    alpha = np.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T**2)
    betas = 1 - alpha
    return betas


def get_noise_schedule(
    noise_schedule: str = "linear",
    num_noises: int = 1000,
    beta_min: float = 1e-4,
    beta_max: float = 0.02,
    s: float = 0.008,
):
    if noise_schedule == "linear":
        betas = linear_beta_schedule(beta_min, beta_max, num_noises)
    elif noise_schedule == "cosine":
        betas = cosine_beta_schedule(s, num_noises)
    elif noise_schedule == "vp":
        betas = vp_beta_schedule(num_noises)
    else:
        raise NotImplementedError(f"Unknown noise schedule: {noise_schedule}")
    alphas = 1 - betas
    alphabars = np.cumprod(alphas, axis=0)
    return (
        torch.as_tensor(betas, dtype=torch.float32),
        torch.as_tensor(alphas, dtype=torch.float32),
        torch.as_tensor(alphabars, dtype=torch.float32),
    )


class PositionalFeature(nn.Module):
    """Sinusoidal positional embedding for diffusion time steps."""

    def __init__(self, dim: int, max_positions: int = 10000, endpoint: bool = False):
        super().__init__()
        self.dim = dim
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, 1).float()
        freqs = torch.arange(0, self.dim // 2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.dim // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x @ freqs.unsqueeze(0)
        x = torch.cat([x.cos(), x.sin()], dim=-1)
        return x


class ResidualMLPBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Pre-normalization operates on the block input. Using ``hidden_dim``
        # only works when adjacent widths happen to match and makes tapered
        # ResidualMLPs fail before their projection layer.
        self.ln = nn.LayerNorm(input_dim)
        self.activation = activation
        if input_dim != hidden_dim:
            self.residual = nn.Linear(input_dim, hidden_dim)
        else:
            self.residual = nn.Identity()
        if dropout and dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

    def forward(self, x):
        residual = x
        x = self.dropout(x)
        x = self.ln(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x + self.residual(residual)


class ResidualMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 0,
        hidden_dims: Sequence[int] = (),
        activation: Callable = F.relu,
        dropout: float = None,
        device: str = "cpu",
    ):
        super().__init__()
        assert len(hidden_dims) > 0, (
            "hidden_dims must be a list of at least one integer"
        )
        self.activation = activation
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[-1], output_dim)
        hidden_dims = [hidden_dims[0]] + list(hidden_dims)
        self.blocks = nn.Sequential(
            *[
                ResidualMLPBlock(
                    hidden_dims[i],
                    hidden_dims[i + 1],
                    dropout,
                    activation,
                )
                for i in range(len(hidden_dims) - 1)
            ]
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.blocks(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
