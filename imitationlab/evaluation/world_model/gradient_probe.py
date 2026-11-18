"""Statistics of stochastic gradient estimates and of rollout Jacobians.

Pure tensor helpers behind ``scripts/tracking/probe_actor_gradient.py``: the
signal-to-noise ratio and pairwise agreement of K independent estimates of one
gradient, and per-sample spectral norms of a batched map's Jacobian.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor


def gradient_stats(
    grads: Tensor, reference: Tensor | None = None, *, max_pairs: int = 4096
) -> dict[str, float]:
    """Summary of K estimates of one gradient, ``grads`` (K, P).

    * ``snr``: ||mean g|| / sqrt(tr Cov g) (the sample covariance's trace is
      the mean squared distance of the estimates to their mean, K/(K-1)-corrected);
    * ``cos_pairwise_median`` / ``cos_pairwise_mean``: cosine between pairs of
      independent estimates (all pairs, or ``max_pairs`` random ones);
    * ``cos_to_mean_median``: cosine of each estimate to the mean of the others;
    * ``cos_mean_to_reference``: cosine of the mean estimate to ``reference``
      (another estimator's mean), when given;
    * ``mean_norm``, ``rms_norm``: norm of the mean and root-mean-square norm
      of the estimates.
    """
    g = grads.double()
    K = g.shape[0]
    if K < 2:
        msg = "gradient_stats needs at least two estimates"
        raise ValueError(msg)
    mean = g.mean(0)
    dev = g - mean
    tr_cov = dev.square().sum() / (K - 1)
    out = {
        "n": float(K),
        "mean_norm": float(mean.norm()),
        "rms_norm": float(g.norm(dim=1).square().mean().sqrt()),
        "snr": float(mean.norm() / tr_cov.sqrt().clamp_min(1e-30)),
    }
    unit = g / g.norm(dim=1, keepdim=True).clamp_min(1e-30)
    n_pairs = K * (K - 1) // 2
    if n_pairs <= max_pairs:
        cos = unit @ unit.T
        iu = torch.triu_indices(K, K, offset=1, device=g.device)
        pair_cos = cos[iu[0], iu[1]]
    else:
        gen = torch.Generator(device="cpu").manual_seed(0)
        i = torch.randint(0, K, (max_pairs,), generator=gen)
        j = torch.randint(0, K - 1, (max_pairs,), generator=gen)
        j = j + (j >= i).long()  # j != i
        pair_cos = (unit[i.to(g.device)] * unit[j.to(g.device)]).sum(1)
    out["cos_pairwise_median"] = float(pair_cos.median())
    out["cos_pairwise_mean"] = float(pair_cos.mean())
    # Leave-one-out: each estimate against the mean of the other K - 1.
    loo = (mean * K - g) / (K - 1)
    loo_cos = (unit * (loo / loo.norm(dim=1, keepdim=True).clamp_min(1e-30))).sum(1)
    out["cos_to_mean_median"] = float(loo_cos.median())
    if reference is not None:
        r = reference.double()
        out["cos_mean_to_reference"] = float(
            (mean @ r) / (mean.norm() * r.norm()).clamp_min(1e-30)
        )
    return out


def batched_jacobian(fn: Callable[[Tensor], Tensor], x: Tensor) -> Tensor:
    """Per-sample Jacobian (B, M, N) of ``fn``: (B, N) -> (B, M), where sample
    b's output depends on sample b's input only (one reverse pass per output
    coordinate)."""
    x = x.detach().requires_grad_(True)
    y = fn(x)
    B, M = y.shape
    rows = []
    for i in range(M):
        (g,) = torch.autograd.grad(y[:, i].sum(), x, retain_graph=i + 1 < M)
        rows.append(g)
    return torch.stack(rows, 1)


def spectral_norms(jac: Tensor) -> Tensor:
    """Largest singular value of each (M, N) block of ``jac`` (B, M, N)."""
    return torch.linalg.matrix_norm(jac.double(), ord=2).float()


def linearized_noise_return(
    slopes: Tensor, eta: Tensor, mask: Tensor, decay: float
) -> Tensor:
    """L_t = sum_{k >= t} decay^(k - t) q_k . eta_k over a window, (B, H).

    ``slopes`` q and ``eta`` are (B, H, A); ``mask`` (B, H) drops steps past the
    window's episode. It is the first-order change in the return from step t on
    that the recorded action noise caused, by the model's own slopes.
    """
    effect = (slopes * eta).sum(-1) * mask
    out = torch.zeros_like(effect)
    run = torch.zeros_like(effect[:, 0])
    for t in range(effect.shape[1] - 1, -1, -1):
        run = effect[:, t] + decay * run
        out[:, t] = run
    return out


def calibration(
    prediction: Tensor, target: Tensor, select: Tensor | None = None
) -> dict[str, float]:
    """How well ``prediction`` explains ``target``: correlation, the slope of
    target on prediction, and the variance left after subtracting prediction
    (as a fraction of target's)."""
    p, y = prediction.double(), target.double()
    if select is not None:
        p, y = p[select], y[select]
    p, y = p.reshape(-1), y.reshape(-1)
    pc, yc = p - p.mean(), y - y.mean()
    cov = (pc * yc).mean()
    return {
        "n": float(p.numel()),
        "corr": float(cov / (pc.std() * yc.std()).clamp_min(1e-30)),
        "slope": float(cov / pc.square().mean().clamp_min(1e-30)),
        "residual_var_ratio": float((y - p).var() / y.var().clamp_min(1e-30)),
        "target_std": float(y.std()),
        "prediction_std": float(p.std()),
    }
