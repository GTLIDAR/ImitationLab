"""PPO learning-rate adaptation from the policy's post-update KL."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import Tensor


class KLLearningRate:
    """Compute the KL diagnostic and apply the configured adaptive schedule."""

    def _prepare_kl_context(
        self, sampled_tensordict: TensorDict, policy_op: TensorDictModule
    ) -> dict[str, Tensor]:
        obs_td = sampled_tensordict.select(*policy_op.in_keys).detach()
        if "loc" not in sampled_tensordict.keys(
            True
        ) or "scale" not in sampled_tensordict.keys(True):
            msg = "Expected 'loc' and 'scale' in sampled_tensordict for KL computation."
            raise KeyError(msg)
        sample_rows = self._kl_sample_rows(sampled_tensordict)
        if sample_rows is not None:
            # Gather only what the diagnostic reads (optim.kl_sample_size).
            sampled_tensordict = sampled_tensordict.select(
                *policy_op.in_keys, "loc", "scale"
            )[sample_rows]
            obs_td = sampled_tensordict.select(*policy_op.in_keys).detach()
        return {
            "obs_td": obs_td,
            "old_loc": sampled_tensordict.get("loc").detach(),
            "old_scale": sampled_tensordict.get("scale").detach(),
        }

    def _kl_sample_rows(self, sampled_tensordict: TensorDict) -> Tensor | None:
        """Row indices for the KL diagnostic, or None for the whole minibatch.

        ``optim.kl_sample_size`` rows of the leading batch dimension, drawn
        uniformly without replacement from a generator owned by this method
        (seeded from ``config.seed``), so enabling it does not shift any other
        consumer of the global RNG.
        """
        optim_cfg = getattr(self.config, "optim", None)
        size = getattr(optim_cfg, "kl_sample_size", None)
        if size is None:
            return None
        size = int(size)
        if size <= 0:
            msg = f"optim.kl_sample_size must be positive or None; got {size}."
            raise ValueError(msg)
        if sampled_tensordict.batch_dims == 0:
            return None
        rows = int(sampled_tensordict.batch_size[0])
        if rows <= size:
            return None
        device = torch.device(sampled_tensordict.device or self.device)
        generator = getattr(self, "_kl_sample_generator", None)
        if generator is None or self._kl_sample_generator_device != device:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(getattr(self.config, "seed", 0)) + 0x4B4C)
            self._kl_sample_generator = generator
            self._kl_sample_generator_device = device
        return torch.randperm(rows, device=device, generator=generator)[:size]

    def _compute_kl_after_update(
        self, kl_context: dict[str, Tensor], policy_op: TensorDictModule
    ) -> Tensor | None:
        kl_approx = self._kl_after_update_tensor(kl_context, policy_op)
        if not torch.isfinite(kl_approx):
            return None
        return kl_approx

    def _kl_after_update_tensor(
        self, kl_context: dict[str, Tensor], policy_op: TensorDictModule
    ) -> Tensor:
        """Post-update KL(old || new) as a device scalar, with no host sync.

        :meth:`_compute_kl_after_update` adds the finiteness check (a host
        sync); the captured update (``ppo.capture_update``) records this
        tensor inside the CUDA graph and applies the same check on the host
        once per iteration.
        """
        with torch.no_grad():
            obs_td = kl_context["obs_td"].clone()
            obs_td = policy_op(obs_td)
            new_loc = obs_td.get("loc")
            new_scale = obs_td.get("scale")
            old_loc = kl_context["old_loc"]
            old_scale = kl_context["old_scale"]

            var_old = old_scale.pow(2)
            var_new = new_scale.pow(2)
            kl = (
                torch.log(new_scale / old_scale)
                + (var_old + (old_loc - new_loc).pow(2)) / (2.0 * var_new)
                - 0.5
            )
            if kl.ndim > 0:
                kl = kl.sum(dim=-1)
            return kl.mean()

    def _record_kl_for_lr_adaptation(
        self, kl_approx: Tensor, schedule_cfg: Any
    ) -> None:
        """Route one minibatch's KL sample to the adaptive-LR rule.

        Under ``optim.kl_adapt_step="update"`` this is a direct call and the
        behaviour is exactly as before. Under ``"iteration"`` the sample is
        buffered instead, and :meth:`_flush_kl_lr_adaptation` applies a single
        bang-bang step per iteration on the mean.
        """
        mode = str(getattr(schedule_cfg, "kl_adapt_step", "update") or "update").lower()
        if mode != "iteration":
            self._maybe_adjust_lr(kl_approx, schedule_cfg)
            return
        kl_value = float(kl_approx.detach().mean().cpu().item())
        if not np.isfinite(kl_value):
            return
        # Lazily created: concrete agents build a lot of state in __init__, and
        # this only has to exist by the first update.
        samples = getattr(self, "_kl_adapt_samples", None)
        if samples is None:
            samples = []
            self._kl_adapt_samples = samples
        samples.append(kl_value)

    def _flush_kl_lr_adaptation(self, schedule_cfg: Any) -> None:
        """Apply the buffered per-iteration adaptive-LR step, if any.

        A no-op under ``kl_adapt_step="update"``, where the buffer never fills,
        so it is safe to call unconditionally at the end of an update loop.
        """
        samples = getattr(self, "_kl_adapt_samples", None)
        if not samples:
            return
        mean_kl = float(np.mean(samples))
        samples.clear()
        self._maybe_adjust_lr(
            torch.as_tensor(mean_kl, dtype=torch.float32), schedule_cfg
        )

    def _maybe_adjust_lr(self, kl_approx: Tensor, schedule_cfg: Any) -> None:
        schedule = (getattr(schedule_cfg, "scheduler", "") or "").lower()
        if schedule != "adaptive":
            return
        kl_value = float(kl_approx.detach().mean().cpu().item())
        if not np.isfinite(kl_value) or kl_value <= 0.0:
            return

        desired_kl = float(getattr(schedule_cfg, "desired_kl", 0.01))
        factor = float(getattr(schedule_cfg, "lr_adaptation_factor", 1.5))
        min_lr = getattr(schedule_cfg, "min_lr", None)
        max_lr = getattr(schedule_cfg, "max_lr", None)

        adaptive_groups = [
            group
            for group in self.optim.param_groups
            if bool(group.get("adaptive_lr", True))
        ]
        if not adaptive_groups:
            return

        reference_group = adaptive_groups[0]
        lr = float(reference_group["lr"])
        if kl_value > desired_kl * 2.0:
            lr /= factor
        elif kl_value < desired_kl / 2.0:
            lr *= factor
        else:
            return

        reference_min_lr = reference_group.get("lr_min", min_lr)
        reference_max_lr = reference_group.get("lr_max", max_lr)
        if reference_min_lr is not None:
            lr = max(lr, float(reference_min_lr))
        if reference_max_lr is not None:
            lr = min(lr, float(reference_max_lr))

        scale = lr / float(reference_group["lr"])
        for group in adaptive_groups:
            group_lr = float(group["lr"]) * scale
            group_min_lr = group.get("lr_min", min_lr)
            group_max_lr = group.get("lr_max", max_lr)
            if group_min_lr is not None:
                group_lr = max(group_lr, float(group_min_lr))
            if group_max_lr is not None:
                group_lr = min(group_lr, float(group_max_lr))
            group["lr"] = group_lr
