"""DiffSR dynamics construction and loss on replay windows."""

from __future__ import annotations

from torch import Tensor

from imitationlab.algorithms.common.world_model.config import WMBPTTConfig
from imitationlab.algorithms.common.world_model.task import WorldModelTask
from imitationlab.algorithms.diffsr.dynamics import DiffSRDynamics


def build_dynamics(task: WorldModelTask, config: WMBPTTConfig) -> DiffSRDynamics:
    # World-model input consists of state features, action, and sim context.
    action_end = task.wm_input_dim - task.context_dim
    return DiffSRDynamics(
        task.wm_input_dim,
        task.increment_dim,
        (action_end - task.action_dim, action_end),
        feature_dim=config.diffsr_feature_dim,
        embed_dim=config.diffsr_embed_dim,
        num_noises=config.diffsr_num_noises,
        readout_detach_phi=False,  # Gradients through the readout reach (s, a).
    )


def dynamics_loss(
    readout: Tensor,
    window: dict[str, Tensor],
    steps: int,
    dynamics: DiffSRDynamics,
    task: WorldModelTask,
    config: WMBPTTConfig,
    metrics: dict[str, list],
) -> Tensor:
    """Add the one-step denoising loss on valid, teacher-forced transitions."""
    batch_size = window["state"].shape[0]
    rows = batch_size * steps
    states = window["state"][:, :steps].reshape(rows, -1)
    next_states = window["next_state"][:, :steps].reshape(rows, -1)
    actions = window["action"][:, :steps].reshape(rows, -1)
    context = window["context"][:, :1].expand(batch_size, steps, -1).reshape(rows, -1)
    inputs = task.wm_input(states, actions, context)
    denoise, _ = dynamics.denoising_loss(
        inputs,
        task.increment(states, next_states),
        window["valid"][:, :steps].reshape(-1),
        bf16=config.bf16,
    )
    metrics["wm/readout_loss"].append(readout.detach())
    metrics["wm/denoise_loss"].append(denoise.detach())
    return readout + config.diffsr_denoise_coeff * denoise
