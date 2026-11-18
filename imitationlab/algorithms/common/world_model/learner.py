"""World-model BPTT actor-critic (SuperMimic, arXiv 2610.28480).

One iteration:

1. **Collect** ``env_steps_per_iter`` vectorized simulator steps with the
   deterministic actor plus N(0, sigma^2) action noise into a per-env replay.
2. **Fit** the dynamics model on ``wm_batches_per_iter`` batches of k-step
   windows, rolled out autoregressively on the logged actions with gradients
   through every step; the error of each step is expressed in the true state's
   frames and divided by the increment scale.
3. **Improve the actor**: from replay states (history rebuilt as the env's
   buffer holds it), imagine H steps through the frozen model with fresh sensor
   noise and backpropagate J = sum_t gamma^t c_t r_t + gamma^H c_H V(s_H),
   c_{t+1} = c_t (1 - d_{t+1}), through the model's Jacobians (or, with
   ``actor_estimator=score_function``, estimate the same gradient by the
   likelihood ratio of sampled actions on the same rollouts, model detached).
4. **Fit the critic** to GAE(lambda) targets of the detached imagined rollout.

The task model (``task.WorldModelTask``) supplies states, features, reward and
termination; this module is task-agnostic. The simulator is stepped directly
(the Isaac env under the TorchRL wrapper); the TorchRL policy module
``collector_policy`` maps the env's observation TensorDict to the deterministic
action for evaluators.
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch._dynamo
from tensordict.nn import TensorDictModule
from torch import Tensor, nn
from torchrl.envs import TransformedEnv

from imitationlab.algorithms.common.world_model.config import (
    WMBPTTTrainerConfig,
    validate_options,
    validate_residual,
)
from imitationlab.algorithms.diffsr.training import build_dynamics, dynamics_loss
from imitationlab.algorithms.common.world_model.muon import MuonAdam
from imitationlab.algorithms.common.world_model.networks import (
    Critic,
    DynamicsModel,
    RunningNorm,
    build_actor,
)
from imitationlab.algorithms.svg.rebar import CorrectionAverage
from imitationlab.algorithms.svg.rollout import RealWindowUpdates
from imitationlab.algorithms.common.world_model.replay import (
    SequenceReplay,
    rollout_mask,
)
from imitationlab.algorithms.common.world_model.speed_options import (
    ActorAverage,
    ActorLR,
    CheckpointWriter,
    GraphedUpdate,
    apply_compile_options,
    atomic_save,
    cpu_snapshot,
    full_checkpoint_due,
    gated_step,
    set_lr,
    update_norm_on_device,
    zero_nonfinite_grads,
)
from imitationlab.algorithms.common.world_model.task import (
    WorldModelTask,
    load_task,
)
from imitationlab.runners.algorithm_runtime import (
    AlgorithmRuntime,
    honours_train_time_budget,
)
from imitationlab.runners.metrics import TrainingMetadata
from imitationlab.algorithms.common.returns import lambda_returns
from imitationlab.algorithms.common.world_model.metrics import DEVICE_COUNTS
from imitationlab.algorithms.supermimic.rollout import ImaginedRollouts
from imitationlab.algorithms.supermimic.rollout import (
    score_function_surrogate as score_function_surrogate,
)

ARCH_TAG = "wmbptt-v1"
RESUME_MATCH: dict[str, Any] = {
    "actor_ema": 0.0,
    "fused_adam": False,
    "actor_opt": "adam",
    "wm_opt": "adam",
    "actor_residual": False,
    "wm_structured": False,
}
"""``wm`` fields a resumed run must share with its checkpoint, with the value a
checkpoint from before the field implies: they set which optimizers and which
policy entries the checkpoint holds, and what the actor's output means."""
ACTOR_SHAPE_KEYS = (
    "actor_arch",
    "actor_hidden",
    "actor_activation",
    "actor_latent_dim",
    "actor_goal_dim",
)
"""``policy_meta`` entries that set the actor's shape (``wm.actor_arch``)."""
LIGHT_CHECKPOINT_KEYS = (
    "policy_state_dict",
    "policy_meta",
    "wm_config",
    "task",
    "iteration",
    "cumulative_env_frames",
    "cumulative_train_seconds",
)


def actor_shape(meta: dict[str, Any]) -> dict[str, Any]:
    """The actor's shape in ``policy_meta``, with what a checkpoint from before an
    entry implies: a flat ELU MLP."""
    shape = {
        "actor_arch": str(meta.get("actor_arch", "flat")),
        "actor_hidden": [int(h) for h in meta.get("actor_hidden", [])],
        "actor_activation": str(meta.get("actor_activation", "elu")).lower(),
    }
    if shape["actor_arch"] != "flat":
        shape["actor_latent_dim"] = meta.get("actor_latent_dim")
        shape["actor_goal_dim"] = meta.get("actor_goal_dim")
    return shape


def _find_isaac_env(env: Any) -> Any:
    stack, seen = [env], set()
    while stack:
        cur = stack.pop()
        if id(cur) in seen:
            continue
        seen.add(id(cur))
        if hasattr(cur, "step") and hasattr(cur, "reward_manager"):
            return cur
        for attr in ("base_env", "_env", "env", "unwrapped"):
            nxt = getattr(cur, attr, None)
            if nxt is not None:
                stack.append(nxt)
    msg = "WMBPTT needs an Isaac Lab manager-based env under the TorchRL wrapper."
    raise RuntimeError(msg)


def _plain(value: Any) -> Any:
    """Builtin copy of a config value (configclass wraps some strings), so a
    checkpoint loads without the config's own packages."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, str):
        return str(value)
    return value


def _read_scalars(values: dict[str, Any]) -> dict[str, float]:
    """``{name: float(value)}`` in the same order, with one host read for all
    device tensors instead of one each. A float32 value reaches float64 exactly
    either way, so the numbers are ``float``'s."""
    on_device = [
        k for k, v in values.items() if isinstance(v, Tensor) and v.device.type != "cpu"
    ]
    read: dict[str, float] = {}
    if on_device:
        stacked = torch.stack(
            [values[k].detach().reshape(()).double() for k in on_device]
        )
        read = dict(zip(on_device, stacked.tolist(), strict=True))
    return {k: read[k] if k in read else float(v) for k, v in values.items()}


def _stacked(update: Callable[[dict[str, list]], None]) -> dict[str, Tensor]:
    """``update``'s metrics, one stacked device tensor per name: the outputs a
    captured update returns (``wm.update_graphs``)."""
    metrics: dict[str, list] = defaultdict(list)
    update(metrics)
    return {key: torch.stack(values) for key, values in metrics.items()}


class WMPolicy(nn.Module):
    """Deployed policy: env observation tensors -> deterministic action.
    ``residual`` (``wm.actor_residual``) adds the task's reference setpoint."""

    def __init__(
        self, task: WorldModelTask, actor: nn.Module, bf16: bool, residual: bool = False
    ) -> None:
        super().__init__()
        self.task = task
        self.actor = actor
        self.bf16 = bf16
        self.residual = residual

    def forward(self, *tensors: Tensor) -> Tensor:
        obs = self.task.policy_obs_from_env(*tensors)
        action = self.actor(obs, bf16=self.bf16)
        if self.residual:
            action = action + self.task.reference_setpoint(obs)  # type: ignore[attr-defined]
        return action


def deployed_policy(
    env: Any, data: dict[str, Any], device: torch.device | str
) -> TensorDictModule:
    """The policy ``WMBPTT.load_policy`` deploys from the checkpoint payload ``data``,
    built without an agent: the task of ``policy_meta["task_model"]`` on ``env`` with
    the checkpoint's sensor noise, the checkpoint's actor shape, bf16 and residual, and
    ``policy_state_dict`` loaded strictly. The interface checks are ``load_policy``'s.

    ``wm.eprobe_share``'s worker acts with it, so a sweep runs the policy path that an
    evaluator scores (the policy average, deterministic, the residual setpoint). A
    later sweep loads new weights into it with ``load_state_dict``.
    """
    meta = data.get("policy_meta") or {}
    if meta.get("arch") != ARCH_TAG:
        msg = f"not a {ARCH_TAG} policy (arch={meta.get('arch')!r})."
        raise ValueError(msg)
    task = load_task(
        str(meta["task_model"]),
        env,
        device,
        sensor_noise=bool(meta.get("sensor_noise", True)),
    )
    expected: dict[str, Any] = {
        "in_keys": [list(k) for k in task.policy_in_keys],
        "policy_obs_dim": task.policy_obs_dim,
        "action_dim": task.action_dim,
    }
    if meta.get("goal_binding") is not None:
        expected["goal_binding"] = (
            _plain(task.goal_binding()) if hasattr(task, "goal_binding") else None
        )
    for key, value in expected.items():
        if meta.get(key) != value:
            msg = f"policy_meta[{key!r}]={meta.get(key)!r} != the task's {value!r}"
            raise ValueError(msg)
    shape = actor_shape(meta)
    actor = build_actor(
        shape["actor_arch"],
        task.policy_obs_dim,
        task.action_dim,
        shape["actor_hidden"],
        1.0,
        shape["actor_activation"],
        goal_dim=shape.get("actor_goal_dim"),
        latent_dim=int(shape.get("actor_latent_dim") or 32),
    ).to(device)
    residual = bool(meta.get("actor_residual", False))
    if residual:
        validate_residual(task)
        task.bind_reference_setpoint()  # type: ignore[attr-defined]
    bf16 = bool(meta.get("bf16", (data.get("wm_config") or {}).get("bf16", True)))
    policy = TensorDictModule(
        WMPolicy(task, actor, bf16, residual),
        in_keys=list(task.policy_in_keys),
        out_keys=["action"],
    )
    policy.load_state_dict(data["policy_state_dict"], strict=True)
    policy.eval()
    return policy


class WorldModelLearner(
    ImaginedRollouts, RealWindowUpdates, AlgorithmRuntime[WMBPTTTrainerConfig]
):
    """Replay, dynamics fitting, and actor-critic optimization shared by world-model learners."""

    # Options start off; ``__init__`` and ``_construct_policy`` switch them on.
    actor_avg: ActorAverage | None = None
    policy_avg: TensorDictModule | None = None
    _actor_lr: ActorLR = ActorLR()
    _device_counts: Tensor | None = None
    _unpublished_ends: Tensor | None = None
    _checkpoint_writer: CheckpointWriter | None = None
    _graphs: dict[str, GraphedUpdate] | None = None
    _residual: bool = False
    _structured: bool = False
    _mirror_rows: int = 0
    _eprobe: Any = None
    """``wm.eprobe_share``: the env's evaluation probe, from ``train`` to its end."""
    _eprobe_resume: dict[str, Any] | None = None

    def __init__(
        self,
        env: TransformedEnv,
        config: WMBPTTTrainerConfig,
        logger: Any = None,
        **kwargs: Any,
    ) -> None:
        self._num_envs = int(env.batch_size.numel()) if len(env.batch_size) else 1
        device = torch.device(config.device)
        wm = config.wm
        task_kwargs = dict(wm.task_kwargs)
        if wm.goal != "explicit" or wm.goal_encoder:
            task_kwargs.update(
                goal_mode=str(wm.goal), skill_encoder=str(wm.goal_encoder) or None
            )
        if (
            wm.contact_reward > 0
        ):  # only when on, so a task without the keyword still loads by default
            task_kwargs.update(
                contact_reward=float(wm.contact_reward),
                ref_contact_fix=bool(wm.ref_contact_fix),
            )
        if wm.skill_bf16:
            task_kwargs.update(skill_bf16=True)
        self.task: WorldModelTask = load_task(
            wm.task_model, env, device, sensor_noise=wm.sensor_noise, **task_kwargs
        )
        validate_options(config, self.task)
        if wm.goal_table or wm.reference_body_table:
            self.task.build_reference_tables(
                goal=wm.goal_table, bodies=wm.reference_body_table
            )  # type: ignore[attr-defined]
        self._residual = bool(wm.actor_residual)
        if self._residual:
            self.task.bind_reference_setpoint()  # type: ignore[attr-defined]
        self._structured = bool(wm.wm_structured)
        if self._structured:
            self.task.bind_structured()  # type: ignore[attr-defined]
        self._mirror_rows = int(wm.wm_batch_size * wm.wm_mirror)
        if self._mirror_rows:
            self.task.bind_mirror()  # type: ignore[attr-defined]
        self._isaac = _find_isaac_env(env)
        self._frames_processed = 0
        self._resume_frame_offset = 0
        self._iteration = 0
        self._step_acc = 0.0
        self.replay: SequenceReplay | None = None
        self._inference_loaded = False
        self._actor_lr = ActorLR(
            wm.actor_lr_mult,
            wm.actor_lr_decay_return,
            wm.actor_lr_decay_len,
            wm.actor_lr_decay_by,
        )
        self._rebar = (
            CorrectionAverage(wm.rebar_ema_rate, wm.rebar_norm_cap)
            if wm.actor_estimator == "rebar"
            else None
        )
        super().__init__(env=env, config=config, logger=logger, **kwargs)

    # ------------------------------------------------------------ construction
    def _construct_policy(self) -> TensorDictModule:
        wm = self.config.wm
        t = self.task
        self.dynamics = self._construct_dynamics().to(self.device)
        # The flat entries are the baseline's metadata; ``wm.actor_arch=enc_dec`` adds its own.
        self._actor_meta: dict[str, Any] = {
            "actor_hidden": list(wm.actor_hidden),
            "actor_activation": str(wm.actor_activation).lower(),
        }
        if wm.actor_arch != "flat":
            self._actor_meta.update(
                actor_arch=str(wm.actor_arch),
                actor_latent_dim=int(wm.actor_latent_dim),
                actor_goal_dim=int(t.policy_goal_dim),
            )
        self.actor = self._build_actor(actor_shape(self._actor_meta))
        self.critic = Critic(
            t.critic_obs_dim, wm.critic_hidden, wm.critic_activation
        ).to(self.device)
        self.policy_net = WMPolicy(t, self.actor, wm.bf16, self._residual)
        if wm.actor_ema > 0:
            self.actor_avg = ActorAverage(self.actor, wm.actor_ema)
            self.policy_avg = TensorDictModule(
                WMPolicy(t, self.actor_avg.module, wm.bf16, self._residual),
                in_keys=list(t.policy_in_keys),
                out_keys=["action"],
            )
        return TensorDictModule(
            self.policy_net, in_keys=list(t.policy_in_keys), out_keys=["action"]
        )

    def _build_actor(self, shape: dict[str, Any]) -> nn.Module:
        """An actor of ``shape`` (``actor_shape``) on this task's observation and action."""
        t = self.task
        return build_actor(
            shape["actor_arch"],
            t.policy_obs_dim,
            t.action_dim,
            shape["actor_hidden"],
            self.config.wm.actor_output_scale,
            shape["actor_activation"],
            goal_dim=shape.get("actor_goal_dim"),
            latent_dim=int(shape.get("actor_latent_dim") or 32),
        ).to(self.device)

    def _construct_dynamics(self) -> nn.Module:
        wm, t = self.config.wm, self.task
        if wm.dynamics_model == "mlp":
            return DynamicsModel(
                t.wm_input_dim,
                t.increment_dim,
                wm.wm_hidden,
                structured=self._structured,
            )
        if wm.dynamics_model != "diffsr":
            msg = f"wm.dynamics_model must be 'mlp' or 'diffsr', got {wm.dynamics_model!r}."
            raise ValueError(msg)
        return build_dynamics(t, wm)

    def _dynamics_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.dynamics.parameters() if p.requires_grad]

    def _construct_feature_extractor(self) -> TensorDictModule:
        msg = "WMBPTT has no feature extractor."
        raise NotImplementedError(msg)

    def _construct_actor_critic(self) -> TensorDictModule:
        assert self.policy is not None
        return self.policy

    @property
    def _deploys_average(self) -> bool:
        """The average exists once the actor has updated; before that it would
        hold stale normalizers, so the live actor is the deployed one."""
        return self.actor_avg is not None and self.actor_avg.updates > 0

    @property
    def collector_policy(self) -> TensorDictModule:
        """The policy evaluators run. Collection during training uses ``actor``."""
        assert self.policy is not None
        return self.policy_avg if self._deploys_average else self.policy  # type: ignore[return-value]

    def _construct_data_buffer(self) -> None:  # type: ignore[override]
        return None  # allocated lazily in train(): evaluation never needs it

    def _construct_collector(
        self, env: TransformedEnv, policy: TensorDictModule
    ) -> None:  # type: ignore[override]
        del env, policy  # the simulator is stepped directly

    def _construct_loss_module(self) -> nn.Module:
        return nn.ModuleDict({"dynamics": self.dynamics, "critic": self.critic})

    def _configure_optimizers(self):  # type: ignore[override]
        wm = self.config.wm
        if wm.update_graphs:
            self._configure_capturable_optimizers()
            return
        adam_kw = {"fused": True} if wm.fused_adam else {}
        if wm.wm_opt == "muon":
            self.wm_optim = MuonAdam(
                self.dynamics,
                [self.dynamics.net],
                lr=wm.wm_lr,
                betas=(0.9, 0.999),
                **adam_kw,
            )
        else:
            self.wm_optim = torch.optim.AdamW(
                self._dynamics_parameters(),
                lr=wm.wm_lr,
                weight_decay=wm.wm_weight_decay,
                **adam_kw,
            )
        betas = tuple(wm.actor_betas)
        if wm.actor_opt == "muon":
            self.actor_optim = MuonAdam(
                self.actor.net,
                self.actor.muon_mlps(),
                lr=wm.actor_lr,
                betas=betas,
                **adam_kw,
            )
        else:
            self.actor_optim = torch.optim.Adam(
                self.actor.net.parameters(), lr=wm.actor_lr, betas=betas, **adam_kw
            )
        self.critic_optim = torch.optim.Adam(
            self.critic.net.parameters(), lr=wm.critic_lr, **adam_kw
        )
        self.lr_scheduler = None

    def _configure_capturable_optimizers(self) -> None:
        """``wm.update_graphs``: the same optimizers as capturable fused kernels, each
        with its own 0-dim device learning rate (one shared tensor would give every
        optimizer one rate), which ``set_lr`` fills in place for the graphs."""
        wm = self.config.wm

        def lr(value: float) -> Tensor:
            return torch.tensor(float(value), device=self.device)

        cap = {"fused": True, "capturable": True}
        if wm.wm_opt == "muon":
            self.wm_optim = MuonAdam(
                self.dynamics,
                [self.dynamics.net],
                lr=wm.wm_lr,
                betas=(0.9, 0.999),
                capturable=True,
            )
        else:
            self.wm_optim = torch.optim.AdamW(
                self._dynamics_parameters(),
                lr=lr(wm.wm_lr),
                weight_decay=wm.wm_weight_decay,
                **cap,
            )
        betas = tuple(wm.actor_betas)
        if wm.actor_opt == "muon":
            self.actor_optim = MuonAdam(
                self.actor.net,
                self.actor.muon_mlps(),
                lr=wm.actor_lr,
                betas=betas,
                capturable=True,
            )
        else:
            self.actor_optim = torch.optim.Adam(
                self.actor.net.parameters(), lr=lr(wm.actor_lr), betas=betas, **cap
            )
        self.critic_optim = torch.optim.Adam(
            self.critic.net.parameters(), lr=lr(wm.critic_lr), **cap
        )
        self.lr_scheduler = None

    def _conform_loaded_optimizers(self) -> None:
        """After ``load_state_dict``, which takes each group's settings from the
        checkpoint: the learning-rate form and ``capturable`` this agent's
        ``wm.update_graphs`` needs, so a checkpoint resumes with the option switched
        either way. A float rate left in a captured step would freeze there."""
        graphs = self.config.wm.update_graphs
        for opt in (self.wm_optim, self.actor_optim, self.critic_optim):
            parts = (opt.muon, opt.adam) if isinstance(opt, MuonAdam) else (opt,)
            for part in parts:
                adam = not isinstance(part, torch.optim.Muon)
                for group in part.param_groups:
                    value = group["lr"]
                    if graphs and not torch.is_tensor(value):
                        group["lr"] = torch.tensor(float(value), device=self.device)
                    elif not graphs and torch.is_tensor(value):
                        group["lr"] = float(value)
                    if adam and "capturable" in group:
                        group["capturable"] = graphs

    def _compile_components(self) -> None:
        self.compile_mode = None
        wm = self.config.wm
        if wm.compile:
            # The two compiled steps have a handful of guard variants (train vs
            # eval mode, first call); a low limit silently falls back to eager.
            torch._dynamo.config.recompile_limit = max(
                torch._dynamo.config.recompile_limit, 64
            )

    def _apply_training_compile_options(self) -> None:
        """The options that change the process or the env, at the first collection
        (``ensure_replay``), not at construction.

        An agent built only to score (``load_policy``) then changes nothing: an
        evaluator builds its env first, env setup compiles, and a late
        ``wm.elementwise_3x3`` would refuse the agent. The env reward stays eager,
        so a board scores every arm with the same rounding. A training entrypoint
        applies the options before its env (``apply_compile_options``); here they
        then change nothing, or refuse a process that compiled without them.
        """
        wm = self.config.wm
        apply_compile_options(wm)
        if wm.compile_eager:
            self.task.compile_collection_reward()  # type: ignore[attr-defined]

    def _progress_summary_fields(self) -> tuple[tuple[str, str], ...]:
        return (
            ("wm/loss", "wm"),
            ("actor/objective", "J"),
            ("imagine/alive_H", "alive"),
        )

    def predict(self, obs: Tensor | np.ndarray) -> Tensor:  # type: ignore[override]
        """The deployed action for raw policy observations ``obs``."""
        x = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        actor = self.actor_avg.module if self._deploys_average else self.actor  # type: ignore[union-attr]
        with torch.no_grad():
            return self._act(actor, x, self.policy_net.bf16)

    def _act(self, actor: nn.Module, obs: Tensor, bf16: bool) -> Tensor:
        """The action of ``actor`` on raw policy observations: its output, plus the
        reference setpoint with ``wm.actor_residual``."""
        action = actor(obs, bf16=bf16)
        if self._residual:
            action = action + self.task.reference_setpoint(obs)  # type: ignore[attr-defined]
        return action

    def _set_residual(self, on: bool) -> None:
        """The action semantics an inference load takes from its checkpoint."""
        if on and not self._residual:
            validate_residual(self.task)
            self.task.bind_reference_setpoint()  # type: ignore[attr-defined]
        self._residual = on
        for policy in (self.policy, self.policy_avg):
            if policy is not None:
                policy.module.residual = on

    # ------------------------------------------------------------ schedule
    def _lr_factor(self) -> float:
        wm = self.config.wm
        p = min(1.0, self._iteration / max(1, wm.lr_cosine_iters))
        return wm.lr_final_ratio + (1 - wm.lr_final_ratio) * 0.5 * (
            1 + math.cos(math.pi * p)
        )

    def _set_lrs(self) -> None:
        f = self._lr_factor()
        wm = self.config.wm
        actor_lr = wm.actor_lr * self._actor_lr.at(self._iteration)
        for opt, base in (
            (self.wm_optim, wm.wm_lr),
            (self.actor_optim, actor_lr),
            (self.critic_optim, wm.critic_lr),
        ):
            set_lr(opt, base * f)

    # ------------------------------------------------------------ collection
    def _policy_features(self, rows: dict[str, Tensor]) -> Tensor:
        """The collector's policy input; its sensor noise is keyed to the
        replay row this step writes (``_real_rollout`` regenerates it)."""
        assert self.replay is not None
        step = torch.full_like(self._env_ids, self.replay.steps)
        kwargs = {}
        if getattr(self.task, "proprio_at", None) is not None:
            kwargs["episode_step"] = rows["episode_step"]
        return self.task.policy_obs(
            rows["hist_states"],
            rows["hist_actions"],
            rows["context"],
            rows["rank"],
            rows["frame"],
            noise_key=(self._noise_seed, self._env_ids, step),
            **kwargs,
        )

    def _update_norm(self, norm: RunningNorm, x: Tensor) -> None:
        if self.config.wm.no_sync:
            update_norm_on_device(norm, x)
        else:
            norm.update(x)

    @torch.no_grad()
    def _collect_step(
        self, obs: dict[str, Any], stats: dict[str, list]
    ) -> dict[str, Any]:
        wm = self.config.wm
        rows = self.task.read_obs(obs[self.task.obs_group])
        feat = self._policy_features(rows)
        self._update_norm(self.actor.obs_norm, feat)
        mean_action = self._act(self.actor, feat, wm.bf16)
        action = mean_action
        if wm.action_noise_std > 0:
            action = action + wm.action_noise_std * torch.randn_like(action)
        action = action.clamp(-wm.action_clip, wm.action_clip)
        assert self.replay is not None
        self.replay.add(
            rows["state"],
            action,
            rows["hist_actions"][:, -1],
            rows["context"],
            rows["rank"],
            rows["frame"],
            rows["episode_step"],
            rows["push"],
            policy_action=mean_action,
        )
        obs, reward, terminated, truncated, extras = self._isaac.step(action)
        self.replay.set_outcome(reward.reshape(-1), terminated.reshape(-1).bool())
        done = (terminated | truncated).reshape(-1).bool()
        self._track_episodes(reward.reshape(-1), done)
        if self._unpublished_ends is not None:
            self._read_true_outcome(rows, action, obs, extras, done, stats)
        stats["env_reward"].append(reward.mean())
        log = extras.get("log") if isinstance(extras, dict) else None
        if log:
            for k, value in log.items():
                if isinstance(value, torch.Tensor):
                    value = value.float().mean()  # noqa: PLW2901
                # ``_flush`` moves a host value to the device at the log row.
                stats[k].append(
                    value
                    if wm.no_sync
                    else torch.as_tensor(value, device=self.device, dtype=torch.float32)
                )
        self._frames_processed += self._num_envs
        return obs

    def _read_true_outcome(
        self,
        rows: dict[str, Tensor],
        action: Tensor,
        obs: dict[str, Any],
        extras: Any,
        done: Tensor,
        stats: dict[str, list],
    ) -> None:
        """What the step really reached. Where an episode ended, the rows of ``obs``
        are already the next episode's, so the env publishes the true post-step rows
        in ``final_obs``, batched, before the reset overwrites them."""
        wm = self.config.wm
        final = extras.get("final_obs") if isinstance(extras, dict) else None
        ended = None
        if final is None:
            self._unpublished_ends += done.sum()
        else:
            ids = final["_env_ids"].long()
            ended = self.task.read_obs(final["obs"][self.task.obs_group])
            if wm.fit_terminal_transitions:
                finite = torch.isfinite(ended["state"]).all(-1)
                stats["wm/terminal_nonfinite"].append((~finite).sum().float())
                self.replay.set_terminal(ids, ended["state"], finite & ~ended["push"])
        if wm.norm_from_collection:
            after = self.task.read_obs(obs[self.task.obs_group])
            state, push = (
                after["state"].contiguous(),
                after["push"],
            )  # one layout, for the compiled features
            if ended is not None:
                state = state.index_copy(0, ids, ended["state"])
                push = push.index_copy(0, ids, ended["push"])
            keep = (
                ~push
                & torch.isfinite(state).all(-1)
                & torch.isfinite(rows["state"]).all(-1)
            )
            self._update_wm_norms((rows["state"], state, action, rows["context"]), keep)

    @torch.no_grad()
    def _track_episodes(self, reward: Tensor, done: Tensor) -> None:
        """Device-side sums behind ``episode/length`` and ``episode/return``."""
        self._episode_length += 1
        self._episode_return += reward
        if self._actor_lr.watches_return:
            self._track_normalized_return(done)
        ended = (self._episode_length, self._episode_return)
        self._episode_sums += torch.stack(
            (done.sum(), *(torch.where(done, x, 0.0).sum() for x in ended))
        )
        for x in ended:
            x.masked_fill_(done, 0.0)

    @torch.no_grad()
    def _track_normalized_return(self, done: Tensor) -> None:
        """Device-side sums of the return ``actor_lr_decay_return`` watches: the
        mean over this step's ended episodes of return / (length * max reward),
        and the count of steps that ended one. The mean of ratios, so
        ``episode/return`` over ``episode/length`` would not reproduce it."""
        ratio = self._episode_return / (
            self._episode_length * self.task.max_step_reward
        )
        ended = done.sum()
        step_mean = torch.where(done, ratio, 0.0).sum() / ended.clamp_min(1)
        self._return_sums += torch.stack((step_mean, (ended > 0).float()))

    # ------------------------------------------------------------ world model
    def _next_state(self, state: Tensor, out: Tensor) -> Tensor:
        """The state the world model's denormalized output ``out`` predicts from ``state``."""
        if self._structured:
            return self.task.structured_integrate(state, out)  # type: ignore[attr-defined]
        return self.task.integrate(state, out)

    def _error_scale(self) -> Tensor:
        """What divides the state error: the increments' std, floored."""
        norm = self.dynamics.inc_norm if self._structured else self.dynamics.out_norm
        return torch.maximum(norm.std(), self.task.increment_floor)

    def _wm_step(
        self, state: Tensor, action: Tensor, ctx: Tensor, target: Tensor
    ) -> tuple[Tensor, Tensor]:
        inc = self.dynamics(
            self.task.wm_input(state, action, ctx), bf16=self.config.wm.bf16
        )
        nxt = self._next_state(state, inc)
        return nxt, self.task.state_error(nxt, target)

    def _wm_step_fn(self):
        fn = getattr(self, "_compiled_wm_step", None)
        if fn is None:
            fn = (
                torch.compile(self._wm_step, dynamic=False)
                if self.config.wm.compile
                else self._wm_step
            )
            self._compiled_wm_step = fn
        return fn

    def _wm_rollout(
        self, win: dict[str, Tensor], k: int, *, grad: bool
    ) -> tuple[Tensor, Tensor]:
        """Open-loop rollout on logged actions; per-step errors (B, k, D) and
        the window mask (B, k) (``rollout_mask``)."""
        state = win["state"][:, 0]
        ctx = win["context"][:, 0]
        if grad:
            state = (
                state.detach().contiguous().requires_grad_(True)
            )  # one compiled variant
        errs = []
        step = self._wm_step_fn() if grad else self._wm_step
        for j in range(k):
            state, err = step(state, win["action"][:, j], ctx, win["next_state"][:, j])
            errs.append(err)
            if not grad:
                state = state.detach()
        return torch.stack(errs, 1), rollout_mask(win, k)

    def _update_world_model(self, metrics: dict[str, list]) -> None:
        wm = self.config.wm
        k = wm.wm_window
        assert self.replay is not None
        scale = self._error_scale()
        for _ in range(wm.wm_batches_per_iter):
            tau, env = self.replay.sample_starts(wm.wm_batch_size, k)
            win = self.replay.window(tau, env, k + 1)
            if not wm.norm_from_collection:
                self._update_wm_norms(self._first_transition(win), win["valid"][:, 0])
                scale = self._error_scale()
            if (
                self._mirror_rows
            ):  # after the normalizers, which never see mirrored data
                win = self.task.mirror_window(win, self._mirror_rows)  # type: ignore[attr-defined]
            errs, mask = self._wm_rollout(win, k, grad=True)
            per = (errs / scale).square().mean(-1)  # (B, k)
            loss = (per * mask).sum() / mask[:, 0].sum().clamp_min(1.0)
            if wm.dynamics_model == "diffsr":
                loss = dynamics_loss(
                    loss, win, k, self.dynamics, self.task, wm, metrics
                )
            self.wm_optim.zero_grad(set_to_none=True)
            loss.backward()
            gn = nn.utils.clip_grad_norm_(
                self._dynamics_parameters(), wm.wm_grad_clip_norm
            )
            if wm.no_sync:
                self._step_if_finite(
                    self.wm_optim, self._dynamics_parameters(), gn, "wm/nonfinite"
                )
            elif torch.isfinite(gn):
                self.wm_optim.step()
            metrics["wm/loss"].append(loss.detach())
            metrics["wm/grad_norm"].append(gn.detach())
            metrics["wm/valid_window_frac"].append(mask[:, -1].mean())
            if wm.fit_terminal_transitions:
                metrics["wm/terminal_frac"].append(
                    (mask * win["ended"][:, :k]).sum() / mask.sum().clamp_min(1.0)
                )
        norms = self._wm_norms()
        if wm.no_sync:
            full = self.dynamics.out_norm.count >= wm.norm_freeze_samples
            for norm in norms:
                norm.frozen.logical_or_(full)
        elif self.dynamics.out_norm.count >= wm.norm_freeze_samples:
            for norm in norms:
                norm.frozen.fill_(True)

    def _wm_norms(self) -> tuple[RunningNorm, ...]:
        """The world model's normalizers, in the order of ``_norm_features``."""
        d = self.dynamics
        return (
            (d.in_norm, d.out_norm, d.inc_norm)
            if self._structured
            else (d.in_norm, d.out_norm)
        )

    def _update_wm_norms(
        self, rows: tuple[Tensor, Tensor, Tensor, Tensor], valid: Tensor
    ) -> None:
        """The world model's normalizers on the ``valid`` transitions of ``rows``
        (state, next state, action, context), until they hold enough samples.
        ``wm.no_sync``: the invalid rows are masked, with no host read."""
        wm, d = self.config.wm, self.dynamics
        if wm.no_sync:
            live = d.out_norm.count < wm.norm_freeze_samples
            features = self._norm_features_fn()(*rows)
            for norm, x in zip(self._wm_norms(), features, strict=True):
                update_norm_on_device(norm, x, valid, live)
        elif d.out_norm.count < wm.norm_freeze_samples and valid.any():
            features = self._valid_norm_features(rows, valid)
            for norm, x in zip(self._wm_norms(), features, strict=True):
                norm.update(x)

    @staticmethod
    def _first_transition(
        win: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return (
            win["state"][:, 0],
            win["next_state"][:, 0],
            win["action"][:, 0],
            win["context"][:, 0],
        )

    def _norm_features(
        self, state: Tensor, next_state: Tensor, action: Tensor, ctx: Tensor
    ) -> tuple[Tensor, ...]:
        """What the world model's normalizers fit (``_wm_norms``): its input and its
        output, and with ``wm.wm_structured`` also the plain increments."""
        inc = self.task.increment(state, next_state)
        if self._structured:
            return (
                self.task.wm_input(state, action, ctx),
                self.task.structured_targets(state, next_state),
                inc,
            )  # type: ignore[attr-defined]
        return self.task.wm_input(state, action, ctx), inc

    def _norm_features_fn(self):
        fn = getattr(self, "_compiled_norm_features", None)
        if fn is None:
            fn = (
                torch.compile(self._norm_features, dynamic=False)
                if self.config.wm.compile_eager
                else self._norm_features
            )
            self._compiled_norm_features = fn
        return fn

    def _valid_norm_features(
        self, rows: tuple[Tensor, Tensor, Tensor, Tensor], v: Tensor
    ) -> tuple[Tensor, ...]:
        """The normalizers' features of the transitions ``v`` keeps."""
        if self.config.wm.compile_eager:
            # Every row, so the compiled function sees one shape; features are per row.
            return tuple(x[v] for x in self._norm_features_fn()(*rows))
        return self._norm_features(*(row[v] for row in rows))

    def _step_if_finite(
        self, optimizer: Any, params: Iterable[Tensor], grad_norm: Tensor, count: str
    ) -> None:
        """``wm.no_sync``: ``optimizer``'s step unless ``grad_norm`` is non-finite,
        decided on the device; a skip adds one to the named device count."""
        finite = torch.isfinite(grad_norm)
        self._device_counts[DEVICE_COUNTS.index(count)] += ~finite
        zero_nonfinite_grads(params, finite)
        gated_step(optimizer, finite)

    @torch.no_grad()
    def _wm_eval(self) -> dict[str, float]:
        """Open-loop k-step error on replay windows in physical units."""
        wm = self.config.wm
        assert self.replay is not None
        L = wm.eval_window
        if self.replay.steps - self.replay.oldest < L + 2:
            return {}
        tau, env = self.replay.sample_starts(wm.eval_batch, L)
        win = self.replay.window(tau, env, L + 1)
        errs, mask = self._wm_rollout(win, L, grad=False)
        # Device values, read together: one host read per horizon (the selection's
        # size) and one for all values, rather than one per value.
        out: dict[str, Tensor] = {}
        scale = self._error_scale()
        for k in (1, 8, 32):
            if k > L:
                continue
            m = mask[:, k - 1] > 0
            e = errs[m, k - 1]
            if e.shape[0] == 0:
                continue
            out[f"wm_eval/k{k}_root_pos_mm"] = e[:, 0:3].norm(dim=-1).mean() * 1000
            out[f"wm_eval/k{k}_root_rot_deg"] = (
                e[:, 3:6].norm(dim=-1).mean() * 180 / math.pi
            )
            out[f"wm_eval/k{k}_root_lin_vel"] = e[:, 6:9].norm(dim=-1).mean()
            out[f"wm_eval/k{k}_joint_pos_deg"] = (
                e[:, 12:41].abs().mean() * 180 / math.pi
            )
            out[f"wm_eval/k{k}_joint_vel"] = e[:, 41:70].abs().mean()
            out[f"wm_eval/k{k}_normalized"] = (e / scale).square().mean()
            out[f"wm_eval/k{k}_windows"] = m.float().sum()
        return _read_scalars(out)

    # ------------------------------------------------------------ actor and critic
    def _critic_obs_fn(self):
        fn = getattr(self, "_compiled_critic_obs", None)
        if fn is None:
            fn = self.task.critic_obs
            if self.config.wm.compile_eager:
                fn = torch.compile(fn, dynamic=False)
            self._compiled_critic_obs = fn
        return fn

    def _critic_obs_seq(self, out: dict[str, Tensor]) -> Tensor:
        """Critic inputs (B * (H + 1), D) of a rollout's states."""
        states = out["states"]
        B, H1, S = states.shape
        frames = out["frame0"].unsqueeze(1) + torch.arange(H1, device=self.device)
        rank = out["rank"].unsqueeze(1).expand(B, H1)
        ctx = out["ctx"].unsqueeze(1).expand(B, H1, out["ctx"].shape[-1])
        return self._critic_obs_fn()(
            states.reshape(B * H1, S),
            ctx.reshape(B * H1, -1),
            rank.reshape(-1),
            frames.reshape(-1),
        )

    def _critic_values(self, out: dict[str, Tensor]) -> Tensor:
        B, H1 = out["states"].shape[:2]
        return self.critic(self._critic_obs_seq(out), bf16=self.config.wm.bf16).reshape(
            B, H1
        )

    def _actor_rollout(self) -> dict[str, Tensor]:
        """Select the rollout from a historical WMBPTT configuration."""
        wm = self.config.wm
        if wm.actor_source == "real":
            return self._real_rollout(wm.imagine_batch, grad=True)
        return self._imagine(
            wm.imagine_batch, grad=True, start_perturb=wm.imag_start_perturb
        )

    def _update_actor_critic(self, metrics: dict[str, list]) -> None:
        wm = self.config.wm
        real_actor = wm.actor_source == "real"
        self.dynamics.requires_grad_(False)
        self.critic.requires_grad_(False)
        try:
            self.actor_optim.zero_grad(set_to_none=True)
            if self._rebar is not None:
                out = self._rebar_gradient(metrics)
            else:
                out = self._actor_rollout()
                loss = -out["objective"].mean()
                loss.backward()
        finally:
            self.dynamics.requires_grad_(True)
            self.critic.requires_grad_(True)
        params = list(self.actor.net.parameters())
        gn = nn.utils.clip_grad_norm_(params, wm.actor_grad_clip_norm)
        nn.utils.clip_grad_value_(params, wm.actor_grad_clip_value)
        if wm.no_sync:
            self._step_if_finite(self.actor_optim, params, gn, "actor/nonfinite")
        elif torch.isfinite(gn):
            self.actor_optim.step()
        if self.actor_avg is not None:
            self.actor_avg.update()
        metrics["actor/objective"].append(
            out.get("J", out["objective"].detach()).mean()
        )
        metrics["actor/grad_norm"].append(gn.detach())
        if "advantages" in out:
            metrics["actor/sf_surrogate"].append(out["objective"].detach().mean())
            metrics["actor/sf_adv_std"].append(out["advantages"].std())
        if real_actor:
            metrics["real/alive_H"].append(out["inep"][:, -1].mean())
            metrics["real/term_frac"].append(out["term"].sum(1).mean())
            metrics["real/trunc_frac"].append(out["trunc"].sum(1).mean())
            # In-episode steps only: past an episode end the re-simulation
            # runs on without a recording (masked out of the objective).
            inep1 = out["inep"][:, 1:]
            metrics["real/reward_mean"].append(
                (out["rewards"].detach() * inep1).sum() / inep1.sum().clamp_min(1.0)
            )
            metrics["real/pelvis_dev_mm"].append(out["pelvis_dev_mm"])
            metrics["real/action_dev0"].append(out["action_dev0"])
        else:
            metrics["imagine/alive_H"].append(out["alive_H"].detach().mean())
            metrics["imagine/reward_mean"].append(out["rewards"].detach().mean())
            metrics["imagine/done_mean"].append(out["dones"].detach().mean())
        try:
            if wm.critic_fit and self._iteration % max(1, wm.critic_update_every) == 0:
                if wm.critic_source == "real":
                    # The recorded states, not the actor's re-simulated ones.
                    self._update_critic_real(
                        self._real_rollout(wm.imagine_batch, grad=False), metrics
                    )
                else:
                    self._update_critic(
                        self._imagine(wm.imagine_batch, grad=False)
                        if real_actor
                        else out,
                        metrics,
                    )
        finally:
            if wm.reference_window:
                # The critic read the rollout's rows by window slot; bank ranks again.
                self.task.library.close_window()  # type: ignore[attr-defined]

    def _update_critic(self, out: dict[str, Tensor], metrics: dict[str, list]) -> None:
        wm = self.config.wm
        with torch.no_grad():
            states = out["states"].detach()  # (B, H+1, S)
            B, H1, S = states.shape
            H = H1 - 1
            frames = out["frame0"].unsqueeze(1) + torch.arange(H1, device=self.device)
            rank = out["rank"].unsqueeze(1).expand(B, H1)
            ctx = out["ctx"].unsqueeze(1).expand(B, H1, out["ctx"].shape[-1])
            cobs = self._critic_obs_fn()(
                states.reshape(B * H1, S),
                ctx.reshape(B * H1, -1),
                rank.reshape(-1),
                frames.reshape(-1),
            )
            self._update_norm(self.critic.obs_norm, cobs)
            values = self.critic(cobs, bf16=wm.bf16).reshape(B, H1)
            notdone = (1 - out["dones"].detach()) * out["cont"].detach()
            targets = lambda_returns(
                out["rewards"].detach(), values, notdone, wm.gamma, wm.gae_lambda
            )
            cobs = cobs.reshape(B, H1, -1)[:, :H].reshape(B * H, -1)
            targets = targets.reshape(-1)
        self._fit_critic(cobs, targets, metrics)

    def _fit_critic(
        self, cobs: Tensor, targets: Tensor, metrics: dict[str, list]
    ) -> None:
        """``critic_epochs`` x ``critic_minibatches`` regression steps."""
        wm = self.config.wm
        n = targets.shape[0]
        if n == 0:
            return
        mb = max(1, n // wm.critic_minibatches)
        losses = []
        for _ in range(wm.critic_epochs):
            perm = torch.randperm(n, device=self.device)
            for j in range(wm.critic_minibatches):
                idx = perm[j * mb : (j + 1) * mb]
                v = self.critic(cobs[idx], bf16=wm.bf16)
                loss = (v - targets[idx]).square().mean()
                self.critic_optim.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.critic.net.parameters(), wm.critic_grad_clip_norm
                )
                self.critic_optim.step()
                losses.append(loss.detach())
        metrics["critic/loss"].append(torch.stack(losses).mean())
        metrics["critic/target_mean"].append(targets.mean())

    # ------------------------------------------------------------ loop
    def _flush(
        self, metrics: dict[str, list], stats: dict[str, list], it_time: float
    ) -> dict[str, float]:
        means: dict[str, Tensor] = {}
        for src in (metrics, stats):
            for k, vals in src.items():
                if not vals:
                    continue
                name = k if "/" in k else f"env/{k}"
                values = [
                    torch.as_tensor(v, device=self.device, dtype=torch.float32)
                    for v in vals
                ]
                means[name] = torch.stack(values).mean()
            src.clear()
        out = _read_scalars(means)  # one host read for every mean
        out.update(self._wm_eval())
        if self._unpublished_ends is not None:
            self._check_final_obs()
        out["train/lr_factor"] = self._lr_factor()
        if self._actor_lr.on:
            self._observe_actor_lr(out)
        if self._device_counts is not None:
            self._read_device_counts(out)
        out["perf/iteration_s"] = it_time
        if "env/env_reward" in out:
            out["train/step_reward_mean"] = out["env/env_reward"]
        count, length, reward = self._episode_sums.tolist()
        self._episode_sums.zero_()
        if count > 0:
            out["episode/length"] = length / count
            out["episode/return"] = reward / count
        if self._eprobe is not None:
            out.update(self._eprobe.log_metrics())  # host values, set once per sweep
        return out

    def _check_final_obs(self) -> None:
        """At a log row: an env without ``final_obs`` must not turn the option off quietly."""
        missing = int(self._unpublished_ends)
        if missing:
            msg = (
                f"wm.fit_terminal_transitions and wm.norm_from_collection read the true last state of an episode, "
                f"and {missing} episodes ended since the last log row with no final_obs. This env does not publish it."
            )
            raise RuntimeError(msg)

    def _read_device_counts(self, row: dict[str, float]) -> None:
        """``wm.no_sync``: at a log row, the only host read of the device counts."""
        row.update(zip(DEVICE_COUNTS, self._device_counts.tolist(), strict=True))
        self._device_counts.zero_()
        if row["imagine/invalid_starts"] > 0:
            msg = (
                f"wm.no_sync: {row['imagine/invalid_starts']:.0f} imagination starts since the last log row had fewer "
                "than the horizon's frames left, because under half of the candidates were valid. Unset wm.no_sync or "
                "reduce wm.horizon."
            )
            raise RuntimeError(msg)

    def _observe_actor_lr(self, row: dict[str, float]) -> None:
        """At a log row, the only host read of the return's device sums."""
        ret = None
        if self._actor_lr.watches_return:
            total, steps = self._return_sums.tolist()
            self._return_sums.zero_()
            if steps > 0:
                ret = total / steps
                row["train/normalized_return"] = ret
        message = self._actor_lr.observe(self._iteration, ret)
        if message:
            self.log.info(message)
        row["train/actor_lr_mult"] = self._actor_lr.at(self._iteration)

    def _env_log_metrics(self) -> dict[str, Any]:
        """The env's live scalars; ``_collect_step`` averages ``extras["log"]`` itself."""
        reader = self._env_attribute("live_log_metrics", None)
        if not callable(reader):
            return {}
        return _read_scalars(reader())

    def _checkpoint_dir(self) -> Path:
        return self.log_dir / self.config.logger.save_path

    def ensure_replay(self) -> SequenceReplay:
        """Allocate the per-env replay on first use (training, or an offline
        probe that collects with ``_collect_step``)."""
        if self.replay is None:
            self._apply_training_compile_options()
            wm, t = self.config.wm, self.task
            self.replay = SequenceReplay(
                self._num_envs,
                wm.replay_steps,
                t.state_dim,
                t.action_dim,
                t.context_dim,
                self.device,
                terminal_states=wm.fit_terminal_transitions,
            )
            # Collection draws its sensor noise as a function of (seed, env,
            # replay row), so a real-window re-simulation sees the same noise.
            self._noise_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
            self._env_ids = torch.arange(self._num_envs, device=self.device)
            self._episode_length = torch.zeros(self._num_envs, device=self.device)
            self._episode_return = torch.zeros(self._num_envs, device=self.device)
            self._episode_sums = torch.zeros(3, device=self.device)
            if self._actor_lr.watches_return:
                self._return_sums = torch.zeros(2, device=self.device)
            if wm.no_sync:
                self._device_counts = torch.zeros(
                    len(DEVICE_COUNTS), device=self.device
                )
            if wm.fit_terminal_transitions or wm.norm_from_collection:
                self._unpublished_ends = torch.zeros(
                    (), dtype=torch.long, device=self.device
                )
            if wm.update_graphs:
                self.replay.use_device_head(fill=wm.graph_early)
                self._build_update_graphs()
        return self.replay

    # ------------------------------------------------------------ update graphs
    def _build_update_graphs(self) -> None:
        """``wm.update_graphs``: the two updates as ``GraphedUpdate``s on one memory
        pool, replayed in the order they capture: world model, then actor-critic.
        Both wait for a full ring, where the sampling bounds stop changing size, or
        with ``wm.graph_early`` for nothing, the fill level being on the device; the
        actor-critic update also waits for the policy average's first copy, which is
        a host branch."""
        rp = self.replay
        assert rp is not None
        pool = torch.cuda.graph_pool_handle()
        early = self.config.wm.graph_early

        def sampling_capturable() -> bool:
            return early or rp.steps >= rp.capacity

        def average_started() -> bool:
            return self.actor_avg is None or self.actor_avg.updates > 0

        self._graphs = {
            "world_model": GraphedUpdate(
                lambda: _stacked(self._update_world_model),
                sampling_capturable,
                "the world-model update",
                pool=pool,
                log=self.log.info,
            ),
            "actor_critic": GraphedUpdate(
                lambda: _stacked(self._update_actor_critic),
                lambda: sampling_capturable() and average_started(),
                "the actor-critic update",
                pool=pool,
                on_replay=self._count_average_step,
                log=self.log.info,
            ),
        }

    def _count_average_step(self) -> None:
        """The captured policy average's step leaves its host counter behind."""
        if self.actor_avg is not None:
            self.actor_avg.updates += 1

    def _run_update(
        self,
        name: str,
        update: Callable[[dict[str, list]], None],
        metrics: dict[str, list],
    ) -> None:
        """``update`` eagerly, or through its graph with ``wm.update_graphs``."""
        if self._graphs is None:
            update(metrics)
            return
        for key, values in self._graphs[name]().items():
            metrics[key].extend(values.unbind(0))

    @honours_train_time_budget
    def train(self) -> None:  # type: ignore[override]
        if self._inference_loaded:
            msg = "load_policy restores a policy for inference; resume training with load_model."
            raise RuntimeError(msg)
        wm = self.config.wm
        t = self.task
        self.ensure_replay()
        total_frames = int(self.config.collector.total_frames)
        self.log.info("WMBPTT task: %s", t.describe())
        obs, _ = self._isaac.reset()
        self._frames_processed = self._resume_frame_offset
        metrics: dict[str, list] = defaultdict(list)
        stats: dict[str, list] = defaultdict(list)
        meta = TrainingMetadata(total_iterations=0)
        started = False
        last_log = time.monotonic()
        iters_since_log = 0
        writer = self._checkpoint_writer = CheckpointWriter() if wm.ckpt_light else None
        self._last_full_seconds = self.train_elapsed_seconds()
        try:
            if wm.eprobe_share > 0:
                self._start_eprobe()
            while True:
                if self._eprobe is not None:
                    # NDM Gym's ``probe.tick(it, ...)``: before this iteration's collection.
                    self._eprobe.tick(self._iteration, self._deployed_payload)
                self._step_acc += wm.env_steps_per_iter
                n_steps = int(self._step_acc)
                self._step_acc -= n_steps
                if not started:
                    n_steps = max(n_steps, wm.min_replay_steps)
                frames_before = self._frames_processed
                phase: dict[str, float] = {}
                with self._profile_iteration_phase(phase, "collect"):
                    for _ in range(n_steps):
                        obs = self._collect_step(obs, stats)
                if not started:
                    self._start_train_clock()
                    started = True
                self._set_lrs()
                refresh = getattr(t, "refresh", None)
                if callable(refresh):
                    refresh()  # e.g. curriculum-moved termination thresholds
                if (
                    self.replay.steps - self.replay.oldest
                    > max(wm.wm_window, t.history) + 2
                ):
                    with self._profile_iteration_phase(phase, "world_model"):
                        self._run_update(
                            "world_model", self._update_world_model, metrics
                        )
                    if self._iteration >= wm.wm_warmup_iters:
                        with self._profile_iteration_phase(phase, "actor_critic"):
                            self._run_update(
                                "actor_critic", self._update_actor_critic, metrics
                            )
                for name, sec in phase.items():
                    metrics[f"perf/{name}_s"].append(torch.tensor(sec))
                self._iteration += 1
                iters_since_log += 1
                meta.frames_processed = self._frames_processed
                stop = (
                    self._train_time_budget_reached()
                    or (total_frames > 0 and self._frames_processed >= total_frames)
                    or self._calibration_step(self._frames_processed)
                )
                if self._iteration % wm.log_interval_iters == 0 or stop:
                    now = time.monotonic()
                    flushed = self._flush(
                        metrics, stats, (now - last_log) / max(1, iters_since_log)
                    )
                    last_log, iters_since_log = now, 0
                    self.log_training_metrics(
                        flushed,
                        frames=self._frames_processed,
                        iteration=self._iteration,
                    )
                    self.log.info(
                        "it=%d frames=%d t=%.0fs wm=%.4f k8_root=%.1fmm k8_joint=%.2fdeg obj=%.3f alive_H=%.3f env_r=%.4f it_s=%.3f",
                        self._iteration,
                        self._frames_processed,
                        self.train_elapsed_seconds(),
                        flushed.get("wm/loss", float("nan")),
                        flushed.get("wm_eval/k8_root_pos_mm", float("nan")),
                        flushed.get("wm_eval/k8_joint_pos_deg", float("nan")),
                        flushed.get("actor/objective", float("nan")),
                        flushed.get("imagine/alive_H", float("nan")),
                        flushed.get("env/env_reward", float("nan")),
                        flushed.get("perf/iteration_s", float("nan")),
                    )
                    if "real/pelvis_dev_mm" in flushed:
                        self.log.info(
                            "real windows: pelvis_dev=%.1fmm action_dev0=%.4f alive_H=%.3f reward=%.4f",
                            flushed["real/pelvis_dev_mm"],
                            flushed.get("real/action_dev0", float("nan")),
                            flushed.get("real/alive_H", float("nan")),
                            flushed.get("real/reward_mean", float("nan")),
                        )
                    if "perf/collect_s" in flushed:
                        self.log.info(
                            "phases: collect=%.3fs world_model=%.3fs actor_critic=%.3fs",
                            flushed.get("perf/collect_s", float("nan")),
                            flushed.get("perf/world_model_s", float("nan")),
                            flushed.get("perf/actor_critic_s", float("nan")),
                        )
                if self._should_save_checkpoint(
                    frames_processed=self._frames_processed,
                    frames_in_iteration=self._frames_processed - frames_before,
                ):
                    if writer is None:
                        self.save_model(
                            self._checkpoint_dir(), step=self._frames_processed
                        )
                    elif (
                        not stop
                    ):  # the end's full file holds these frames (NDM: ``not final_ck``)
                        self._write_light_checkpoint(writer)
                if stop:
                    meta.stop_after_iteration = True
                    break
            self._save_after_writer(writer, lambda: self._save_end(meta))
        except KeyboardInterrupt:
            self._save_after_writer(
                writer,
                lambda: self.save_model(
                    self._checkpoint_dir(), step=self._frames_processed
                ),
            )
            raise
        finally:
            self._checkpoint_writer = None
            if writer is not None:
                writer.close()
            if self._eprobe is not None:
                self._eprobe.close()
                self._eprobe = None

    def _start_eprobe(self) -> None:
        """``wm.eprobe_share``: the env's probe, with a resumed run's probe state."""
        self._eprobe = self._isaac.make_eval_probe(self.config.wm, self.log_dir)
        if self._eprobe_resume is not None:
            self._eprobe.load_state_dict(self._eprobe_resume)
            self._eprobe_resume = None

    def _deployed_payload(self) -> dict[str, Any]:
        """What a probe sweep runs: the checkpoint's policy entries, copied to the CPU."""
        return cpu_snapshot(
            {
                "policy_state_dict": self.collector_policy.state_dict(),
                "policy_meta": self.policy_meta(),
            }
        )

    def _save_end(self, meta: TrainingMetadata) -> None:
        if self._train_time_budget_reached():
            self._finish_train_time_budget(meta)
        else:
            self.save_model(self._checkpoint_dir(), step=self._frames_processed)

    @staticmethod
    def _save_after_writer(
        writer: CheckpointWriter | None, save: Callable[[], None]
    ) -> None:
        """``save`` on the main thread once ``wm.ckpt_light``'s writer is idle.

        Both can write one ``model_step_<frames>.pt``, and two writes through one
        temporary name leave a crash and a corrupt file. A failed background
        write still raises, after ``save``.
        """
        try:
            if writer is not None:
                writer.close()
        finally:
            save()

    # ------------------------------------------------------------ checkpoint
    def policy_meta(self) -> dict[str, Any]:
        t = self.task
        # Only a residual policy records it: the default metadata stays the baseline's.
        residual = {"actor_residual": True} if self._residual else {}
        return {
            "arch": ARCH_TAG,
            "in_keys": [list(k) for k in t.policy_in_keys],
            "policy_obs_dim": t.policy_obs_dim,
            "action_dim": t.action_dim,
            **self._actor_meta,
            "task_model": str(self.config.wm.task_model),
            "sensor_noise": bool(self.config.wm.sensor_noise),
            "bf16": bool(self.config.wm.bf16),
            "goal_binding": _plain(t.goal_binding())
            if hasattr(t, "goal_binding")
            else None,
            **residual,
        }

    def _checkpoint_payload(self, frames: int) -> dict[str, Any]:
        payload = {
            "policy_state_dict": self.collector_policy.state_dict(),
            "policy_meta": self.policy_meta(),
            "dynamics_state_dict": self.dynamics.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "optimizers": {
                "wm": self.wm_optim.state_dict(),
                "actor": self.actor_optim.state_dict(),
                "critic": self.critic_optim.state_dict(),
            },
            "wm_config": _plain(dataclasses.asdict(self.config.wm)),
            "task": _plain(self.task.describe()),
            "iteration": int(self._iteration),
            "cumulative_env_frames": frames,
            "cumulative_train_seconds": self.train_elapsed_seconds(),
        }
        if self._rebar is not None:
            payload["rebar"] = self._rebar.state_dict()
        if self.actor_avg is not None:
            payload["policy_live_state_dict"] = self.policy.state_dict()
            payload["actor_ema_updates"] = self.actor_avg.updates
        payload.update(self._actor_lr.state_dict())
        if run_metadata := getattr(self, "_run_metadata", None):
            payload["imitationlab"] = run_metadata
        if self._eprobe is not None:
            payload["eprobe"] = (
                self._eprobe.state_dict()
            )  # NDM Gym's eprobe_fail, eprobe_focus, eprobe_focus_share
        return payload

    def save_model(
        self, path: str | Path | None = None, step: int | None = None
    ) -> None:
        base = Path(path) if path else self._checkpoint_dir()
        frames = int(step if step is not None else self._frames_processed)
        target = base if base.suffix == ".pt" else base / f"model_step_{frames}.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        if self._checkpoint_writer is not None:
            # The run's last periodic checkpoint can be a background write of this file.
            self._checkpoint_writer.wait()
        atomic_save(self._checkpoint_payload(frames), target)

    def _write_light_checkpoint(self, writer: CheckpointWriter) -> None:
        """``wm.ckpt_light``: the periodic checkpoint. The copy to the CPU blocks
        the loop; the write to disk does not. ``policy_step_<frames>.pt`` is not
        a name a resume looks for, and ``load_model`` refuses the file anyway."""
        frames, directory = self._frames_processed, self._checkpoint_dir()
        payload = self._checkpoint_payload(frames)
        light = {key: payload[key] for key in LIGHT_CHECKPOINT_KEYS}
        if "imitationlab" in payload:
            light["imitationlab"] = payload["imitationlab"]
        writer.submit(cpu_snapshot(light), directory / f"policy_step_{frames}.pt")
        elapsed = self.train_elapsed_seconds()
        if full_checkpoint_due(
            elapsed, self._last_full_seconds, self.config.wm.last_every_min
        ):
            writer.submit(cpu_snapshot(payload), directory / f"model_step_{frames}.pt")
            self._last_full_seconds = elapsed

    def save(self, path: str | Path) -> None:
        self.save_model(path)

    def _read_checkpoint(
        self, path: str, map_location: torch.device | str, *, inference: bool = False
    ) -> dict[str, Any]:
        """A checkpoint whose policy interface matches this agent, and for a resume
        (not ``inference``) its actor's shape too."""
        data = torch.load(path, map_location=map_location, weights_only=False)
        meta = data.get("policy_meta") or {}
        if meta.get("arch") != ARCH_TAG:
            msg = f"{path} is not a {ARCH_TAG} checkpoint (arch={meta.get('arch')!r})."
            raise ValueError(msg)
        mine = self.policy_meta()
        keys = ["in_keys", "policy_obs_dim", "action_dim"]
        # Checkpoints written before the activation field were ELU.
        meta.setdefault("actor_activation", "elu")
        if not inference:  # an inference load builds the checkpoint's actor
            theirs, ours = actor_shape(meta), actor_shape(mine)
            for key in ACTOR_SHAPE_KEYS:
                if theirs.get(key) != ours.get(key):
                    msg = f"checkpoint policy_meta[{key!r}]={theirs.get(key)!r} != agent {ours.get(key)!r}"
                    raise ValueError(msg)
        # The goal input is bound by what produces it (e.g. the frozen skill
        # encoder's weights), not by its width; older checkpoints predate it.
        if meta.get("goal_binding") is not None:
            keys.append("goal_binding")
        for key in keys:
            if meta.get(key) != mine[key]:
                msg = f"checkpoint policy_meta[{key!r}]={meta.get(key)!r} != agent {mine[key]!r}"
                raise ValueError(msg)
        return data

    def load_policy(self, path: str) -> None:
        """Inference load: ``policy_state_dict`` and ``policy_meta``, nothing else.

        The checkpoint's world model, critic and optimizers do not matter here,
        so a run with other sizes or optimizers scores on an agent built from
        the task's default config. How the policy acts comes from the
        checkpoint, not from that default config. An agent loaded this way
        refuses to train: use ``load_model`` to resume.
        """
        data = self._read_checkpoint(path, "cpu", inference=True)
        meta = data["policy_meta"]
        assert self.policy is not None
        shape = actor_shape(meta)
        if shape != actor_shape(self.policy_meta()):
            self._build_inference_actor(shape)
        self.policy.load_state_dict(data["policy_state_dict"], strict=True)
        # Checkpoints written before ``policy_meta["bf16"]`` keep it in ``wm_config``.
        self.policy_net.bf16 = bool(
            meta.get(
                "bf16", (data.get("wm_config") or {}).get("bf16", self.policy_net.bf16)
            )
        )
        if "sensor_noise" in meta and hasattr(self.task, "sensor_noise"):
            self.task.sensor_noise = bool(meta["sensor_noise"])
        # Checkpoints written before ``policy_meta["actor_residual"]`` output the action itself.
        self._set_residual(bool(meta.get("actor_residual", False)))
        if self.actor_avg is not None:
            self.actor_avg.updates = 0  # ``self.policy`` now holds what is deployed
        self._inference_loaded = True

    def _build_inference_actor(self, shape: dict[str, Any]) -> None:
        """``load_policy``: the checkpoint's actor shape in place of this agent's. The
        agent's own average goes: the loaded ``policy_state_dict`` is what deploys."""
        goal, task_goal = (
            shape.get("actor_goal_dim"),
            getattr(self.task, "policy_goal_dim", None),
        )
        if goal is not None and task_goal is not None and goal != task_goal:
            msg = f"checkpoint policy_meta['actor_goal_dim']={goal} != the task's goal width {task_goal}"
            raise ValueError(msg)
        self.actor = self._build_actor(shape)
        self.policy_net.actor = self.actor
        self.actor_avg = None
        self.policy_avg = None
        self._actor_meta = {
            k: v for k, v in shape.items() if (k, v) != ("actor_arch", "flat")
        }

    def load_model(self, path: str) -> None:
        """Resume load: every component, strictly. A checkpoint from another
        configuration or without the training state raises (see ``load_policy``)."""
        data = self._read_checkpoint(path, self.device)
        wm = self.config.wm
        required = ["dynamics_state_dict", "critic_state_dict", "optimizers"]
        if wm.actor_ema > 0:
            required += ["policy_live_state_dict", "actor_ema_updates"]
        missing = [key for key in required if data.get(key) is None]
        if missing:
            msg = f"{path} has no {missing}: it cannot resume training. Use load_policy to score it."
            raise ValueError(msg)
        if wm.eprobe_share > 0 and data.get("eprobe") is None:
            msg = f"{path} holds no probe state (eprobe): it cannot resume wm.eprobe_share; resume it with the option off."
            raise ValueError(msg)
        saved = data.get("wm_config") or {}
        for name, before_the_field in RESUME_MATCH.items():
            if saved.get(name, before_the_field) != getattr(wm, name):
                msg = f"wm.{name}={getattr(wm, name)!r} must match the checkpoint's {saved.get(name, before_the_field)!r} to resume."
                raise ValueError(msg)
        assert self.policy is not None
        if self.actor_avg is not None:
            self.policy.load_state_dict(data["policy_live_state_dict"], strict=True)
            self.policy_avg.load_state_dict(data["policy_state_dict"], strict=True)  # type: ignore[union-attr]
            self.actor_avg.updates = int(data["actor_ema_updates"])
        else:
            self.policy.load_state_dict(data["policy_state_dict"], strict=True)
        self.dynamics.load_state_dict(data["dynamics_state_dict"])
        self.critic.load_state_dict(data["critic_state_dict"])
        for name, opt in (
            ("wm", self.wm_optim),
            ("actor", self.actor_optim),
            ("critic", self.critic_optim),
        ):
            opt.load_state_dict(data["optimizers"][name])
        self._conform_loaded_optimizers()
        if self._graphs is not None:
            # The loaded optimizer state is new tensors, which a captured graph does not
            # read: new graphs, on a new pool, warm up and capture again.
            self._build_update_graphs()
        if self._rebar is not None and "rebar" in data:
            self._rebar.load_state_dict(data["rebar"])
        self._actor_lr.load_state_dict(data)
        if wm.eprobe_share > 0:
            self._eprobe_resume = data["eprobe"]
            if self._eprobe is not None:  # a load during training
                self._eprobe.load_state_dict(self._eprobe_resume)
                self._eprobe_resume = None
        self._iteration = int(data.get("iteration", 0))
        frames = data.get("cumulative_env_frames")
        if frames is not None:
            self._resume_frame_offset = int(frames)
        self._restore_train_clock(data)


# Historical task registrations instantiate this configurable learner by name.
WMBPTT = WorldModelLearner
