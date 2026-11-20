"""Runtime services shared by the native TorchRL learners."""

from __future__ import annotations

import logging
import math
import time
from abc import abstractmethod
from collections import deque
from collections.abc import Callable, Iterator, Mapping, MutableMapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import torch
import torch.nn
import torch.optim
from tensordict import TensorDict
from tensordict.base import TensorDictBase
from tensordict.nn import TensorDictModule
from torch.nn.parameter import UninitializedParameter
from torch.optim import lr_scheduler
from torchrl.collectors import Collector
from torchrl.data import (
    ReplayBuffer,
)
from torchrl.envs import TransformedEnv
from torchrl.objectives import group_optimizers
from torchrl.record.loggers.common import Logger

from imitationlab.algorithms.base import BaseAlgorithm as Learner
from imitationlab.algorithms.common.config import AlgorithmConfig
from imitationlab.algorithms.common.config_utils import ObsKey
from imitationlab.algorithms.common.gpu_telemetry import GpuTelemetry
from imitationlab.algorithms.common.logging_utils import (
    ELAPSED_ENTRY,
    FRAMES_ENTRY,
    ROOT_LOGGER_NAME,
    LoggingManager,
    MetricReporter,
)
from imitationlab.algorithms.common.type_aliases import OptimizerClass, SchedulerClass

from imitationlab.runners.checkpointing import Checkpointing
from imitationlab.runners.metrics import IterationData as IterationData
from imitationlab.runners.metrics import TrainingMetadata as TrainingMetadata
from imitationlab.runners.metrics import TrainingMetrics

CfgT = TypeVar("CfgT", bound=AlgorithmConfig)


def honours_train_time_budget(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Mark a ``train`` loop that reads ``trainer.max_train_seconds``.

    A subclass that overrides ``train`` without this marker does not honour
    the budget, and ``AlgorithmRuntime.supports_train_time_budget`` is False.
    """
    fn._honours_train_time_budget = True  # type: ignore[attr-defined]
    return fn


class AlgorithmRuntime(Checkpointing, TrainingMetrics, Learner[CfgT]):
    """TorchRL lifecycle, checkpointing, metrics, and reusable collection hooks.

    Concrete algorithms own their objectives and training loop. The runtime
    initializes their components, collects rollouts where applicable, and
    provides consistent persistence and logging.
    """

    # ------------------------------------------------------- walltime budget
    # Host-side clock only (time.monotonic): one call per iteration, no device
    # synchronisation. ``_train_seconds_offset`` is the cumulative training
    # time restored from a checkpoint (``cumulative_train_seconds``).
    _train_clock_start: float | None = None
    _train_seconds_offset: float = 0.0
    _calibration_clock = None

    def _calibration_step(self, frames: int) -> bool:
        """Optional synchronized measurement; ordinary training takes no timing path."""
        if self._calibration_clock is None:
            import json
            import os

            settings = os.environ.get("IMITATIONLAB_CALIBRATION")
            if not settings:
                self._calibration_clock = False
                return False
            from imitationlab.runners.calibration import CalibrationClock

            self._calibration_clock = CalibrationClock(**json.loads(settings))
        if self._calibration_clock is False:
            return False
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return self._calibration_clock.step(frames)

    @property
    def supports_train_time_budget(self) -> bool:
        """Whether this agent's ``train`` loop honours ``trainer.max_train_seconds``."""
        return bool(getattr(type(self).train, "_honours_train_time_budget", False))

    def _max_train_seconds(self) -> float | None:
        trainer = getattr(self.config, "trainer", None)
        value = getattr(trainer, "max_train_seconds", None)
        if value is None or float(value) <= 0.0:
            return None
        return float(value)

    def _start_train_clock(self) -> None:
        """Start the training clock once (call after the first collected batch)."""
        if self._train_clock_start is None:
            self._train_clock_start = time.monotonic()
            if getattr(self.config.trainer, "checkpoint_seconds", ()):
                import json

                (self.log_dir / "training_clock.json").write_text(
                    json.dumps(
                        {
                            "monotonic_start": self._train_clock_start,
                            "unix_start": time.time(),
                            "restored_train_seconds": self._train_seconds_offset,
                        }
                    )
                    + "\n"
                )
            budget = self._max_train_seconds()
            if budget is not None:
                self.log.info(
                    "Training clock started: budget %.1f s (%.1f s already used "
                    "by earlier segments).",
                    budget,
                    self._train_seconds_offset,
                )

    def train_elapsed_seconds(self) -> float:
        """Cumulative training seconds: earlier segments plus this one."""
        start = self._train_clock_start
        own = 0.0 if start is None else time.monotonic() - start
        return float(self._train_seconds_offset) + own

    def _train_time_budget_reached(self) -> bool:
        budget = self._max_train_seconds()
        if budget is None or self._train_clock_start is None:
            return False
        return self.train_elapsed_seconds() >= budget

    def _train_clock_metrics(self, frames_processed: int) -> dict[str, float]:
        """Time-axis metrics logged with every flush (host values only)."""
        metrics = {
            ELAPSED_ENTRY: self.train_elapsed_seconds(),
            FRAMES_ENTRY: float(frames_processed),
        }
        if self.device.type == "cuda":
            # Host-side allocator statistic: no device synchronisation.
            metrics["perf/cuda_max_memory_allocated_gib"] = (
                torch.cuda.max_memory_allocated(self.device) / 2**30
            )
            trainer = getattr(self.config, "trainer", None)
            if bool(getattr(trainer, "gpu_telemetry", False)):
                # NVML host query: the clocks a speed number ran at.
                telemetry = getattr(self, "_gpu_telemetry", None)
                if telemetry is None:
                    telemetry = self._gpu_telemetry = GpuTelemetry(self.device)
                metrics.update(telemetry.read())
        return metrics

    def _restore_train_clock(self, checkpoint: Mapping[str, Any]) -> None:
        seconds = checkpoint.get("cumulative_train_seconds")
        if seconds is not None:
            self._train_seconds_offset = float(seconds)

    def _final_checkpoint_path(self, frames: int) -> Path:
        return (
            self.log_dir / self.config.logger.save_path / f"model_step_{int(frames)}.pt"
        )

    def _write_budget_checkpoint(self, frames: int) -> None:
        """Write the final checkpoint unless one exists for this frame count."""
        target = self._final_checkpoint_path(frames)
        if target.exists():
            return
        custom_save = getattr(self, "save", None)
        if callable(custom_save):
            # Agents with a full-state ``save`` (the PPO record() path uses it too).
            target.parent.mkdir(parents=True, exist_ok=True)
            custom_save(target)
        else:
            self.save_model(path=target.parent, step=int(frames))

    def _finish_train_time_budget(self, metadata: TrainingMetadata) -> None:
        """Log the stop and write the final checkpoint named by cumulative frames."""
        frames = int(metadata.frames_processed)
        if self._calibration_clock and self._calibration_clock.done:
            self.log.info("Calibration measurement completed at %d frames.", frames)
            self._write_budget_checkpoint(frames)
            return
        elapsed = self.train_elapsed_seconds()
        self.log.info(
            "Training time budget reached: %.1f s >= %.1f s at %d cumulative env "
            "frames; stopping and writing the final checkpoint.",
            elapsed,
            float(self._max_train_seconds() or 0.0),
            frames,
        )
        self._write_budget_checkpoint(frames)
        metrics = self._train_clock_metrics(frames)
        metrics["train/stopped_by_time_budget"] = 1.0
        self.log_metrics(metrics, step=frames, log_python=False)

    def _iteration_profiling_enabled(self) -> bool:
        """Whether synchronized per-iteration phase profiling is enabled."""
        trainer = getattr(self.config, "trainer", None)
        return bool(trainer and getattr(trainer, "profile_iterations", False))

    def _synchronize_profile_device(self) -> None:
        """Drain queued accelerator work at an opt-in profiling boundary."""
        if not self._iteration_profiling_enabled() or not torch.cuda.is_available():
            return
        device = torch.device(self.device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    @contextmanager
    def _profile_iteration_phase(
        self,
        phase_times: MutableMapping[str, float],
        name: str,
    ) -> Iterator[None]:
        """Accumulate synchronized wall time for one named iteration phase."""
        if not self._iteration_profiling_enabled():
            yield
            return

        self._synchronize_profile_device()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._synchronize_profile_device()
            elapsed = time.perf_counter() - start
            phase_times[name] = phase_times.get(name, 0.0) + elapsed

    def _finish_iteration_profile(
        self,
        metadata: TrainingMetadata,
        iteration: IterationData,
    ) -> None:
        """Emit scalar and text summaries after all iteration phases complete."""
        if not self._iteration_profiling_enabled():
            return

        top_level_names = ("collect", "prepare", "learn", "record")
        total = sum(iteration.phase_times.get(name, 0.0) for name in top_level_names)
        metrics = {
            f"profile/{name}_s": value for name, value in iteration.phase_times.items()
        }
        metrics["profile/total_s"] = total
        metrics["profile/frames_per_second"] = (
            float(iteration.frames) / total if total > 0.0 else 0.0
        )
        self.log_metrics(
            metrics,
            step=metadata.frames_processed,
            log_python=False,
        )

        top_level = " ".join(
            f"{name}={iteration.phase_times.get(name, 0.0) * 1_000.0:.2f}ms"
            for name in top_level_names
        )
        details = " ".join(
            f"{name.split('/', maxsplit=1)[-1]}={elapsed * 1_000.0:.2f}ms"
            for name, elapsed in iteration.phase_times.items()
            if "/" in name
        )
        message = (
            "iteration_profile | "
            f"iter={iteration.iteration_idx + 1}/{metadata.total_iterations} | "
            f"{top_level} total={total * 1_000.0:.2f}ms | "
            f"fps={metrics['profile/frames_per_second']:.1f}"
        )
        if details:
            message = f"{message} | detail: {details}"
        self.log.info(message)

    def __init__(
        self,
        env: TransformedEnv,
        config: CfgT,
        logger: Logger | None = None,
        **kwargs,
    ):
        """Initialize the base algorithm with environment, configuration, and logging.

        Sets up all core components including networks, data collection, optimization,
        and monitoring infrastructure. Concrete imitationlab.algorithms should not override this
        unless absolutely necessary; use the abstract construction methods instead.

        Args:
            env: Configured TorchRL environment with appropriate transforms.
            config: Algorithm configuration with hyperparameters and settings.
            logger: Optional external logger for metrics tracking.
            **kwargs: Additional algorithm-specific parameters.
        """
        super().__init__()
        self.env = env
        self.config: CfgT = config
        self.kwargs = kwargs

        # Keep Python logger and TorchRL logger state on the instance for reuse
        self.log: logging.Logger = logging.getLogger(
            f"{ROOT_LOGGER_NAME}.{self.__class__.__name__}"
        )
        self.logger: Logger | None = None
        self.metrics: MetricReporter | None = None
        self._logging_manager: LoggingManager | None = None
        self.logger_video = False

        self._configure_logger(logger_override=logger)
        self.log_dir: Path = self._logging_manager.run_dir  # type: ignore[union-attr]
        self.log.debug(
            "Initialized %s (metrics_backend=%s, level=%s)",
            self.__class__.__name__,
            self.config.logger.backend or "none",
            logging.getLevelName(self.log.level),
        )

        # Seed for reproducibility
        self.seed(config.seed)

        self.mp_context = "fork"

        # Construct or attach networks based on existence in config
        self.lr_scheduler = None
        self.lr_scheduler_step = "update"

        # Initialize component placeholders
        self.policy: TensorDictModule | None = None
        self.value_function: TensorDictModule | None = None
        self.q_function: TensorDictModule | None = None
        self.feature_extractor: TensorDictModule | None = None

        # construct networks based on config
        self.policy = self._construct_policy()
        if config.feature_extractor:
            self.feature_extractor = self._construct_feature_extractor()
        if config.value_function:
            self.value_function = self._construct_value_function()
        if config.q_function:
            self.q_function = self._construct_q_function()

        # Use actor_critic to encapsulate policy and value networks for easy management
        self.actor_critic = self._construct_actor_critic()

        # buffer
        self.data_buffer = self._construct_data_buffer()

        self.step_count = 0

        self.start_time = time.time()

        # build collector, collector_policy can be customized
        self.collector = self._construct_collector(self.env, self.collector_policy)

        # build loss module
        self.loss_module = self._construct_loss_module()

        # optimizers
        self.optim = self._configure_optimizers()

        # episode length
        self.episode_lengths = deque(maxlen=100)

        # episode rewards
        self.episode_rewards = deque(maxlen=100)

        # Cache parameters for fast NaN checks during training
        self._parameter_monitor: list[tuple[str, torch.nn.Parameter]] = []
        self._refresh_parameter_monitor()
        self._update_stage_context: str = ""

        # Print model overview once components are initialized
        try:
            self._print_model_overview()
        except (AttributeError, RuntimeError, ValueError) as exc:
            # The overview is informational; a module whose repr fails must
            # not stop training.
            self.log.debug("Skipped the model overview: %s", exc, exc_info=True)

        self._compile_components()

    def seed(self, seed: int) -> None:
        """Set random seeds for reproducibility across all random number generators.

        Configures seeds for PyTorch (CPU and CUDA), NumPy, and creates dedicated
        RNG instances for controlled stochasticity in the algorithm.

        Args:
            seed: Integer seed value for all random number generators.
        """
        torch.manual_seed(seed)
        self.np_rng = np.random.default_rng(seed)
        self.th_rng = torch.Generator(device=self.device)
        self.th_rng.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @property
    def collector_policy(self) -> TensorDictModule:
        """Return the policy used for data collection during training.

        By default, returns the policy operator from the actor-critic module.
        Override in subclasses if a different exploration policy is needed
        (e.g., with added noise for off-policy imitationlab.algorithms).

        Returns:
            TensorDictModule that maps observations to actions for collection.
        """
        return self.actor_critic.get_policy_operator()

    @property
    def policy_output_shape(self) -> int:
        """Get the dimensionality of the action space.

        Returns:
            Integer representing the action space dimension.
        """
        return int(self.env.action_spec_unbatched.shape[-1])  # type: ignore

    @property
    def value_output_shape(self) -> int:
        """Get the output dimension for value function networks.

        Returns:
            1 (scalar value function output).
        """
        return 1

    def observation_feature_shape(self, key: ObsKey) -> tuple[int, ...]:
        """Return the unbatched feature shape registered for one observation key.

        TorchRL specs may include the environment batch prefix. This strips that
        prefix so downstream code can reason about the per-sample feature layout.
        """
        shape = tuple(int(dim) for dim in self.env.observation_spec[key].shape)
        batch_prefix = tuple(int(dim) for dim in self.env.batch_size)
        if batch_prefix and shape[: len(batch_prefix)] == batch_prefix:
            return shape[len(batch_prefix) :]
        return shape

    def observation_feature_rank(self, key: ObsKey) -> int:
        """Return how many trailing dimensions belong to one observation sample."""
        return len(self.observation_feature_shape(key))

    def observation_feature_size(self, key: ObsKey) -> int:
        """Return the flattened feature size for one observation sample."""
        shape = self.observation_feature_shape(key)
        return int(math.prod(shape)) if shape else 1

    def action_feature_shape(self) -> tuple[int, ...]:
        """Return the unbatched action shape used by the algorithm."""
        return tuple(int(dim) for dim in self.env.action_spec_unbatched.shape)  # type: ignore[attr-defined]

    def action_feature_rank(self) -> int:
        """Return how many trailing dimensions belong to one action sample."""
        return len(self.action_feature_shape())

    def action_feature_size(self) -> int:
        """Return the flattened action size for one action sample."""
        shape = self.action_feature_shape()
        return int(math.prod(shape)) if shape else 1

    @property
    def device(self) -> torch.device:
        """Return the PyTorch device used for training computations.

        Returns:
            torch.device instance (CPU or CUDA based on config).
        """
        return self._get_device(self.config.device)

    def _get_device(self, device_str: str) -> torch.device:
        """Resolve device string to a concrete PyTorch device.

        Args:
            device_str: Device specification ("auto", "cpu", "cuda", "cuda:0", etc.).
                "auto" selects CUDA if available, otherwise CPU.

        Returns:
            torch.device instance corresponding to the specification.
        """
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_str)

    def _initialize_weights(
        self, module: torch.nn.Module, init_type: str | None
    ) -> None:
        """Initialize neural network weights using specified initialization scheme.

        Applies weight initialization to all Linear layers in the module recursively.
        Biases are always initialized to zero.

        Args:
            module: PyTorch module whose weights should be initialized.
            init_type: Initialization method. Supported values:
                - "orthogonal": Orthogonal initialization with gain=1.0
                - "xavier_uniform": Xavier/Glorot uniform initialization
                - "kaiming_uniform": He uniform initialization
                - None: Skip initialization (use PyTorch defaults)
        """
        if init_type is None:
            return  # No initialization specified, skip
        for layer in module.modules():
            if isinstance(layer, torch.nn.Linear):
                if init_type == "orthogonal":
                    torch.nn.init.orthogonal_(layer.weight, 1.0)
                elif init_type == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(layer.weight)
                elif init_type == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(layer.weight)
                # bias exists for Linear by default
                layer.bias.data.zero_()

    def _construct_collector(
        self, env: TransformedEnv, policy: TensorDictModule
    ) -> Collector:
        """Create a synchronized data collector for environment interaction.

        Configures a TorchRL Collector with appropriate settings for
        parallel data collection, initialization, and device placement.

        Args:
            env: Environment instance to collect data from.
            policy: Policy module for action selection during collection.

        Returns:
            Configured Collector instance ready for use.

        Note:
            Uses fork context by default. Compilation is enabled if config.compile
            is set, with warmup period for JIT optimization.
        """
        # We can't use nested child processes with mp_start_method="fork"

        frames_per_batch = int(self.config.collector.frames_per_batch)
        total_frames = int(self.config.collector.total_frames)
        aligned_total_frames = total_frames
        if frames_per_batch > 0 and total_frames % frames_per_batch != 0:
            aligned_total_frames = max(
                frames_per_batch,
                (total_frames // frames_per_batch) * frames_per_batch,
            )
            self.log.warning(
                "collector.total_frames (%d) is not divisible by frames_per_batch (%d); "
                "using %d to avoid over-collection warnings.",
                total_frames,
                frames_per_batch,
                aligned_total_frames,
            )

        collector = Collector(
            env,
            policy=policy,
            init_random_frames=self.config.collector.init_random_frames,
            frames_per_batch=frames_per_batch,
            total_frames=aligned_total_frames,
            # this is the default behavior: the collector runs in ``"random"`` (or explorative) mode
            # exploration_type=ExplorationType.RANDOM,
            # We set the all the devices to be identical. Below is an example of
            compile_policy=(
                {
                    "mode": self.config.compile.compile_mode,
                    "warmup": int(getattr(self.config.compile, "warmup", 1)),
                }
                if self.config.compile.compile
                else False
            ),
            cudagraph_policy=(
                {"warmup": 10}
                if self.config.compile.cudagraphs and self.device.type == "cuda"
                else False
            ),
            # heterogeneous devices
            device=self.device,
            # storing_device=self.device,
            # reset_at_each_iter=False,
            # set_truncated=self.config.collector.set_truncated,
            no_cuda_sync=self.config.collector.no_cuda_sync,
        )
        collector.set_seed(self.config.seed)
        return collector

    @abstractmethod
    def _construct_policy(self) -> TensorDictModule:
        """Construct the policy network (actor) from configuration.

        Implement this method in subclasses to build the policy network architecture.
        The network should map observations to action distributions or deterministic actions.

        Args:
            policy_net: Optional pre-constructed network. If None, build from config.

        Returns:
            TensorDictModule wrapping the policy network with appropriate input/output keys.

        Note:
            For stochastic policies, typically wraps the network with a distribution
            (e.g., TanhNormal, Normal) via ProbabilisticActor.
        """

    def _construct_value_function(self) -> TensorDictModule:
        """Construct the value function network (critic) from configuration.

        Implement this method in subclasses that use value-based methods (e.g., PPO, A2C).

        Args:
            value_net: Optional pre-constructed network. If None, build from config.

        Returns:
            TensorDictModule wrapping the value network.

        Raises:
            NotImplementedError: If the algorithm doesn't use a value function.
        """
        msg = "Subclasses must implement this method"
        raise NotImplementedError(msg)

    def _construct_q_function(self) -> TensorDictModule:
        """Construct the Q-function network from configuration.

        Implement this method in subclasses that use Q-learning (off-policy learners).

        Args:
            q_net: Optional pre-constructed network. If None, build from config.

        Returns:
            TensorDictModule wrapping the Q-network.

        Raises:
            NotImplementedError: If the algorithm doesn't use Q-functions.
        """
        msg = "Subclasses must implement this method"
        raise NotImplementedError(msg)

    @abstractmethod
    def _construct_feature_extractor(self) -> TensorDictModule:
        """Construct a shared feature extraction network from configuration.

        Implement this method when using shared representations between policy and value networks.

        Args:
            feature_extractor_net: Optional pre-constructed network. If None, build from config.

        Returns:
            TensorDictModule for shared feature extraction.

        Note:
            Common for vision-based tasks where convolutional layers are shared.
        """

    @abstractmethod
    def _construct_actor_critic(self) -> TensorDictModule:
        """Construct combined actor-critic module for unified forward passes.

        Implement this method to combine policy and value networks into a single module.
        This enables efficient computation when both are needed simultaneously.

        Returns:
            TensorDictModule combining actor and critic components.

        Example:
            For PPO, this typically returns an ActorValueOperator wrapping
            the policy and value function.
        """

    @abstractmethod
    def _construct_loss_module(self) -> torch.nn.Module:
        """Construct the loss module for computing training objectives.

        Implement this method to create the algorithm-specific loss module
        (e.g., ClipPPOLoss).

        Returns:
            TorchRL loss module configured for the algorithm.

        Note:
            The loss module should accept TensorDict inputs and return
            a TensorDict with loss components.
        """

    @abstractmethod
    def _construct_data_buffer(self) -> ReplayBuffer:
        """Construct the replay buffer for storing and sampling experience.

        Implement this method to configure the buffer based on algorithm needs
        (on-policy vs off-policy, prioritization, etc.).

        Returns:
            Configured ReplayBuffer instance.

        Example:
            On-policy imitationlab.algorithms typically use simple TensorDictReplayBuffer,
            while off-policy methods may use larger buffers with prioritization.
        """

    def _compile_components(self) -> None:
        """Compile performance-critical methods using torch.compile for acceleration.

        Override this method in subclasses to apply torch.compile to hot paths
        (e.g., action computation, advantage estimation, return calculation).

        Note:
            Only beneficial for imitationlab.algorithms with heavy computational bottlenecks.
            May increase compilation time and memory usage.

        Example:
            >>> if self.config.compile:
            ...     self.compute_action = torch.compile(self._compute_action, dynamic=True)
            ...     self.compute_returns = torch.compile(self._compute_returns, dynamic=True)
            >>> else:
            ...     self.compute_action = self._compute_action
            ...     self.compute_returns = self._compute_returns
        """
        compile_mode = None

        if self.config.compile.compile:
            compile_mode = self.config.compile.compile_mode
            if compile_mode in ("", None):
                if self.config.compile.cudagraphs:
                    compile_mode = "default"
                else:
                    compile_mode = "reduce-overhead"
        self.compile_mode = compile_mode

    def _configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizers and learning rate schedulers for training.

        Creates optimizers for all trainable components (policy, value, Q-networks, etc.)
        based on configuration settings. Supports common optimizers (Adam, SGD, RMSprop)
        and optional learning rate scheduling.

        Returns:
            Grouped optimizer containing all component optimizers.

        Raises:
            ValueError: If unknown optimizer or scheduler is specified in config.

        Note:
            Subclasses can override _set_optimizers() to add algorithm-specific
            optimizers (e.g., for temperature parameters, reward models).
        """
        cfg = self.config.optim

        optimizer_map: dict[str, OptimizerClass] = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
            "adamax": torch.optim.Adamax,
            "sgd": torch.optim.SGD,
            "rmsprop": torch.optim.RMSprop,
        }

        optimizer_name = cfg.optimizer.lower()
        if optimizer_name not in optimizer_map:
            available = ", ".join(sorted(optimizer_map))
            msg = f"Unknown optimizer '{cfg.optimizer}'. Choose one of: {available}."
            raise ValueError(msg)

        optimizer_cls = optimizer_map[optimizer_name]

        base_kwargs = {
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
        }
        optimizer_kwargs = {**base_kwargs, **cfg.optimizer_kwargs}

        # Create optimizers for different components
        optimizers: list[torch.optim.Optimizer] = self._set_optimizers(
            optimizer_cls, optimizer_kwargs
        )

        if not optimizers:
            msg = "No optimizers could be created. Check that the algorithm has the required components."
            raise ValueError(msg)

        optim = group_optimizers(*optimizers)

        # Configure learning rate scheduler
        scheduler_name = (cfg.scheduler or "").lower() or None
        scheduler_map: dict[str, SchedulerClass] = {
            "steplr": lr_scheduler.StepLR,
            "multiplicativelr": lr_scheduler.MultiplicativeLR,
            "exponentiallr": lr_scheduler.ExponentialLR,
            "cosineannealinglr": lr_scheduler.CosineAnnealingLR,
            "cosineannealingwarmrestarts": lr_scheduler.CosineAnnealingWarmRestarts,
            "linearlr": lr_scheduler.LinearLR,
            "polynomiallr": lr_scheduler.PolynomialLR,
        }

        self.lr_scheduler = None
        if scheduler_name and scheduler_name != "adaptive":
            if scheduler_name not in scheduler_map:
                available = ", ".join(sorted(scheduler_map))
                msg = (
                    f"Unknown scheduler '{cfg.scheduler}'. Choose one of: {available}."
                )
                raise ValueError(msg)

            scheduler_cls = scheduler_map[scheduler_name]
            scheduler_kwargs = dict(cfg.scheduler_kwargs)
            self.lr_scheduler = scheduler_cls(optim, **scheduler_kwargs)

        self.lr_scheduler_step = getattr(cfg, "scheduler_step", "update").lower()

        return optim

    def _set_optimizers(
        self, _optimizer_cls: OptimizerClass, _optimizer_kwargs: dict[str, Any]
    ) -> list[torch.optim.Optimizer]:
        """Create optimizers for algorithm-specific trainable components.

        Override this method in subclasses to instantiate optimizers for all
        trainable modules (policy, value, Q-networks, temperature parameters, etc.).

        Args:
            optimizer_cls: Optimizer class to instantiate (e.g., torch.optim.Adam).
            optimizer_kwargs: Keyword arguments for optimizer construction (lr, weight_decay, etc.).

        Returns:
            List of optimizer instances, one per trainable component.

        Example:
            >>> def _set_optimizers(self, optimizer_cls, optimizer_kwargs):
            ...     return [
            ...         optimizer_cls(self.policy.parameters(), **optimizer_kwargs),
            ...         optimizer_cls(self.value_function.parameters(), **optimizer_kwargs)
            ...     ]
        """
        return []

    def _print_model_overview(self) -> None:
        """Print a comprehensive overview of model architecture and components.

        Logs network architectures for all initialized components (policy, value,
        Q-networks) to help with debugging and verification.
        """
        self.log.info("Model Overview:\n%s", str(self))
        self.log.info("Policy Network:\n%s", str(self.policy))

        if self.config.value_function:
            self.log.info("Value Network:\n%s", str(self.value_function))
        if self.config.q_function:
            self.log.info("Q Network:\n%s", str(self.q_function))

    def _refresh_parameter_monitor(self) -> None:
        """Build a registry of all trainable parameters for numerical stability monitoring.

        Collects references to all parameters from core modules (actor_critic, loss_module,
        feature_extractor) while avoiding duplicates. Used by validation methods to
        efficiently check for NaN/Inf values during training.

        Note:
            Should be called after network initialization and when architecture changes.
            Automatically filters out non-floating-point parameters.
        """
        modules_to_check: list[tuple[str, torch.nn.Module | None]] = [
            ("actor_critic", self.actor_critic),
            ("loss_module", self.loss_module),
        ]
        if getattr(self.config, "use_feature_extractor", False):
            modules_to_check.append(("feature_extractor", self.feature_extractor))

        seen: set[int] = set()
        monitored: list[tuple[str, torch.nn.Parameter]] = []
        for module_label, module in modules_to_check:
            if module is None:
                continue
            for name, param in module.named_parameters(recurse=True):
                if param is None or isinstance(param, UninitializedParameter):
                    continue
                if not torch.is_floating_point(param):
                    continue
                param_id = id(param)
                if param_id in seen:
                    continue
                seen.add(param_id)
                monitored.append((f"{module_label}.{name}", param))

        self._parameter_monitor = monitored

    def _validate_parameters(self, stage: str) -> None:
        """Verify that all monitored parameters contain finite values.

        Checks all registered parameters for NaN or Inf values, which typically
        indicate numerical instability, exploding gradients, or improper initialization.

        Args:
            stage: Descriptive label for the current training stage (e.g., "loss_computation",
                "after_optimizer_step") for informative error messages.

        Raises:
            FloatingPointError: If any parameter contains NaN or Inf values, with
                detailed information about the problematic parameter.

        Note:
            Call this at critical points in training (after loss computation, after
            optimizer steps) to catch numerical issues early.
        """
        if not self._parameter_monitor:
            self._refresh_parameter_monitor()

        for name, param in self._parameter_monitor:
            if not torch.isfinite(param).all():
                nan_count = torch.isnan(param).sum().item()
                inf_count = torch.isinf(param).sum().item()
                msg = (
                    "Detected non-finite parameter values "
                    f"during '{stage}' in {name}: nan={nan_count}, inf={inf_count}"
                )
                raise FloatingPointError(msg)

    def _validate_gradients(self, stage: str, raise_error: bool = False) -> bool:
        """Verify that parameter gradients are finite.

        Checks gradients of all monitored parameters for NaN or Inf values,
        which can occur due to vanishing/exploding gradients, numerical issues
        in loss computation, or improper backpropagation.

        Args:
            stage: Descriptive label for the current training stage for error context.
            raise_error: If True, raises exception on detection. If False, logs
                warnings and returns False.

        Returns:
            True if all gradients are finite, False otherwise.

        Raises:
            FloatingPointError: If raise_error=True and non-finite gradients detected.

        Note:
            Typically called after loss.backward() but before optimizer.step() to
            prevent corrupted parameter updates.
        """
        if not self._parameter_monitor:
            self._refresh_parameter_monitor()

        all_finite = True
        for name, param in self._parameter_monitor:
            grad = param.grad
            if grad is None or not torch.is_floating_point(grad):
                continue
            if not torch.isfinite(grad).all():
                nan_count = torch.isnan(grad).sum().item()
                inf_count = torch.isinf(grad).sum().item()
                msg = (
                    "Detected non-finite gradients "
                    f"during '{stage}' in {name}: nan={nan_count}, inf={inf_count}"
                )
                all_finite = False
                if raise_error:
                    raise FloatingPointError(msg)
                self.log.debug("%s", msg)
        return all_finite

    def _validate_tensordict(
        self,
        td: TensorDictBase,
        stage: str,
        prefix: tuple[str, ...] = (),
        raise_error: bool = False,
    ) -> bool:
        """Recursively validate that all floating-point tensors in a TensorDict are finite.

        Performs deep inspection of TensorDict structures to detect NaN or Inf values
        in observations, actions, rewards, or other intermediate computations.

        Args:
            td: TensorDict to validate (may contain nested TensorDicts).
            stage: Training stage label for error context.
            prefix: Tuple of parent keys for nested path construction (internal use).
            raise_error: If True, raises exception on detection. If False, logs and returns False.

        Returns:
            True if all tensors are finite, False otherwise.

        Raises:
            FloatingPointError: If raise_error=True and non-finite values detected.

        Note:
            Useful for validating batches from replay buffer, rollout data, or loss outputs.
        """

        all_finite = True
        for key, value in td.items():  # type: ignore
            key_str = str(key)
            next_prefix = (*prefix, key_str)
            if isinstance(value, TensorDict):
                all_finite &= self._validate_tensordict(
                    value, stage, next_prefix, raise_error
                )
            elif (
                torch.is_tensor(value)
                and torch.is_floating_point(value)  # type: ignore[call-arg]
                and not torch.isfinite(value).all()  # type: ignore
            ):
                nan_count = torch.isnan(value).sum().item()  # type: ignore[call-arg]
                inf_count = torch.isinf(value).sum().item()  # type: ignore[call-arg]
                joined_key = ".".join(next_prefix)
                msg = (
                    "Detected non-finite tensor values "
                    f"during '{stage}' at key '{joined_key}': "
                    f"nan={nan_count}, inf={inf_count}"
                )
                all_finite = False
                if raise_error:
                    raise FloatingPointError(msg)
                self.log.debug("%s", msg)

        return all_finite

    def _sanitize_loss_tensordict(self, loss_td: TensorDict, stage: str) -> TensorDict:
        """Replace non-finite loss values with zeros to prevent gradient corruption.

        Defensive mechanism to handle numerical instabilities in loss computation.
        Replaces NaN/Inf loss terms with zero while logging warnings for debugging.

        Args:
            loss_td: TensorDict containing loss components from loss_module.
            stage: Training stage label for logging context.

        Returns:
            Sanitized TensorDict with non-finite values replaced by zeros.

        Warning:
            This is a fallback for numerical issues. Frequent sanitization indicates
            underlying problems (e.g., unstable hyperparameters, gradient explosion)
            that should be addressed at the source.

        Note:
            Operates in-place on the TensorDict structure.
        """

        for key, value in list(loss_td.items()):
            if isinstance(value, TensorDict):
                self._sanitize_loss_tensordict(value, stage)
                continue

            if torch.is_tensor(value) and torch.is_floating_point(value):  # type: ignore[call-arg]
                if torch.isfinite(value).all():  # type: ignore[call-arg]
                    continue

                nan_count = torch.isnan(value).sum().item()  # type: ignore[call-arg]
                inf_count = torch.isinf(value).sum().item()  # type: ignore[call-arg]
                key_repr = key if isinstance(key, str) else str(key)
                self.log.debug(
                    "Non-finite loss detected; replacing with zeros | stage=%s key=%s nan=%d inf=%d",
                    stage,
                    key_repr,
                    nan_count,
                    inf_count,
                )
                loss_td.set(key, torch.zeros_like(value))  # type: ignore

        return loss_td


# Compatibility for configurations that reference algorithms.base.BaseAlgorithm.
BaseAlgorithm = AlgorithmRuntime
