"""Training rows, backend logging, and progress display for native learners."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import Any

from torchrl.record.loggers.common import Logger

from imitationlab.algorithms.common.logging_utils import (
    ITERATION_ENTRY,
    REQUIRED_TRAINING_ENTRIES,
    THROUGHPUT_ENTRY,
    LoggingManager,
    log_to_file_only,
    missing_training_entries,
)
from imitationlab.algorithms.common.utils import as_float, log_info


@dataclass(kw_only=True)
class TrainingMetadata:
    """Generic metadata shared across the outer training loop.

    The goal is to keep only algorithm-agnostic terms here so every training loop
    can reuse the same vocabulary without inheriting family-specific details such
    as replay sampling, GAE, or discriminator updates.
    """

    # Total number of outer-loop cycles expected for this train() call.
    total_iterations: int
    # Total environment frames consumed so far across the full outer-loop.
    frames_processed: int = 0
    # Optional UI handle used by imitationlab.algorithms that render a live progress bar.
    progress_bar: Any | None = None
    # Whether loop progress is shown via the progress bar instead of periodic logs.
    progress_bar_enabled: bool = False
    # Frame cadence for periodic text logging when no live progress bar is shown.
    log_interval_frames: int = 1000
    # Absolute frame count threshold for the next periodic text log.
    next_log_frame: int = 1000
    # Absolute frame count threshold for the next periodic file-only summary.
    next_file_log_frame: int = 1000
    # Number of table rows already emitted to the file-only summary.
    file_summary_rows: int = 0
    # Set when ``trainer.max_train_seconds`` ran out: the current iteration is
    # the last one (it logs, then the loop stops and writes a final checkpoint).
    stop_after_iteration: bool = False


@dataclass(kw_only=True)
class IterationData:
    """Generic per-iteration data accumulated while processing one outer-loop iteration."""

    # Zero-based index of the current outer-loop iteration.
    iteration_idx: int
    # Number of environment frames represented by this iteration.
    frames: int
    # Time spent collecting fresh environment data for this iteration.
    collect_time: float = 0.0
    # Time spent learning/logging after collection for this iteration.
    learn_time: float = 0.0
    # Opt-in synchronized wall times for the named phases in this iteration.
    phase_times: dict[str, float] = field(default_factory=dict)
    # Scalar metrics accumulated while handling this iteration.
    metrics: dict[str, Any] = field(default_factory=dict)


class TrainingMetrics(ABC):
    """Report metrics and progress on the cumulative environment-frame axis."""

    _FILE_SUMMARY_HEADER_INTERVAL = 20

    # (host time, frames) of the previous training row, for ``time/speed``.
    _last_training_row: tuple[float, int] | None = None
    # Required entries already warned about, so a gap warns once, not per row.
    _reported_missing_entries: frozenset[str] = frozenset()

    def _configure_logger(self, logger_override: Logger | None = None) -> None:
        """Configure Python logging and metrics tracking backend.

        Initializes the logging infrastructure including Python logger, optional
        TorchRL metrics backend (TensorBoard, WandB, etc.), and video recording.

        Args:
            logger_override: Optional external TorchRL Logger to use instead of
                creating one from config. Useful for sharing loggers across components.
        """

        self._logging_manager = LoggingManager(
            config=self.config,
            component=self.__class__.__name__,
            metrics_logger=logger_override,
        )
        self.log = self._logging_manager.logger
        self.logger = self._logging_manager.metrics_logger
        self.metrics = self._logging_manager.metric_reporter
        self.logger_video = self._logging_manager.video_enabled

    def log_metrics(
        self,
        metrics: Mapping[str, Any],
        *,
        step: Any,
        log_python: bool | None = None,
        python_level: int = logging.INFO,
    ) -> None:
        """Log scalar metrics to configured backends (TensorBoard, WandB, console).

        Sends metrics to the configured TorchRL logger and optionally prints
        formatted metrics to the Python logger for console monitoring.

        Args:
            metrics: Dictionary mapping metric names to scalar values.
            step: Training step or outer-loop cycle number for x-axis alignment.
            log_python: Whether to also log to Python logger. If None, uses
                config.logger.log_to_console setting.
            python_level: Logging level for Python logger output (default: INFO).

        Example:
            >>> self.log_metrics(
            ...     {"loss": 0.5, "reward": 100},
            ...     step=1000,
            ...     log_python=True
            ... )
        """

        if self.metrics is None:
            backend = self.config.logger.backend
            if backend not in (None, ""):
                msg = (
                    f"logger.backend={backend!r} is configured but no metric "
                    "reporter exists; the metrics would be dropped."
                )
                raise RuntimeError(msg)
            return

        should_log_python = (
            bool(self.config.logger.log_to_console)
            if log_python is None
            else log_python
        )
        self.metrics.log_scalars(
            metrics,
            step=step,
            log_python=should_log_python,
            python_level=python_level,
        )

    def _env_attribute(self, name: str, default: Any) -> Any:
        """``env.<name>`` by capability.

        A ``SerialEnv`` that is not running has no member envs, and its
        ``__getattr__`` recurses looking for them. The check cannot be
        ``env.is_closed``: the collector closes the env at the frame budget,
        before the last row, whose payload is still in ``log_infos``.
        """
        try:
            return getattr(self.env, name, default)
        except RecursionError:
            return default

    def _env_log_metrics(self) -> dict[str, Any]:
        """Newest env log payload as scalars; ``{}`` for envs without one.

        ``env.log_infos`` holds the payload of the latest step, detached on the
        device. Reading it here is the only host transfer, once per row.
        """
        log_infos = self._env_attribute("log_infos", None)
        if not isinstance(log_infos, deque) or not log_infos:
            return {}
        payload = log_infos.pop()
        log_infos.clear()
        metrics: dict[str, Any] = {}
        log_info(payload, metrics)
        return metrics

    def _required_training_entries(self) -> tuple[str, ...]:
        """Contract entries of this run: the algorithm's plus the env's own."""
        declared = self._env_attribute("required_log_entries", ())
        return (*REQUIRED_TRAINING_ENTRIES, *declared)

    def _interval_frame_rate(self, frames: int) -> float:
        """Frames per second since the previous training row; 0.0 on the first."""
        now = time.monotonic()
        previous, self._last_training_row = self._last_training_row, (now, frames)
        if previous is None or now <= previous[0]:
            return 0.0
        return (frames - previous[1]) / (now - previous[0])

    def log_training_metrics(
        self,
        metrics: MutableMapping[str, Any],
        *,
        frames: int,
        iteration: int,
        log_python: bool | None = False,
    ) -> None:
        """Write one training row: the only path from a training loop to the backend.

        The loop passes what it measured, at its own flush cadence. This merges
        the env log payload, stamps the entries every row must carry (in place,
        so the loop's text summaries see them), warns about any required entry
        still absent, and writes at ``step=frames``. Do not gate the call on the
        console cadence: ``trainer.log_interval`` and the progress bar govern
        text output only.

        Args:
            metrics: The row. Keys the env payload also defines keep this value.
            frames: Cumulative environment frames at this row.
            iteration: Completed outer-loop iterations.
            log_python: Forwarded to :meth:`log_metrics`.
        """
        frames = int(frames)
        self._start_train_clock()
        for key, value in self._env_log_metrics().items():
            metrics.setdefault(key, value)
        metrics.update(self._train_clock_metrics(frames))
        metrics[ITERATION_ENTRY] = float(iteration)
        rate = self._interval_frame_rate(frames)
        metrics.setdefault(THROUGHPUT_ENTRY, rate)
        missing = missing_training_entries(metrics, self._required_training_entries())
        unreported = frozenset(missing) - self._reported_missing_entries
        if unreported:
            self._reported_missing_entries |= unreported
            self.log.warning(
                "Training row at frame %d omits required entries: %s",
                frames,
                sorted(unreported),
            )
        self.log_metrics(metrics, step=frames, log_python=log_python)

    @abstractmethod
    def _progress_summary_fields(self) -> tuple[tuple[str, str], ...]:
        pass

    def _should_log_iteration(
        self, metadata: TrainingMetadata, iteration: IterationData
    ) -> bool:
        """Return whether this iteration prints a periodic text summary.

        Console cadence only. Metric rows go through
        :meth:`log_training_metrics` regardless: with the progress bar on this
        is False until the last iteration.
        """
        if metadata.stop_after_iteration:
            return True
        if metadata.progress_bar_enabled:
            # The progress bar already provides live progress updates; avoid
            # duplicate periodic logs and only emit logs at the final iteration.
            return (iteration.iteration_idx + 1) == metadata.total_iterations
        return (
            metadata.frames_processed >= metadata.next_log_frame
            or (iteration.iteration_idx + 1) == metadata.total_iterations
        )

    def _should_log_iteration_to_file(
        self, metadata: TrainingMetadata, iteration: IterationData
    ) -> bool:
        """Return whether this iteration should emit a training.log summary."""
        return (
            metadata.stop_after_iteration
            or metadata.frames_processed >= metadata.next_file_log_frame
            or (iteration.iteration_idx + 1) == metadata.total_iterations
        )

    def _file_summary_fields(self) -> tuple[tuple[str, str], ...]:
        """Return scalar metrics persisted in compact periodic file summaries."""
        return (
            *self._progress_summary_fields(),
            ("train/step_reward_std", "r_step_std"),
            ("train/step_reward_min", "r_step_min"),
            ("train/step_reward_max", "r_step_max"),
            ("train/loss_critic", "v_loss"),
            ("train/loss_entropy", "entropy_loss"),
            ("train/entropy", "entropy"),
            ("train/explained_variance", "ev"),
            ("train/grad_norm", "grad_norm"),
            ("train/lr", "lr"),
            ("train/clip_epsilon", "clip"),
        )

    def _format_iteration_summary(
        self,
        metadata: TrainingMetadata,
        iteration: IterationData,
        fields: tuple[tuple[str, str], ...],
    ) -> str:
        status_parts = [
            f"iter={iteration.iteration_idx + 1}/{metadata.total_iterations}",
            f"frames={metadata.frames_processed}/{self.config.collector.total_frames}",
        ]
        seen_aliases = set[str]()
        for metric_key, alias in fields:
            if alias in seen_aliases:
                continue
            seen_aliases.add(alias)
            metric_value = as_float(iteration.metrics.get(metric_key))
            if metric_value is not None:
                status_parts.append(f"{alias}={metric_value:.4f}")
        return " | ".join(status_parts)

    @staticmethod
    def _format_summary_count(value: int) -> str:
        abs_value = abs(value)
        if abs_value >= 1_000_000_000:
            return f"{value / 1_000_000_000:.3f}B"
        if abs_value >= 1_000_000:
            return f"{value / 1_000_000:.3f}M"
        if abs_value >= 1_000:
            return f"{value / 1_000:.3f}K"
        return str(value)

    def _format_iteration_table_summary(
        self,
        metadata: TrainingMetadata,
        iteration: IterationData,
        fields: tuple[tuple[str, str], ...],
    ) -> tuple[str, str]:
        columns = [
            (
                "iter",
                f"{iteration.iteration_idx + 1}/{metadata.total_iterations}",
                12,
            ),
            (
                "frames",
                (
                    f"{self._format_summary_count(metadata.frames_processed)}/"
                    f"{self._format_summary_count(self.config.collector.total_frames)}"
                ),
                17,
            ),
        ]
        seen_aliases = set[str]()
        for metric_key, alias in fields:
            if alias in seen_aliases:
                continue
            seen_aliases.add(alias)
            metric_value = as_float(iteration.metrics.get(metric_key))
            if metric_value is not None:
                columns.append((alias, f"{metric_value:.4f}", max(10, len(alias))))

        header = " ".join(f"{label:>{width}}" for label, _, width in columns)
        row = " ".join(f"{value:>{width}}" for _, value, width in columns)
        return header, row

    def _log_iteration_file_summary(
        self, metadata: TrainingMetadata, iteration: IterationData
    ) -> None:
        """Write compact periodic summaries to training.log without touching tqdm."""
        if not metadata.progress_bar_enabled:
            return
        if not self._should_log_iteration_to_file(metadata, iteration):
            return
        if (
            self.config.logger.backend
            and self.metrics is not None
            and self.metrics.rows_written == 0
        ):
            self.log.warning(
                "logger.backend=%s has received no metric rows after %d frames; "
                "the loop is not reaching log_training_metrics.",
                self.config.logger.backend,
                metadata.frames_processed,
            )

        header, row = self._format_iteration_table_summary(
            metadata,
            iteration,
            self._file_summary_fields(),
        )
        if metadata.file_summary_rows % self._FILE_SUMMARY_HEADER_INTERVAL == 0:
            log_to_file_only(self.log, logging.INFO, header)
        log_to_file_only(self.log, logging.INFO, row)
        metadata.file_summary_rows += 1

        while metadata.frames_processed >= metadata.next_file_log_frame:
            metadata.next_file_log_frame += metadata.log_interval_frames

    def _refresh_progress_display(
        self, metadata: TrainingMetadata, iteration: IterationData
    ) -> None:
        """Refresh tqdm or emit periodic text summaries for headless runs."""
        if not self._should_log_iteration(metadata, iteration):
            return

        self.log.info(
            self._format_iteration_summary(
                metadata,
                iteration,
                self._progress_summary_fields(),
            )
        )

        while metadata.frames_processed >= metadata.next_log_frame:
            metadata.next_log_frame += metadata.log_interval_frames
