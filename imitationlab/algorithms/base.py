"""The learner interface used by training and evaluation entry points."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from imitationlab.algorithms.common.config import AlgorithmConfig

ConfigT = TypeVar("ConfigT", bound="AlgorithmConfig")


class BaseAlgorithm(ABC, Generic[ConfigT]):
    """Train a policy, compute actions, and save or restore training state."""

    @abstractmethod
    def train(self) -> None:
        """Run the learning loop."""

    @abstractmethod
    def predict(self, obs: Any) -> Any:
        """Return deterministic actions for the supplied observations."""

    @abstractmethod
    def save_model(
        self, path: str | Path | None = None, step: int | None = None
    ) -> None:
        """Save a training checkpoint."""

    @abstractmethod
    def load_model(self, path: str) -> None:
        """Restore the model and optimizer state needed to resume training."""
