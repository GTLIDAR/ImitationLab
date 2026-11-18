import logging
from abc import ABC, abstractmethod
from typing import Any

from imitationlab.data_tools.common.metadata import DatasetMeta


class BaseLoader(ABC):
    """Interface for loading reference trajectories and their metadata."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @property
    @abstractmethod
    def metadata(self) -> DatasetMeta:
        """
        Returns the metadata of the dataset.
        """
        ...

    @abstractmethod
    def _get_trajectories(
        self, **kwargs: Any
    ) -> tuple[list[dict], dict[str, dict[str, dict[str, Any]]]]:
        """
        Returns the trajectories and motion information from the dataset.
        The trajectories are a list of dictionaries, each containing the information about a trajectory.
        The motion information is a dictionary of dictionaries, each containing the information about a motion.
        """
        ...
