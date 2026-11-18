"""Imitation learning with Isaac Lab environments and TorchRL learners."""

from importlib import metadata as _metadata

try:
    __version__ = _metadata.version("imitationlab")
except _metadata.PackageNotFoundError:  # running from a source tree without install
    __version__ = "0.0.0+unknown"
