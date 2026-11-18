"""Diffusion-based spectral representations and policy learning."""

from .model import BilinearSR, DiffSRBilinear

__all__ = ["BilinearSR", "DiffSRBilinear", "DiffSR"]


def __getattr__(name: str):
    if name == "DiffSR":
        from imitationlab.algorithms.diffsr.learner import DiffSR

        return DiffSR
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
