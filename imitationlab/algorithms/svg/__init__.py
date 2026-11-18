"""Real-window updates for stochastic value gradients."""

__all__ = ["SVG"]


def __getattr__(name: str):
    if name == "SVG":
        from imitationlab.algorithms.svg.learner import SVG

        return SVG
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
