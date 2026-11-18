"""Run an ``imitationlab`` module from a thin script under ``scripts/``."""

from __future__ import annotations

import importlib
import runpy
import sys


def run_module(target: str, namespace: dict) -> None:
    """Run ``target`` as ``__main__``, or alias it when the script is imported."""
    if namespace.get("__name__") == "__main__":
        runpy.run_module(target, run_name="__main__", alter_sys=True)
    else:
        module = importlib.import_module(target)
        sys.modules[namespace["__name__"]] = module
        namespace.update(vars(module))
