#!/usr/bin/env python3
"""Script entry point of the ImitationLab implementation."""

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from imitationlab._entry_points import run_module

run_module("imitationlab.evaluation.evaluate_checkpoint_tree", globals())
