"""Entry point; the imitationlab CLI is the documented interface."""

from imitationlab._entry_points import run_module

run_module("imitationlab.runners.pretrain", globals())
