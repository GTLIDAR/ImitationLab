"""Errors that the command line reports as usage problems, not crashes."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager


class UserError(ValueError):
    """An input the user can correct: a missing file, a conflicting option.

    The command line prints the message with the usage line of the command
    that received the input and exits with status 2. Any other exception is a
    defect and keeps its traceback.
    """


@contextmanager
def reported_by(parser: argparse.ArgumentParser) -> Iterator[None]:
    """Report a ``UserError`` raised in the block through ``parser``."""
    try:
        yield
    except UserError as exc:
        parser.error(str(exc))
