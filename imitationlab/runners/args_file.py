"""Argument files for the launchers: ``--args-file PATH`` expands in place to the file's tokens.

A file holds one command-line token per line, exactly as it would be typed: an argparse flag
and its value on two lines or on one (``--num-envs 4096``), a Hydra override on one line
(``agent.wm.gamma=0.97``). Blank lines and lines that start with ``#`` are skipped. Two more
line forms exist:

* ``--args-file other.args`` includes another file, resolved against the including file's
  directory first;
* ``%env NAME=VALUE`` sets a process environment variable unless the caller already set it
  (some switches are read at import time and have no config field).

Expansion is positional, so precedence is the command line's own: argparse and Hydra both keep
the last value of a repeated option, hence a token typed after ``--args-file`` overrides the
file, and a tier file that includes the shared recipe first overrides the recipe.
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping, Sequence
from pathlib import Path

from imitationlab.paths import REPO_ROOT

FLAG = "--args-file"
ENV_PREFIX = "%env "


def _resolve(raw: str, base: Path | None) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    roots = ([base] if base is not None else []) + [Path.cwd(), REPO_ROOT]
    for root in roots:
        if (root / path).is_file():
            return (root / path).resolve()
    tried = ", ".join(str(root) for root in roots)
    raise FileNotFoundError(f"{FLAG} {raw}: not found under {tried}")


def _file_tokens(
    path: Path, environ: MutableMapping[str, str], stack: tuple[Path, ...]
) -> list[str]:
    if path in stack:
        chain = " -> ".join(str(p) for p in (*stack, path))
        raise ValueError(f"{FLAG} includes itself: {chain}")
    if not path.is_file():
        raise FileNotFoundError(f"{FLAG} {path}: no such file")
    raw: list[str] = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if text.startswith(ENV_PREFIX):
            name, sep, value = text[len(ENV_PREFIX) :].strip().partition("=")
            if not sep or not name.strip():
                raise ValueError(
                    f"{path}:{number}: expected '%env NAME=VALUE', got {text!r}"
                )
            environ.setdefault(name.strip(), value.strip())
            continue
        if text.startswith("--") and " " in text:
            flag, value = text.split(None, 1)
            raw += [flag, value.strip()]
        else:
            raw.append(text)
    return _expand(raw, environ, base=path.parent, stack=(*stack, path))


def _expand(
    tokens: Sequence[str],
    environ: MutableMapping[str, str],
    *,
    base: Path | None,
    stack: tuple[Path, ...],
) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == "--":  # everything after a bare "--" belongs to another program
            out += tokens[i:]
            break
        if token == FLAG:
            if i + 1 >= len(tokens):
                raise ValueError(f"{FLAG} needs a path")
            raw, i = tokens[i + 1], i + 2
        elif token.startswith(FLAG + "="):
            raw, i = token.partition("=")[2], i + 1
        else:
            out.append(token)
            i += 1
            continue
        out += _file_tokens(_resolve(raw, base), environ, stack)
    return out


def expand_args_files(
    argv: Sequence[str], environ: MutableMapping[str, str] | None = None
) -> list[str]:
    """Return ``argv`` with every ``--args-file PATH`` replaced by the file's tokens."""
    return _expand(
        list(argv), os.environ if environ is None else environ, base=None, stack=()
    )
