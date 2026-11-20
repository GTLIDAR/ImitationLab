"""Kebab-case spellings for command-line options.

Documented options are kebab-case (``--num-envs``). The trainer, the encoder
pretrainer, the evaluator and Isaac Lab's launcher declare many options in
snake case (``--num_envs``), and saved runs, recipes and cluster scripts
record those spellings, so both must keep working.

Two functions cover this:

* :func:`add_kebab_aliases` registers ``--foo-bar`` next to every
  ``--foo_bar`` option of a parser. The alias shares the option's action, so
  the destination, type and default are unchanged.
* :func:`canonical_argv` rewrites an alias back to the spelling the option was
  declared with. Code that inspects tokens by name (argument files, hardware
  sizing, resume) then sees one spelling.

Only option names are rewritten, and only when the option is known. Option
values, positional arguments, Hydra ``key=value`` overrides, unknown options
and everything after a bare ``--`` are left as typed.
"""

from __future__ import annotations

import argparse
import functools
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

from imitationlab.data_tools.common.cli_args import hide_legacy_aliases

_SNAKE_OPTION = re.compile(r"""["'](--[a-z0-9]+(?:_[a-z0-9]+)+)["']""")
_PACKAGE_ROOT = Path(__file__).resolve().parent


def _kebab(option: str) -> str:
    return "--" + option[2:].replace("_", "-")


def add_kebab_aliases(
    parser: argparse.ArgumentParser, *, hide_snake_case: bool = True
) -> argparse.ArgumentParser:
    """Accept ``--foo-bar`` for every ``--foo_bar`` option of ``parser``.

    Call this after all options are added. Subcommand parsers are included.
    An alias is skipped when its spelling already names an option. Help lists
    the kebab-case spelling first; ``hide_snake_case`` removes the snake-case
    spelling from help. Pass ``hide_snake_case=False`` for a parser whose help
    formatter is chosen later by other code.
    """
    for action in list(parser._actions):
        if isinstance(action, argparse._SubParsersAction):
            for subparser in {id(p): p for p in action.choices.values()}.values():
                add_kebab_aliases(subparser, hide_snake_case=hide_snake_case)
            continue
        spellings = []
        for option in action.option_strings:
            alias = _kebab(option)
            if (
                option.startswith("--")
                and "_" in option
                and alias not in parser._option_string_actions
            ):
                parser._option_string_actions[alias] = action
                spellings.append(alias)
            spellings.append(option)
        action.option_strings[:] = spellings
    if hide_snake_case:
        hide_legacy_aliases(parser)
    return parser


def _declared_spellings(parser: argparse.ArgumentParser) -> dict[str, str]:
    """Map each kebab-case alias of ``parser`` to the snake-case spelling beside it."""
    spellings = {}
    for action in parser._actions:
        for option in action.option_strings:
            if option.startswith("--") and "_" in option:
                alias = _kebab(option)
                if alias != option and alias in action.option_strings:
                    spellings[alias] = option
    return spellings


@functools.cache
def snake_case_options(*modules: str) -> frozenset[str]:
    """Return the snake-case long options that the named modules declare.

    ``modules`` are dotted names under ``imitationlab``. Their source is read,
    not imported: the modules that own these parsers start Isaac Lab on
    import. Every quoted ``--snake_case`` literal counts, which also covers
    options added in a loop.
    """
    options: set[str] = set()
    for module in modules:
        relative = Path(*module.split(".")[1:])
        source = _PACKAGE_ROOT / relative.with_suffix(".py")
        options.update(_SNAKE_OPTION.findall(source.read_text(encoding="utf-8")))
    return frozenset(options)


def canonical_argv(
    argv: Sequence[str],
    known: argparse.ArgumentParser | Iterable[str],
) -> list[str]:
    """Rewrite kebab-case aliases in ``argv`` to their declared snake-case spelling.

    ``known`` is a parser prepared with :func:`add_kebab_aliases`, or the
    snake-case options of a parser that cannot be built here (see
    :func:`snake_case_options`). A token is rewritten only when it is
    ``--name`` or ``--name=value`` and ``name`` is the kebab-case form of a
    known snake-case option; ``--no-name`` follows a known ``--name``.
    """
    if isinstance(known, argparse.ArgumentParser):
        spellings = _declared_spellings(known)
    else:
        spellings = {}
        for option in known:
            spellings.setdefault(_kebab(option), option)
            # argparse.BooleanOptionalAction negations.
            spellings.setdefault("--no-" + _kebab(option)[2:], "--no-" + option[2:])
    result = []
    for index, token in enumerate(argv):
        if token == "--":
            result.extend(argv[index:])
            break
        if token.startswith("--"):
            name, separator, value = token.partition("=")
            token = spellings.get(name, name) + separator + value
        result.append(token)
    return result


# Snake-case options behind each public command's pass-through arguments.
_SHARED = ("imitationlab.runners.launch_compat", "imitationlab.runners.args_file")
TRAIN_MODULES = ("imitationlab.runners.train", *_SHARED)
PRETRAIN_MODULES = ("imitationlab.runners.pretrain", *_SHARED)
EVALUATE_MODULES = ("imitationlab.evaluation.evaluate_checkpoint", *_SHARED)
