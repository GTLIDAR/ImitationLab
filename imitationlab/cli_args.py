"""Kebab-case command-line options.

Every option is kebab-case (``--num-envs``). Isaac Lab's launcher declares its
own options in snake case (``--kit_args``); :func:`kebab_options` renames
those, so one spelling is accepted and documented.
"""

from __future__ import annotations

import argparse


def kebab_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Rename every ``--foo_bar`` option of ``parser`` to ``--foo-bar``.

    Call this after all options are added. Subcommand parsers are included.
    The destination, type and default of an option are unchanged.
    """
    for action in list(parser._actions):
        if isinstance(action, argparse._SubParsersAction):
            for subparser in {id(p): p for p in action.choices.values()}.values():
                kebab_options(subparser)
            continue
        for index, option in enumerate(action.option_strings):
            if not option.startswith("--") or "_" not in option:
                continue
            renamed = "--" + option[2:].replace("_", "-")
            del parser._option_string_actions[option]
            parser._option_string_actions.setdefault(renamed, action)
            action.option_strings[index] = renamed
        # An option declared in both spellings now lists one of them twice.
        action.option_strings[:] = dict.fromkeys(action.option_strings)
    return parser
