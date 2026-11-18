"""Command-line conventions shared by the data tools.

Long options are kebab-case (``--input-fps``). Earlier releases spelled many
of them in snake case (``--input_fps``). Those spellings stay valid: declare
them as additional option strings after the kebab-case form, for example
``parser.add_argument("--input-fps", "--input_fps", ...)``. Argparse derives
the attribute name from the first long option, so ``args.input_fps`` is
unchanged. :func:`hide_legacy_aliases` removes the snake-case spellings from
help output so each option is documented once.
"""

from __future__ import annotations

import argparse
import copy


def _is_legacy_alias(option: str, options: list[str]) -> bool:
    return (
        option.startswith("--")
        and "_" in option
        and option.replace("_", "-") in options
    )


class _HideLegacyAliases(argparse.HelpFormatter):
    def _format_action_invocation(self, action: argparse.Action) -> str:
        visible = [
            option
            for option in action.option_strings
            if not _is_legacy_alias(option, action.option_strings)
        ]
        if len(visible) != len(action.option_strings):
            action = copy.copy(action)
            action.option_strings = visible
        return super()._format_action_invocation(action)


def hide_legacy_aliases(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Omit snake-case option aliases from the help of ``parser`` and its subcommands."""
    base = parser.formatter_class
    if isinstance(base, type) and not issubclass(base, _HideLegacyAliases):
        parser.formatter_class = type(
            f"HideLegacyAliases{base.__name__}", (_HideLegacyAliases, base), {}
        )
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for subparser in {id(p): p for p in action.choices.values()}.values():
                hide_legacy_aliases(subparser)
    return parser
