"""Options are kebab-case; ``kebab_options`` renames the ones a library declares in snake case."""

import argparse

import pytest

from imitationlab.cli_args import kebab_options


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tool", allow_abbrev=False)
    parser.add_argument("--num_envs", type=int)
    parser.add_argument("--gumbel_hard", action=argparse.BooleanOptionalAction)
    parser.add_argument("--assert-kitless", action="store_true")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir")
    return kebab_options(parser)


def test_renamed_options_keep_their_destination():
    args, rest = _parser().parse_known_args(
        [
            "--num-envs=8",
            "--no-gumbel-hard",
            "--output-dir",
            "a_b-c",
            "env.some_key=a_b",
        ]
    )
    assert (args.num_envs, args.gumbel_hard, args.output_dir) == (8, False, "a_b-c")
    # Hydra overrides are not this parser's to change.
    assert rest == ["env.some_key=a_b"]


@pytest.mark.parametrize("flag", ["--num_envs", "--gumbel_hard", "--output_dir"])
def test_the_snake_case_spelling_is_not_an_option(flag):
    parser = _parser()
    assert flag not in parser._option_string_actions
    assert flag not in parser.format_help()


def test_subcommand_parsers_are_renamed():
    parser = argparse.ArgumentParser()
    run = parser.add_subparsers(dest="command").add_parser("run")
    run.add_argument("--num_envs", type=int)
    kebab_options(parser)
    assert parser.parse_args(["run", "--num-envs", "3"]).num_envs == 3
