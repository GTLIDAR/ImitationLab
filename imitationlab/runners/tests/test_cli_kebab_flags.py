"""Every option is accepted in kebab case; snake case keeps working."""

import argparse
import shlex

import pytest
import torch

from imitationlab.cli import main
from imitationlab.cli_args import (
    EVALUATE_MODULES,
    PRETRAIN_MODULES,
    TRAIN_MODULES,
    add_kebab_aliases,
    canonical_argv,
    snake_case_options,
)
from imitationlab.runners.args_file import expand_args_files


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tool", allow_abbrev=False)
    parser.add_argument("--num_envs", type=int)
    parser.add_argument("--trajectory_ranks", nargs="+")
    parser.add_argument("--wandb_mode")
    parser.add_argument("--gumbel_hard", action=argparse.BooleanOptionalAction)
    parser.add_argument("--assert-kitless", action="store_true")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir")
    parser.add_argument("--seed", type=int)
    return add_kebab_aliases(parser)


@pytest.mark.parametrize(
    ("kebab", "snake"),
    [
        (["--num-envs", "4"], ["--num_envs", "4"]),
        (["--num-envs=4"], ["--num_envs=4"]),
        (["--trajectory-ranks", "0", "1"], ["--trajectory_ranks", "0", "1"]),
        (["--gumbel-hard"], ["--gumbel_hard"]),
        (["--no-gumbel-hard"], ["--no-gumbel_hard"]),
        (["--output-dir", "a_b-c"], ["--output_dir", "a_b-c"]),
        (["--wandb-mode", "off-line_x"], ["--wandb_mode", "off-line_x"]),
    ],
)
def test_alias_parses_like_the_declared_spelling(kebab, snake):
    assert vars(_parser().parse_args(kebab)) == vars(_parser().parse_args(snake))


def test_alias_keeps_destination_and_kebab_only_options():
    parser = _parser()
    args, rest = parser.parse_known_args(
        ["--num-envs", "8", "--assert-kitless", "env.some-key=a_b-c", "--other-flag"]
    )
    assert args.num_envs == 8 and args.assert_kitless
    # Hydra overrides and unknown options are not this parser's to change.
    assert rest == ["env.some-key=a_b-c", "--other-flag"]
    # A kebab-only option gains no snake-case spelling.
    assert "--assert_kitless" not in parser._option_string_actions


def test_help_documents_kebab_case_only():
    text = _parser().format_help()
    assert "--num-envs" in text and "--trajectory-ranks" in text
    assert "--num_envs" not in text and "--output_dir" not in text
    both = argparse.ArgumentParser()
    both.add_argument("--num_envs")
    listed = add_kebab_aliases(both, hide_snake_case=False).format_help()
    assert "--num-envs NUM_ENVS, --num_envs NUM_ENVS" in listed


def test_alias_is_skipped_when_the_spelling_is_taken():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry_run", dest="other", action="store_true")
    add_kebab_aliases(parser)
    assert parser.parse_args(["--dry-run"]).other is False


def test_subcommand_parsers_get_aliases():
    parser = argparse.ArgumentParser()
    run = parser.add_subparsers(dest="command").add_parser("run")
    run.add_argument("--num_envs", type=int)
    add_kebab_aliases(parser)
    assert parser.parse_args(["run", "--num-envs", "3"]).num_envs == 3


def test_canonical_argv_rewrites_only_known_option_names():
    known = {"--num_envs", "--wandb_mode", "--gumbel_hard", "--args_file"}
    argv = [
        "--num-envs", "16",
        "--wandb-mode=dry-run_mode",
        "--no-gumbel-hard",
        "--assert-kitless",
        "--output-dir", "runs/my-run_1",
        "--note", "--num-envs is a flag",
        "agent.logger.log_dir=outputs/smoke-g1",
        "env.data.persist-id=lafan1@3d-30_66",
        "num-envs",
        "--",
        "--num-envs", "2",
    ]  # fmt: skip
    assert canonical_argv(argv, known) == [
        "--num_envs", "16",
        "--wandb_mode=dry-run_mode",
        "--no-gumbel_hard",
        "--assert-kitless",
        "--output-dir", "runs/my-run_1",
        "--note", "--num-envs is a flag",
        "agent.logger.log_dir=outputs/smoke-g1",
        "env.data.persist-id=lafan1@3d-30_66",
        "num-envs",
        "--",
        "--num-envs", "2",
    ]  # fmt: skip
    snake = ["--num_envs", "16", "--no-gumbel_hard", "a.b-c=d_e"]
    assert canonical_argv(snake, known) == snake


def test_canonical_argv_from_a_parser_matches_its_declarations():
    parser = _parser()
    argv = ["--num-envs", "4", "--output-dir", "x-y", "--assert-kitless", "k-a=v-b"]
    assert canonical_argv(argv, parser) == [
        "--num_envs", "4", "--output_dir", "x-y", "--assert-kitless", "k-a=v-b"
    ]  # fmt: skip


def test_downstream_options_are_found_without_importing_the_simulator():
    train = snake_case_options(*TRAIN_MODULES)
    assert {"--num_envs", "--max_iterations", "--max_train_seconds"} <= train
    assert {"--args_file", "--enable_cameras", "--log_interval"} <= train
    pretrain = snake_case_options(*PRETRAIN_MODULES)
    assert {"--num_updates", "--wandb_mode", "--output_dir"} <= pretrain
    evaluate = snake_case_options(*EVALUATE_MODULES)
    assert {"--trajectory_ranks", "--output_json", "--output_csv"} <= evaluate
    for options in (train, pretrain, evaluate):
        assert all(
            option.startswith("--") and "-" not in option[2:] for option in options
        )


def test_args_file_is_expanded_under_either_spelling(tmp_path):
    recipe = tmp_path / "run.args"
    recipe.write_text("--num_envs 8\nagent.some-key=a_b\n")
    expected = ["--num_envs", "8", "agent.some-key=a_b", "--seed", "1"]
    for flag in ("--args_file", "--args-file"):
        assert expand_args_files([flag, str(recipe), "--seed", "1"]) == expected
        assert expand_args_files([f"{flag}={recipe}", "--seed", "1"]) == expected


def _dry_run(capsys, argv: list[str]) -> list[str]:
    assert main([*argv, "--dry-run"]) == 0
    return shlex.split(capsys.readouterr().out)


def test_train_passes_kebab_and_snake_options_identically(tmp_path, capsys):
    recipe = tmp_path / "user.args"
    recipe.write_text("agent.save_interval=1\n")
    base = ["train", "--policy", "ppo", "--hardware", "off"]
    kebab = _dry_run(
        capsys,
        [
            *base,
            "--num-envs", "16",
            "--max-iterations=1",
            "--max-train-seconds", "30",
            "--args-file", str(recipe),
            "--log-interval", "5",
            "--video-length", "3",
            "--assert-kitless",
            "agent.logger.log_dir=outputs/smoke-g1_a",
            "env.some-key=a-b_c",
        ],
    )  # fmt: skip
    snake = _dry_run(
        capsys,
        [
            *base,
            "--num_envs", "16",
            "--max_iterations=1",
            "--max_train_seconds", "30",
            "--args_file", str(recipe),
            "--log_interval", "5",
            "--video_length", "3",
            "--assert-kitless",
            "agent.logger.log_dir=outputs/smoke-g1_a",
            "env.some-key=a-b_c",
        ],
    )  # fmt: skip
    assert kebab == snake
    assert kebab[-7:] == [
        "--log_interval", "5",
        "--video_length", "3",
        "--assert-kitless",
        "agent.logger.log_dir=outputs/smoke-g1_a",
        "env.some-key=a-b_c",
    ]  # fmt: skip
    assert "--num_envs" in kebab and "--args_file" in kebab
    assert not any(arg.startswith(("--num-envs", "--max-iterations")) for arg in kebab)


def test_pretrain_passes_kebab_and_snake_options_identically(tmp_path, capsys):
    out = tmp_path / "enc-run_1"
    base = ["pretrain", "--interface", "fsq", "--hardware", "off"]
    kebab = _dry_run(
        capsys,
        [
            *base,
            "--output-dir", str(out),
            "--num-updates", "2",
            "--wandb-mode", "disabled",
            "--no-gumbel-hard",
            "--log-interval=1",
            "env.some-key=a-b_c",
        ],
    )  # fmt: skip
    snake = _dry_run(
        capsys,
        [
            *base,
            "--output-dir", str(out),
            "--num_updates", "2",
            "--wandb_mode", "disabled",
            "--no-gumbel_hard",
            "--log_interval=1",
            "env.some-key=a-b_c",
        ],
    )  # fmt: skip
    assert kebab == snake
    assert kebab[kebab.index("--output_dir") + 1] == str(out)
    assert kebab[-7:] == [
        "--num_updates", "2",
        "--wandb_mode", "disabled",
        "--no-gumbel_hard",
        "--log_interval=1",
        "env.some-key=a-b_c",
    ]  # fmt: skip


def test_evaluate_passes_kebab_and_snake_options_identically(tmp_path, capsys):
    model = tmp_path / "model.pt"
    torch.save(
        {"imitationlab": {"argv": ["--task", "imitation-g1-v1", "--algo", "PPO"]}},
        model,
    )
    output = tmp_path / "eval-out_1" / "result.json"
    base = ["evaluate", "--checkpoint", str(model), "--output", str(output)]
    kebab = _dry_run(
        capsys,
        [
            *base,
            "--num-envs", "4",
            "--trajectory-ranks", "0", "1", "2", "3",
            "--output-csv", str(tmp_path / "per-clip_rows.csv"),
            "--keep-after-done",
            "env.some-key=a-b_c",
        ],
    )  # fmt: skip
    snake = _dry_run(
        capsys,
        [
            *base,
            "--num_envs", "4",
            "--trajectory_ranks", "0", "1", "2", "3",
            "--output_csv", str(tmp_path / "per-clip_rows.csv"),
            "--keep_after_done",
            "env.some-key=a-b_c",
        ],
    )  # fmt: skip
    assert kebab == snake
    assert kebab[kebab.index("--output_json") + 1] == str(output)
    ranks = kebab.index("--trajectory_ranks")
    assert kebab[ranks : ranks + 5] == ["--trajectory_ranks", "0", "1", "2", "3"]
    assert kebab[kebab.index("--output_csv") + 1] == str(tmp_path / "per-clip_rows.csv")
    assert "env.some-key=a-b_c" in kebab and "--keep_after_done" in kebab
