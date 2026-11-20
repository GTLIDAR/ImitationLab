# Contributing

This guide describes how to propose a change to ImitationLab: the branch and
pull-request workflow, the checks to run, where code belongs, and what to
record when reporting results. Please use
[GitHub issues](https://github.com/GTLIDAR/ImitationLab/issues) for bugs and
questions; the
[bug report](.github/ISSUE_TEMPLATE/bug_report.md) and
[feature request](.github/ISSUE_TEMPLATE/feature_request.md) templates list
what to include. Report security problems privately as described in
[SECURITY.md](SECURITY.md).

**Before you start.** Install [Pixi](https://pixi.sh) and the environments
described in [setup](docs/setup.md). Run all commands from the repository
root.

## Workflow

1. Create a feature branch from `main`. Use one branch for a coherent change.
2. Make the change and run the [checks](#checks).
3. Open a pull request against `main`. Describe the behavior you changed and
   include the commands and results that exercise it; the
   [pull-request template](.github/PULL_REQUEST_TEMPLATE.md) has the
   checklist.

```bash
git switch main
git switch -c feature/my-change
pixi install --locked
pixi run test-default
pixi run -e static ci-static
```

## Checks

| Command | When | Covers |
| --- | --- | --- |
| `pixi run test-default` | Every change | CPU test suites |
| `pixi run -e static ci-static` | Every change | `ruff check` and `ruff format --check` over the whole repository, and a compile check |
| `pixi run -e isaaclab test-isaaclab` | Simulator or environment changes | Isaac Lab suite on a local GPU |
| `BANK=<bank> pixi run -e isaaclab smoke-g1` | Trainer or simulator wiring changes | The [local Isaac Lab smoke](docs/setup.md#isaac-lab) on a packed reference bank |

CI runs the first two commands. Run `pixi run -e static ruff format .` before
committing. To run the ruff and file checks on every commit, run
`pixi run pre-commit install` once; the default environment provides
`pre-commit`.

## Tests

Put a new test file in the `tests/` directory of the package it covers. A
guard test fails when a test file is not run by any `test-*` pixi task.

## Where code belongs

Follow the package boundaries in [architecture](docs/architecture.md).

| Kind of code | Location |
| --- | --- |
| Reusable policy learners | `imitationlab/algorithms/` |
| Command encoders | `imitationlab/encoders/` |
| G1 tasks | `imitationlab/envs/isaaclab/` |
| Motion processing | `imitationlab/data_tools/` |
| Reusable run settings | `experiments/recipes/` |
| Entry points (keep them thin) | `scripts/` |

## Adding a learner or a command interface

- **New policy learner.** Provide settings for the existing observation and
  action contract and wire it through the training launcher.
- **New command interface.** Specify its encoding and checkpoint contract and
  test that the live and imagined observations agree.
- **Task contract changes.** Version the public task ID when its observation,
  action, or MDP contract changes.

## Reporting results

Record the resolved task ID, command interface, policy learner, seed, motion
manifest or bank identity, and encoder hash when reporting results.

Keep machine-specific paths, credentials, datasets, and checkpoints outside
Git.

## Documentation

Update the corresponding setup or training guide when changing an entry point
or recipe. Write command-line options in kebab-case (`--num-envs`); the
snake-case spellings recorded by earlier runs remain accepted. The
[documentation index](docs/README.md) lists every guide.
