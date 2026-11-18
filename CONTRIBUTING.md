# Contributing

Create a feature branch from `main` and open a pull request against `main`.
Use one branch for a coherent change. Describe the behavior you changed and
include the commands and results that exercise it.

```bash
git switch main
git switch -c feature/my-change
pixi install --locked
pixi run test-default
pixi run -e static ci-static
```

CI runs the same two commands. `ci-static` runs `ruff check` and
`ruff format --check` over the whole repository; run `pixi run -e static ruff
format .` before committing. For simulator or environment changes, also run
the Isaac Lab suite on a local GPU:

```bash
pixi run -e isaaclab test-isaaclab
```

Put a new test file in the `tests/` directory of the package it covers. A
guard test fails when a test file is not run by any `test-*` pixi task.

Follow the package boundaries in [architecture](docs/architecture.md). Put
reusable learners in `imitationlab/algorithms/`, command encoders in
`imitationlab/encoders/`, G1 tasks in `imitationlab/envs/isaaclab/`, and
motion processing in `imitationlab/data_tools/`. Keep reusable run settings in
`experiments/recipes/` and entry points in `scripts/` thin.

For a new learner, provide settings for the existing observation and action
contract and wire it through the training launcher. For a new command
interface, specify its encoding and checkpoint contract and test that the
live and imagined observations agree. Version the public task ID when its
observation, action, or MDP contract changes. Run the [local Isaac Lab
smoke](docs/setup.md#isaac-lab) for changes to trainer or simulator wiring.

Record the resolved task ID, command interface, policy learner, seed, motion
manifest or bank identity, and encoder hash when reporting results. Keep
machine-specific paths, credentials, datasets, and checkpoints outside Git.
Update the corresponding setup or training guide when changing an entry
point or recipe.
