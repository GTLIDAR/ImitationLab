# scripts

Use the `imitationlab` command line for all workflows; see the
[project README](../README.md#commands) and
[training and evaluation](../docs/training.md).

This directory holds thin launchers and compatibility entry points. Each file
runs a module of the `imitationlab` package under a stable file path:

- The cluster control plane (`imitationlab/runners/pipeline/cluster/` and
  `docker/cluster/run_singularity.sh`) starts jobs by script path, for example
  `scripts/tracking/train.py`.
- Run directories written by earlier versions record commands that name
  these paths, such as `python scripts/tracking/train.py ...`.

Keep new logic in the package and its tests in the package's `tests/`
directory. [Migration](../docs/migration.md) describes the move from the
earlier layout.
