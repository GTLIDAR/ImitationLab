# scripts

Use the `imitationlab` command line for all workflows; see the
[project README](../README.md#commands) and
[training and evaluation](../docs/training.md).

This directory holds thin launchers for data tools, validation and
visualization. Each file runs a module of the `imitationlab` package. Training
and evaluation start by module name, for example
`python -m imitationlab.runners.train`; the cluster control plane
(`imitationlab/runners/pipeline/cluster/` and
`docker/cluster/run_singularity.sh`) does the same.

Keep new logic in the package and its tests in the package's `tests/`
directory.
