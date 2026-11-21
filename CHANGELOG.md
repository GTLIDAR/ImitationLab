# Changelog

Notable changes to ImitationLab are recorded in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- A documentation site built from the Markdown guides with Sphinx:
  `pixi run -e docs docs-build`. The `Docs` workflow builds it on every pull
  request and deploys it to GitHub Pages from the public repository.
- A `Release` workflow that builds the source distribution and wheel when a
  GitHub release is published and uploads them to PyPI through trusted
  publishing.

### Fixed

- The wheel now contains the three USD layers under
  `g1_description/configuration/` that the G1 robot file references. They were
  missing from the package data, so only an editable install could load the
  robot. A test now checks that every tracked data file is package data.

## [0.1.0] - 2026-11-20

First public release.

### Added

- The `imitation-g1-v1` task: whole-body motion tracking for the Unitree G1
  in Isaac Lab with the Newton/MJWarp physics backend.
- The `imitationlab` command line with the `data`, `pretrain`, `train`,
  `evaluate`, `export`, `hardware`, `tune`, and `benchmark` commands.
- Four policy learners on the same observation, action, and reward contract:
  SuperMimic-style BPTT (`supermimic`), `svg`, DiffSR dynamics (`diffsr`),
  and `ppo`.
- Three command interfaces: `explicit` future reference frames, and the
  `fsq` and `spectral` latent interfaces, which read a frozen spectral skill
  encoder.
- The `reference_heading` encoder window for the latent interfaces. It reads
  only the reference motion, so a latent policy needs an IMU but no odometry
  on the robot.
- Data tools that download or convert LAFAN1 and BONES-SEED motions,
  validate them, and pack them into reference banks. No motion data is
  distributed with the repository.
- Automatic hardware sizing: execution sizes are chosen from the GPU, host
  memory, and motion bank, and `imitationlab tune` measures them locally.
- Benchmark and report tooling that measures tracking quality against
  training time.
- Measured reports, including the
  [four-tier release check](docs/benchmarks/g1-release-check.md) of all 24
  learner, interface, and dataset cases on four GPUs.
- Locked Pixi environments, versioned recipes, and checkpoints that record
  their task, learner, interface, data identity, encoder weights, and seed.
