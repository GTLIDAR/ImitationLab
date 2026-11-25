# Changelog

Notable changes to ImitationLab are recorded in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `env.reward_heading_velocity`, off by default: the `imitation-g1-v1`
  tracking reward compares body velocities with reference velocities turned by
  the same heading offset as the re-anchored pose targets, as upstream
  SuperMimic `5fb9229` and mujocolab/mjlab#1274 do, so a yaw offset lowers
  only the root orientation term. The simulator and world-model rewards change
  together. On BONES-SEED (SuperMimic explicit, H200, 30 minutes, seeds 42 and
  43) it tracked no better than the default on the 256-clip board.

## [0.1.1] - 2026-11-23

### Added

- A documentation site built from the Markdown guides with Sphinx:
  `pixi run -e docs docs-build`. The `Docs` workflow builds it on every pull
  request and deploys it to GitHub Pages from the public repository.
- A `Release` workflow that builds the source distribution and wheel when a
  GitHub release is published and uploads them to PyPI through trusted
  publishing.

### Changed

- The simulator stack moves to Isaac Lab
  [`f9af6e3`](https://github.com/isaac-sim/IsaacLab/commit/f9af6e3c52ccb43533996713985808fed2cd46db),
  Isaac Sim 6.2.0.0, Newton 1.8.0 and MuJoCo/MJWarp 3.17.0, from Isaac Lab
  `0612893`, Isaac Sim 6.1.0.0, Newton 1.6.1 and MuJoCo/MJWarp 3.12.0. Warp
  stays at 1.17.0 and PyTorch at 2.12.0. Isaac Lab now brings Hydra 1.4.
  Released checkpoints score the same on both stacks, and training throughput
  is unchanged.
- G1 actuator configs use Isaac Lab's limit fields: `joint_effort_limit`,
  `joint_velocity_limit` and `actuator_velocity_limit` replace
  `effort_limit_sim` and `velocity_limit_sim`, with the same values.
- `attach_video_recorder` builds a `VisualizerCfg` with a Newton GL backend and
  takes the frame size as `window_width` and `window_height`. With
  `--visualizer newton_gl`, a recording run now opens a windowed viewer next to
  the headless one that records.
- `launch_compat.needs_kit` asks Isaac Lab's `validate_launch` instead of
  repeating its rules.
- Environment variables use the `IMITATIONLAB_` prefix:
  `IMITATIONLAB_ENV_COLLECT_FASTPATH`, `IMITATIONLAB_UNITREE_USD_PATH`,
  `IMITATIONLAB_UNITREE_USD_CACHE_ROOT`, `IMITATIONLAB_COMPILE_MDP` and
  `IMITATIONLAB_MOTION_CACHE_ROOT`.
- The provenance page lists only the revisions of vendored and upstream
  components.

### Removed

- The migration guide.
- `--interface diffsr` and the `encoders/diffsr.args` recipe; use `spectral`.
  Saved records that name the interface `diffsr` are no longer read.
- Names carried over from earlier packages. The agent config keys are
  `agent_cfg_entry_point`, `ppo_cfg_entry_point` and `wmbptt_cfg_entry_point`;
  the config classes are `G1PPOConfig` and `G1WMBPTTConfig`; a packed bank's
  sidecar is `replay_manifest.json`; motion caches are named
  `imitationlab_g1_*`; the cluster runtime defaults to
  `/opt/imitationlab-runtime`. Rename `iltools_rb_manifest.json` in an existing
  packed bank to keep using it.
- Compatibility shims: the `rl-video-step-*.mp4` clip names, cluster profiles
  loaded from the package directory (only the example ships there), the
  `legacy`/`f_matrix` and `fz`/`z_fz` option spellings, alternate command
  component names such as `qpos_qvel`, and `--emulate_joint_order_from`.
- snake_case option spellings. Every option is kebab-case only, for example
  `--num-envs` and `--max-train-seconds`; update recipes, job scripts and
  argument files. A run recorded with snake_case options cannot be resumed
  from its record.
- The `scripts/tracking/` launchers. Start training and evaluation by module
  name (`python -m imitationlab.runners.train`); a cluster stage's
  `executable` is a module name. The PhysX bootstrap is
  `imitationlab.runners.train_physx` and the CU130 evaluator bridge is
  `imitationlab.runners.evaluate_cu130`.

### Fixed

- `collect_equivalence run` starts training with `imitationlab.runners.train`;
  it still pointed at the removed `scripts/tracking/train.py`.
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
