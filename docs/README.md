# Documentation

## Use

- [Setup](setup.md): Pixi environments, Isaac Lab, and the optional official
  SuperMimic runner.
- [Motion data](../data/README.md): LAFAN1 and BONES-SEED preparation and
  packed reference banks.
- [Training and evaluation](training.md): commands, learners, interfaces,
  checkpoints, export, and learning-speed measurements.
- [Hardware selection](hardware.md): automatic execution sizes and local
  throughput tuning.
- [Running on a Slurm cluster](cluster.md): profiles, campaigns, and the
  plan/submit workflow.

## Design

- [Architecture](architecture.md): package boundaries and module ownership.
- [Migration](migration.md): old scripts, task IDs, and checkpoints.
- [Provenance](provenance.md): source revisions this repository started from.

## Results

Each report states its task, data identity, encoder weights, seeds, hardware,
and limits. Reports written before 2026-11-13 call the continuous spectral
skill interface `diffsr` or "DiffSR commands"; it is now `--interface
spectral`. "DiffSR" as a policy learner is unchanged.

- [G1 release check, four GPU tiers](benchmarks/g1-release-check.md): all 24
  learner/interface/dataset cases on an RTX 4080, RTX PRO 6000, H100 and H200,
  with full-bank BONES-SEED evaluation.
- [G1 learning speed, 30-minute runs](benchmarks/g1-learning-speed-30min.md):
  the earlier H100 study of the same 24 cases, with the first-release encoder
  window for latent policies.
- [PPO tuning](benchmarks/ppo-tuning.md): stronger PPO recipes against the
  published PPO default.
- [SVG tuning](benchmarks/svg-tuning.md): SVG recipe search against BPTT.
- [G1 learning speed, 10-minute runs](benchmarks/g1-learning-speed.md): the
  earlier study and its follow-ups.
- [Compatibility matrix](compatibility-matrix.md) and
  [release verification](release-verification.md): bounded integration checks.
