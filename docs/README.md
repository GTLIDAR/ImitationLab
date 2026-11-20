# Documentation

This index lists every ImitationLab guide and measured report. New users
should read the guides under [Using ImitationLab](#using-imitationlab) in
order; the [project README](../README.md) has the short version.

## Using ImitationLab

| Guide | Contents |
| --- | --- |
| [Setup](setup.md) | Pixi environments, Isaac Lab, and the optional official SuperMimic runner |
| [Motion data](../data/README.md) | LAFAN1 and BONES-SEED preparation and packed reference banks |
| [Training and evaluation](training.md) | Commands, policy learners, command interfaces, checkpoints, export, and learning-speed measurements |
| [Hardware selection](hardware.md) | Automatic execution sizes and local throughput tuning |
| [Running on a Slurm cluster](cluster.md) | Profiles, campaigns, and the plan/submit workflow |

## Design and development

| Guide | Contents |
| --- | --- |
| [Architecture](architecture.md) | Package boundaries and module ownership |
| [Policy learners](../imitationlab/algorithms/README.md) | Layout of the learner package |
| [Motion data tools](../imitationlab/data_tools/README.md) | Python API for loaders, replay storage, and reset sampling |
| [Contributing](../CONTRIBUTING.md) | Workflow, checks, and what to record when reporting results |
| [Migration](migration.md) | Earlier scripts, task IDs, module names, and checkpoints |
| [Provenance](provenance.md) | Source revisions this repository started from |

## Results

Each report states its task, data identity, encoder weights, seeds, hardware,
and limits.

| Report | Contents |
| --- | --- |
| [G1 release check, four GPU tiers](benchmarks/g1-release-check.md) | All 24 learner/interface/dataset cases on an RTX 4080, RTX PRO 6000, H100 and H200, with full-bank BONES-SEED evaluation |
| [G1 learning speed, 30-minute runs](benchmarks/g1-learning-speed-30min.md) | The earlier H100 study of the same 24 cases, with the first-release encoder window for latent policies |
| [Hardware selection: local verification](benchmarks/hardware-selection-verification.md) | Dated record of the automatic-sizing and `tune` checks on one workstation |
| [PPO tuning](benchmarks/ppo-tuning.md) | Stronger PPO recipes against the published PPO default |
| [SVG tuning](benchmarks/svg-tuning.md) | SVG recipe search against BPTT |
| [G1 learning speed, 10-minute runs](benchmarks/g1-learning-speed.md) | The earlier study and its follow-ups |
| [Compatibility matrix](compatibility-matrix.md) | Bounded integration check of all 24 learner/interface/dataset combinations |
| [Release verification](release-verification.md) | Bounded integration checks of motion ingestion, one training iteration, checkpoint reload, and a short rollout |

**Note on names.** Reports written before 2026-11-13 call the continuous
spectral skill interface `diffsr` or "DiffSR commands"; it is now
`--interface spectral`, and `--interface diffsr` remains an alias. "DiffSR" as
a policy learner (`--policy diffsr`) is unchanged.
