# Architecture

This guide describes how the repository is organized: which package owns
which responsibility, and how the task, the command interfaces, the policy
learners, and the data fit together. It is for contributors and for users who
want to read or extend the code.

## Repository layout

```text
imitationlab/
  cli.py                 train, pretrain, evaluate, export, data, hardware,
                         tune, and benchmark commands
  algorithms/
    base.py              policy learner interface
    common/              components shared by multiple learners
    ppo/
    supermimic/
    svg/
    diffsr/
  encoders/
    fsq/
    diffsr/
  envs/isaaclab/
    tasks/               Gym registration and configuration of imitation-g1-v1
    envs/                Isaac Lab environment and simulator adapters
    contracts/           observation, action, and reference contracts
    task_model/          G1 kinematics and differentiable task math
    assets/              robot assets
  data_tools/            conversion, reference banks, and validation
  runners/               launch, training, pretraining, and checkpoints
  evaluation/            checkpoint rollouts, metrics, and export helpers
experiments/recipes/     reusable command and learner settings
scripts/                 thin entry points and operational helpers
third_party/supermimic/  optional official mjlab baseline
```

## The tracking task

`imitation-g1-v1` is the public Isaac Lab task and the only registered one.
The task configuration owns the observation and action layout, reward,
termination, and reset behavior.

The live simulator and imagined rollout paths use the same G1 task math and
reference contract. The pure tensor modules in `task_model/` can be imported
without starting Isaac Lab. Simulator entry points register the task when
needed.

## Command interfaces and policy learners

The command interface (`--interface`) determines what reference information
the actor sees:

- The explicit interface exposes reference features directly.
- The FSQ and spectral interfaces encode the same future motion window with a
  frozen spectral skill encoder (`encoders/diffsr`), with a quantized (FSQ,
  finite scalar quantization) or continuous latent.

The policy learner (`--policy`) consumes the task observations and handles its
update under `algorithms/`. See the
[policy learner package](../imitationlab/algorithms/README.md).

**Note.** The name `diffsr` appears in two places. `encoders/diffsr` is the
spectral skill encoder used by the latent interfaces. `algorithms/diffsr` is
the `diffsr` policy learner, a different component: a DiffSR spectral model of
the low-level dynamics used for backpropagation through time (BPTT).

## Data

Dataset-specific conversion produces retargeted G1 clips and a motion
manifest. A packed reference bank stores arrays the environment can read
directly and stage by window. The data CLI prepares, packs, and validates
these artifacts. See [motion data](../data/README.md) for the commands and the
[data tools package](../imitationlab/data_tools/README.md) for the Python API.

## Runs and checkpoints

Runners resolve the task, learner, and interface, then record the settings in
the training run and its checkpoints. Evaluation uses the recorded network
settings and accepts a separate motion bank. A native export bundles the
checkpoint and any frozen encoder. See
[training and evaluation](training.md).

## Tests and import paths

Tests for the learner, data, runner, and environment contracts live with the
owning packages. Code uses the `imitationlab` import paths. See
[migration](migration.md) for the names of earlier releases and
[contributing](../CONTRIBUTING.md) for where new code belongs.
