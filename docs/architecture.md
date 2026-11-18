# Architecture

```text
imitationlab/
  cli.py                 train, pretrain, evaluate, export, and data commands
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

## Tracking

`imitation-g1-v1` is the public Isaac Lab task. The task configuration owns
observation and action layout, reward, termination, and reset behavior. The
live simulator and imagined rollout paths use the same G1 task math and
reference contract. The pure tensor modules in `task_model/` can be imported
without starting Isaac Lab. Simulator entry points register the tasks when
needed.

The command interface determines what reference information the actor sees.
Explicit commands expose reference features directly. The FSQ and spectral
interfaces encode the same future motion window with a frozen spectral skill
encoder (`encoders/diffsr`, quantized or continuous latent). The policy
learner consumes the task observations and handles its update under
`algorithms/`. The `diffsr` learner (`algorithms/diffsr`) is a different
component: a DiffSR spectral model of the low-level dynamics used for BPTT.

## Data and checkpoints

Dataset-specific conversion produces retargeted G1 clips and a motion
manifest. A packed reference bank stores arrays the environment can read
directly and stage by window. The data CLI prepares, packs, and validates
these artifacts. See [motion data](../data/README.md).

Runners resolve the task, learner, and interface, then record the settings
in the training run and its checkpoints. Evaluation uses the recorded
network settings and accepts a separate motion bank. A native export bundles
the checkpoint and any frozen encoder. See [training and
evaluation](training.md).

Tests for the learner, data, runner, and environment contracts live with
the owning packages. Code uses the `imitationlab` import paths. See
[migration](migration.md) for the names of earlier releases.
