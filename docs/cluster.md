# Running on a Slurm cluster

This guide describes the cluster control plane, which submits ImitationLab
stages to a Slurm cluster that runs the Isaac Lab container image. It is for
users who already train on a workstation and want to run the same commands as
cluster jobs.

**Note.** The guide assumes that your site provides an Isaac Lab container
image. No image recipe or example campaign ships with the repository yet; you
write the campaign file and the site profile yourself.

**Before you start.** You need:

- a working local installation ([setup](setup.md)), since planning and
  submission run on your machine;
- ssh access to the cluster's login node through an alias in `~/.ssh/config`;
- the Isaac Lab container image and your motion data on the cluster;
- a [profile](#profiles) for your site and a [campaign](#campaigns) file.

## Workflow

Submission works in two steps:

1. `plan` resolves a campaign, runs preflight checks over ssh, and freezes the
   job scripts.
2. `submit` packs the working tree, uploads it, and submits the frozen plan
   only when you confirm the plan hash.

Run the commands from the repository root in the default Pixi environment.
Replace the upper-case placeholders with the plan directory, plan hash, and
submission record that `plan` and `submit` print:

```bash
pixi run python -m imitationlab.runners.pipeline.cluster plan \
  --campaign campaign.yaml --arm sweep --seed 42 --out-root outputs/cluster/plans
pixi run python -m imitationlab.runners.pipeline.cluster submit \
  --plan PLAN_DIR --confirm PLAN_SHA
pixi run python -m imitationlab.runners.pipeline.cluster status \
  --submission SUBMISSION_JSON
pixi run python -m imitationlab.runners.pipeline.cluster logs \
  --submission SUBMISSION_JSON --stage STAGE
pixi run python -m imitationlab.runners.pipeline.cluster cancel \
  --submission SUBMISSION_JSON --yes
```

| Command | Purpose |
| --- | --- |
| `plan` | Resolve, preflight, and freeze a campaign arm; never submits |
| `submit` | Execute a confirmed plan |
| `status` | Show the `squeue`/`sacct` view of a submission |
| `logs` | Tail a submitted stage's Slurm log |
| `cancel` | Cancel a submission's jobs with `scancel` |

`submit` writes the submission record (`submission-*.json`) into the plan
directory. Run any command with `--help` for its options.

**Note.** Submission packs the current working tree. To run a fixed revision
while you keep editing, plan and submit from a separate `git worktree` of that
commit.

## Profiles

A profile names the login alias, the cluster paths, the container image, and
the Slurm resources. Profiles contain user directories, allocations, and
credential-file paths, so they stay out of Git. The package ships only the
commented
[example profile](../imitationlab/runners/pipeline/cluster/conf/profile_example.yaml).

Copy the example to a profile directory and fill in your site values. The
control plane searches these locations in order:

1. `$IMITATIONLAB_CLUSTER_PROFILE_DIR/profile_<name>.yaml`
2. `$XDG_CONFIG_HOME/imitationlab/cluster/profile_<name>.yaml`, which defaults
   to `~/.config/imitationlab/cluster/`
3. The packaged `conf/` directory. This location is deprecated and emits a
   warning; it remains for one release.

A campaign selects a profile with `profile: <name>`, and commands accept
`--profile <name>` or a path to a `.yaml` file.

## Campaigns

A campaign is a YAML file that lists stages, each with an executable and
arguments, plus preflight paths that must exist inside the container. A
learning-speed stage usually runs `-m imitationlab benchmark run ...`; see
[training and evaluation](training.md#measure-learning-speed).

Keep raw results, checkpoints, and BONES-SEED-derived data on the cluster or
under `outputs/`, never in Git.

## Related guides

- [Training and evaluation](training.md): the commands a stage runs.
- [Hardware selection](hardware.md): how execution sizes are chosen for a GPU.
