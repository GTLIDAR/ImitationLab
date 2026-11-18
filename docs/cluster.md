# Running on a Slurm cluster

The cluster control plane submits ImitationLab stages to a Slurm cluster that
runs the Isaac Lab container image. It works in two steps. `plan` resolves a
campaign, runs preflight checks over ssh, and freezes the job scripts. `submit`
packs the working tree, uploads it, and submits the frozen plan only when you
confirm the plan hash.

```bash
python -m imitationlab.runners.pipeline.cluster plan \
  --campaign campaign.yaml --arm sweep --seed 42 --out-root outputs/cluster/plans
python -m imitationlab.runners.pipeline.cluster submit --plan PLAN_DIR --confirm PLAN_SHA
python -m imitationlab.runners.pipeline.cluster status --submission SUBMISSION_JSON
python -m imitationlab.runners.pipeline.cluster logs --submission SUBMISSION_JSON --stage STAGE
python -m imitationlab.runners.pipeline.cluster cancel --submission SUBMISSION_JSON --yes
```

`submit` writes the submission record (`submission-*.json`) into the plan
directory.

Submission packs the current working tree. To run a fixed revision while you
keep editing, plan and submit from a separate `git worktree` of that commit.

## Profiles

A profile names the login alias, the cluster paths, the container image, and
the Slurm resources. Profiles contain user directories, allocations, and
credential-file paths, so they stay out of Git. The package ships only the
commented [example profile](../imitationlab/runners/pipeline/cluster/conf/profile_example.yaml).

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

A campaign lists stages, each with an executable and arguments, plus preflight
paths that must exist inside the container. A learning-speed stage usually
runs `-m imitationlab benchmark run ...`; see
[training and evaluation](training.md#measure-learning-speed). Keep raw
results, checkpoints, and BONES-SEED-derived data on the cluster or under
`outputs/`, never in Git.
