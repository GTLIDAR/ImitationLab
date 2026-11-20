# Hardware selection

This guide explains how ImitationLab chooses execution sizes (environment
count, batch sizes, and where the reference library is stored) for the local
GPU, how to inspect or override that choice, and how to measure throughput. It
is for users who train on a GPU other than the ones in the published results
or who want to pin sizes by hand.

**Before you start.** Install the `isaaclab` environment ([setup](setup.md))
and have a packed reference bank, called `$BANK` below
([motion data](../data/README.md)).

## How automatic selection works

Fresh `imitationlab train` and `imitationlab pretrain` runs use
`--hardware auto`. The selector reads the selected CUDA device, free and total
VRAM, available host RAM, CPU affinity/container limits, and packed-bank
dimensions. It chooses execution sizes and compact reference-library
placement.

It leaves the task, observation contract, encoder/network dimensions, horizon,
precision, optimizer, and learning rates unchanged.

**Note.** Batch sizes can affect learning, so compare tracking quality
separately from throughput.

## Inspect the recommendation

```bash
pixi run -e isaaclab imitationlab hardware --policy supermimic --data "$BANK"
pixi run -e isaaclab imitationlab hardware --policy ppo --data "$BANK" --json
pixi run -e isaaclab imitationlab hardware --stage pretrain --interface fsq --data "$BANK"
pixi run -e isaaclab imitationlab train --policy svg --data "$BANK"
```

The first three commands print the recommendation without starting the
simulator; the last trains with it. `--device cuda:1` selects a different
**local** device; indices follow `CUDA_VISIBLE_DEVICES`. No remote jobs are
submitted. These commands require the Isaac Lab Pixi environment for CUDA
detection. CLI help works without CUDA or Isaac Lab.

## Profiles

The same profiles apply to all four policy learners (PPO, SuperMimic, SVG, and
DiffSR) with explicit, FSQ, or spectral commands. The policy-training
candidates are:

| Profile | Environments | World-model / imagination batch | Estimated runtime VRAM before library and headroom |
| --- | ---: | ---: | ---: |
| `small` | 1,024 | 512 | 6 GiB |
| `standard` | 4,096 | 2,048 | 12 GiB |
| `large` | 8,192 | 4,096 | 25 GiB |
| `xlarge` | 16,384 | 8,192 | 48 GiB |

The estimate is scaled by learner and interface:

| Case | Multiplier on the estimate |
| --- | --- |
| PPO, explicit commands | 2.0 |
| PPO, latent commands | 1.6 |
| World-model learners, latent commands | 1.25 |
| DiffSR policy learner | An additional 1.15 |

PPO uses the environment count and retains its existing minibatch settings;
its rollout storage is why it needs a larger budget. These budgets are sizing
heuristics, not measured peak-memory guarantees. Custom network sizes and
other overrides can exceed them.

Encoder pretraining uses batch sizes 32/128/256/512 for the respective
profiles and at most four data environments. Compiler workers are capped by
CPU and RAM.

### Memory headroom and library placement

The selector reserves at least 1 GiB or 8% of total VRAM, whichever is larger,
from currently free memory. A packed G1 library requires 71 float32 values per
frame. Placement is chosen in this order:

1. GPU placement, when a standard-or-larger resident profile fits.
2. Otherwise, the largest estimated feasible profile with the library in host
   memory and 64-frame GPU windows.
3. If host RAM is insufficient, a smaller resident profile.

Full reference arrays remain CPU-mapped in every case.

For example, a mostly free 16 GiB GPU with 32 GiB available host RAM selects
the standard explicit configuration and host placement for full BONES-SEED. A
small LAFAN1 bank can stay on the GPU. Available memory matters more than the
GPU name. Unsupported or insufficient resources produce an actionable error,
not a remote fallback or an implicit CPU simulation run.

### H100 cap for latent SVG and DiffSR

On H100 GPUs, automatic sizing caps SVG and DiffSR with latent commands at
`standard`. In the
[600-second measurements](benchmarks/g1-learning-speed/README.md), this
repaired both stalled LAFAN1 DiffSR runs and improved LAFAN1 SVG. BONES-SEED
DiffSR also gained success; BONES-SEED SVG lost 3.5–6.6 percentage points.
This uses one consistent profile across both banks, not the best result per
cell.

PPO retains its memory-based sizes: reducing its profile worsened all six
follow-up cases. Explicit profiles and measured tuning selections remain
available. Other GPU families retain memory-based estimates; for measured
runs on an H200 and three other GPU tiers, see the
[G1 release check](benchmarks/g1-release-check.md).

## Override the selection

| Setting | Effect |
| --- | --- |
| `--hardware auto` (default) | Choose sizes from local resources |
| `--hardware small\|standard\|large\|xlarge` | Pin a profile; the memory estimate is still checked |
| `--hardware off` | Retain fixed recipe settings without probing the GPU |
| Explicit size overrides and arguments files | Take precedence over the selection |

```bash
pixi run -e isaaclab imitationlab train --policy supermimic --data "$BANK" \
  --num-envs 2048 agent.wm.wm_batch_size=1024 agent.wm.imagine_batch=1024
```

## What is recorded

Training stores detected resources, the recommendation, and resolved execution
settings in `run.json` and checkpoint metadata. Pretraining records them in
`config.yaml`.

Resume preserves saved settings; explicit overrides remain available when
intentionally resizing a resumed run. Multi-GPU launches retain their existing
settings rather than using auto sizing. The optional official mjlab SuperMimic
baseline has its own configuration.

## Measure candidate throughput

Calibration with `imitationlab tune` is optional. It starts fresh disposable
runs sequentially on the selected local GPU, measures completed training
iterations after warmup, and chooses the highest measured frames/second among
passing candidates. Pass the saved selection to `train` with `--tuning`:

```bash
pixi run -e isaaclab imitationlab tune \
  --policy supermimic --interface explicit --data "$BANK" \
  --candidates small standard --seconds 30 --warmup-iterations 5 \
  --output outputs/tuning/my-pc

pixi run -e isaaclab imitationlab train \
  --policy supermimic --interface explicit --data "$BANK" \
  --tuning outputs/tuning/my-pc/selection.json
```

Usage notes:

- Omit `--candidates` to try all profiles whose estimates fit.
- For latent interfaces, pass the same `--encoder` to both commands.
- Training overrides used during calibration must also be supplied when
  reusing the selection.
- `--timeout` bounds each trial (900 seconds by default); first-time compiler
  startup can take longer.
- Run calibration while the machine is otherwise idle for representative
  timings.

### What a trial measures

Startup and the warmup iterations are excluded. At least two complete
iterations must be measured. Each trial must exit successfully, write finite
checkpoint tensors, and update every saved optimizer.

World-model warmup is set to zero in these disposable trials so actor and
critic work is included. New public G1 runs also use zero warmup through the
shared fast defaults. Trials use seed 42 and local file logging.

Failures remain in `selection.json` and are excluded from ranking; a failed or
interrupted trial cannot supply a winner. Calibration currently covers policy
training, not encoder pretraining.

### When a selection can be reused

The saved selection is bound to source/recipe hashes, package versions,
GPU/CPU and total RAM, relevant process settings, bank manifest, encoder
weights, learner, interface, and workload overrides. Changed context requires
recalibration; reusing a selection also rechecks available memory and library
placement.

The report, checkpoints, and logs stay under ignored `outputs/`. No data or
weights are uploaded. Keep BONES-SEED artifacts within authorized storage.

**Note.** The result is the fastest **measured candidate for that workload and
timing window**, not a guarantee of global optimality, convergence, or
tracking quality. Short windows and concurrent activity can change the
ranking.

## Verification

A dated record of the selection and calibration checks run on one workstation
is in
[Hardware selection: local verification](benchmarks/hardware-selection-verification.md).
For measurements on four GPU tiers, see the
[G1 release check](benchmarks/g1-release-check.md).

## Related guides

- [Training and evaluation](training.md): the commands these sizes apply to.
