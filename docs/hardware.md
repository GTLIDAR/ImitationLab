# Hardware selection

Fresh `imitationlab train` and `imitationlab pretrain` runs use `--hardware auto`.
The selector reads the selected CUDA device, free and total VRAM, available
host RAM, CPU affinity/container limits, and packed-bank dimensions. It chooses
execution sizes and compact reference-library placement. It leaves the task,
observation contract, encoder/network dimensions, horizon, precision, optimizer,
and learning rates unchanged. Batch sizes can affect learning, so compare
tracking quality separately from throughput.

## Inspect or override

```bash
pixi run -e isaaclab imitationlab hardware --policy supermimic --data "$BANK"
pixi run -e isaaclab imitationlab hardware --policy ppo --data "$BANK" --json
pixi run -e isaaclab imitationlab hardware --stage pretrain --interface fsq --data "$BANK"
pixi run -e isaaclab imitationlab train --policy svg --data "$BANK"
```

`--device cuda:1` selects a different **local** device; indices follow
`CUDA_VISIBLE_DEVICES`. No remote jobs are submitted. These commands require the
Isaac Lab Pixi environment for CUDA detection, but `hardware` does not start the
simulator. CLI help works without CUDA or Isaac Lab.

The same profiles apply to PPO, SuperMimic, SVG, and DiffSR, with explicit,
FSQ, or spectral commands. The policy-training candidates are:

| Profile | Environments | World-model / imagination batch | Estimated runtime VRAM before library and headroom |
| --- | ---: | ---: | ---: |
| `small` | 1,024 | 512 | 6 GiB |
| `standard` | 4,096 | 2,048 | 12 GiB |
| `large` | 8,192 | 4,096 | 25 GiB |
| `xlarge` | 16,384 | 8,192 | 48 GiB |

PPO uses the environment count and retains its existing minibatch settings.
Its rollout storage needs a larger budget: multiply the estimate by 2.0 for
explicit commands, or 1.6 for latent commands. For world-model learners, latent
interfaces multiply the estimate by 1.25; the DiffSR policy learner additionally
multiplies it by 1.15. These budgets are sizing heuristics, not measured
peak-memory guarantees. Custom network sizes and other overrides can exceed them.
Encoder pretraining uses batch sizes 32/128/256/512 for the respective profiles
and at most four data environments. Compiler workers are capped by CPU and RAM.

The selector reserves at least 1 GiB or 8% of total VRAM, whichever is larger,
from currently free memory. A packed G1 library requires 71 float32 values per
frame. It prefers GPU placement when a standard-or-larger resident profile fits;
otherwise it keeps the largest estimated feasible profile and uses host memory
with 64-frame GPU windows. If host RAM is insufficient, it can fall back to a
smaller resident profile. Full reference arrays remain CPU-mapped.

On H100 GPUs, automatic sizing caps SVG and DiffSR with latent commands at
`standard`. In the [600-second measurements](benchmarks/g1-learning-speed/README.md),
this repaired both stalled LAFAN1 DiffSR runs and improved LAFAN1 SVG. BONES-SEED
DiffSR also gained success; BONES-SEED SVG lost 3.5–6.6 percentage points. This
uses one consistent profile across both banks, not the best result per cell.
PPO retains its memory-based sizes: reducing its profile worsened all six
follow-up cases. Explicit profiles and measured tuning selections remain
available. Other GPU families retain memory-based estimates; H200 performance
has not been measured here.

For example, a mostly free 16 GiB GPU with 32 GiB available host RAM selects the
standard explicit configuration and host placement for full BONES-SEED. A small
LAFAN1 bank can stay on the GPU. Available memory matters more than the GPU name.
Unsupported or insufficient resources produce an actionable error, not a remote
fallback or an implicit CPU simulation run.

Use `--hardware small|standard|large|xlarge` to pin a profile; it still checks the
memory estimate. `--hardware off` retains fixed recipe settings without probing
the GPU. Explicit size overrides and arguments files take precedence:

```bash
pixi run -e isaaclab imitationlab train --policy supermimic --data "$BANK" \
  --num-envs 2048 agent.wm.wm_batch_size=1024 agent.wm.imagine_batch=1024
```

Training stores detected resources, the recommendation, and resolved execution
settings in `run.json` and checkpoint metadata. Pretraining records them in
`config.yaml`. Resume preserves saved settings; explicit overrides remain
available when intentionally resizing a resumed run. Multi-GPU launches retain
their existing settings rather than using auto sizing.
The optional official mjlab SuperMimic baseline has its own configuration.

## Measure candidate throughput

Calibration is optional. It starts fresh disposable runs sequentially on the
selected local GPU, measures completed training iterations after warmup, and
chooses the highest measured frames/second among passing candidates.

```bash
pixi run -e isaaclab imitationlab tune \
  --policy supermimic --interface explicit --data "$BANK" \
  --candidates small standard --seconds 30 --warmup-iterations 5 \
  --output outputs/tuning/my-pc

pixi run -e isaaclab imitationlab train \
  --policy supermimic --interface explicit --data "$BANK" \
  --tuning outputs/tuning/my-pc/selection.json
```

Omit `--candidates` to try all profiles whose estimates fit. For latent interfaces,
pass the same `--encoder` to both commands. Training overrides used during
calibration must also be supplied when reusing the selection. `--timeout` bounds
each trial (900 seconds by default); first-time compiler startup can take longer.
Run calibration while the machine is otherwise idle for representative timings.

Startup and the warmup iterations are excluded. At least two complete iterations
must be measured. Each trial must exit successfully, write finite checkpoint
tensors, and update every saved optimizer. World-model warmup is set to zero
in these disposable trials so actor and critic work is included. New public
G1 runs also use zero warmup through the shared fast defaults. Trials use seed
42 and local file logging.
Failures remain in `selection.json` and are excluded from ranking; a failed or
interrupted trial cannot supply a winner. Calibration currently covers policy
training, not encoder pretraining.

The saved selection is bound to source/recipe hashes, package versions, GPU/CPU
and total RAM, relevant process settings, bank manifest, encoder weights, learner,
interface, and workload overrides. Changed context requires recalibration;
reusing a selection also rechecks available memory and library placement.
The report, checkpoints, and logs stay under ignored `outputs/`. No data or
weights are uploaded. Keep BONES-SEED artifacts within authorized storage.

The result is the fastest **measured candidate for that workload and timing
window**, not a guarantee of global optimality, convergence, or tracking quality.
Short windows and concurrent activity can change the ranking.

## Local verification (2026-11-12)

On the local RTX A4500 (20 GiB VRAM, approximately 62.5 GiB host RAM), automatic
selection recommended `standard` for explicit SuperMimic on both release banks:
GPU-resident LAFAN1 references and host-backed full BONES-SEED references.
The CPU tests cover all 24 learner/interface/bank-size combinations, simulated
8–80 GiB GPU budgets, low available memory, override precedence, preserved
resume settings, and rejection of stale or unsuccessful calibration results.

A real two-candidate SuperMimic calibration used `imitation-g1-v1`, explicit
commands, seed 42, and `lafan1@0ff4dce7ae91` (40 clips, 441,080 frames). It used
the normal network sizes, simulator capture, and compiled/captured learner
updates. No encoder was used. Five warmup iterations preceded each measurement.

| Profile | Environments | WM / imagination batch | Measured iterations | Measured time | Frames/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| `small` | 1,024 | 512 | 42 | 5.078 s | 12,704 |
| `standard` | 4,096 | 2,048 | 21 | 5.165 s | **25,378** |

Both trials exited successfully and saved finite world-model, actor, and critic
state with positive optimizer counters. The selector chose `standard`. This is
a short functional calibration check, not a sustained benchmark or learning
quality comparison; the command's normal measurement window is 30 seconds.

Reproduce the check with `tune --policy supermimic --interface explicit
--data "$LAFAN1_BANK" --candidates small standard --seconds 5
--warmup-iterations 5 --assert-kitless --output outputs/tuning/local-check`.
Supply `--assert-kitless` again when reusing this selection. The local evidence
is under ignored `outputs/tuning/2026-11-12-supermimic-lafan1-v4/`; its selection
records source fingerprint
`44636b1679853daf48b5ff88430331ba9b762545d78ecdc660379754b3db3338` and exact
software/data identities. Other GPU sizes have sizing tests, not simulator
qualification on those devices.

PPO calibration also passed on the full proprietary BONES-SEED bank
(`bones_seed_sonic_full_129785@e714bbff`, 129,785 clips, 47,491,234 frames),
using explicit commands, seed 42, and host-backed references. Its `small`
profile used 1,024 environments and completed three measured updates in
2.003 seconds after one warmup update (36,810 frames/s). Checkpoint tensors
were finite and optimizer counters advanced. Evidence stays under ignored
`outputs/tuning/2026-11-12-ppo-bones-v2/`.

The initial 4,096-environment PPO attempt exhausted this card's VRAM while
copying full observation rollouts. That result motivated PPO's larger memory
budget; auto now chooses 1,024 environments here. This single-candidate check
validates the bounded run and stop/checkpoint path. Its speed is not comparable
to the SuperMimic row above because the learner and dataset differ.

Additional workflow checks passed with this source:

- Automatic encoder sizing selected four data environments and batch size 128
  for FSQ/LAFAN1 and DiffSR/BONES-SEED. Both completed two updates with finite
  checkpoints. These encoder checks used explicitly reduced 64-wide hidden
  layers, not the normal encoder network sizes.
- A fresh SuperMimic run reused the winning selection and saved its resolved
  4,096/2,048 settings. Its checkpoint reloaded through the public evaluation
  command on four clips, producing 165 valid transitions across 49 steps.
  All four barely trained policies terminated early; this verifies reload,
  not tracking quality.
- The documented `smoke-g1` command passed. `pixi run test-default` passed
  960 tests with two skips; `pixi run -e static ci-static` passed.

Workflow logs and the hash/metadata audit stay under ignored
`outputs/hardware-selection/`. BONES-SEED data and all model weights remain
outside Git.
