# Training and evaluation

## Choose the command and the learner

| `--interface` | Tracker input |
| --- | --- |
| `explicit` | Ten frames of full-body reference features expressed relative to the robot |
| `fsq` | A frozen spectral skill encoder's 64 normalized scalar lattice values and two phase values with the supplied recipe |
| `spectral` | A frozen spectral skill encoder's 64 continuous values and two phase values with the supplied recipe |

`--interface diffsr` is the earlier name of `spectral` and still works. Both encoders use ten 38-value root-and-joint frames at 50 Hz in the robot's
heading frame. The tracker encodes reference frames starting at the next
control step. Keep the encoder weights with the tracker checkpoint.

| `--policy` | Update |
| --- | --- |
| `ppo` | Clipped policy updates from simulator rollouts |
| `supermimic` | SuperMimic-style BPTT through imagined rollouts with a value bootstrap |
| `svg` | Pathwise gradients through the world model along recorded simulator windows, with noise inference and an imagined critic |
| `diffsr` | BPTT through imagined rollouts of a DiffSR spectral dynamics model of the low-level state, using its differentiable mean readout |

All combinations use `imitation-g1-v1` on Newton/MJWarp. With no learner or
interface specified, training selects SuperMimic and explicit commands.
See [setup](setup.md#official-supermimic) for the optional official mjlab runner.

## Prepare motion data

Use the bundled Dance102 manifest to check an installation. Set `BANK` to a
packed reference-bank directory for standard training.

```bash
MANIFEST=data/unitree/manifests/g1_unitree_dance102_manifest.json
pixi run imitationlab data validate "$MANIFEST"
```

The [motion data guide](../data/README.md) covers BONES-SEED and LAFAN1
preparation and parallel packing. Use a separate evaluation set.
BONES-SEED is proprietary and separately licensed. Obtain it from Bones
Studio and keep it in storage accessible to authorized collaborators.

## Pretrain a command encoder

This step applies to the FSQ and spectral interfaces. Set `INTERFACE` to
`fsq` or `spectral`.

```bash
INTERFACE=fsq
pixi run -e isaaclab imitationlab pretrain \
  --interface "$INTERFACE" --data "$MANIFEST" \
  --output-dir "outputs/encoders/$INTERFACE" \
  --num_updates 2000 --wandb_mode disabled
```

Both encoders train the same spectral skill objective
(`experiments/recipes/encoders/g1.args`). A bilinear spectral energy
φ(s, z) = g(z)ᵀF(s) ties the skill latent z to the motion transition, a
predictor keeps z predictable one chunk ahead with a diffusion head over the
next chunk's frames, and SIGReg prevents collapse. FSQ quantizes z to a
normalized lattice; `spectral` keeps it continuous. This command encoder is
separate from the `diffsr` policy learner, which models low-level dynamics.
Pretraining samples reference windows from `imitation-g1-v1`'s data plane.
Pass `checkpoints/encoder.pt` to the policy trainer and retain
`checkpoints/latest.pt` for encoder pretraining state. With no output path
specified, pretraining writes to `outputs/encoders/<latent_mode>/<timestamp>/`.

## Train a tracker

The standard packed-bank configuration uses 4,096 environments and a captured
G1 step for all four learners with explicit commands. Across explicit, FSQ,
and spectral commands, the world-model learners share batches of 2,048,
the same model and optimizer defaults, and zero world-model-only warmup
iterations. Policy updates start once the initial replay is collected.
These learner defaults also apply when training from an NPZ manifest.
Automatic hardware selection adjusts these execution sizes
and reference-library placement to the selected GPU and available host memory.
Use `--hardware off` to keep fixed recipe sizes, or inspect and calibrate a
configuration with the [hardware commands](hardware.md).
SuperMimic adds structured MLPs and captured learner updates. The DiffSR
learner fits its dynamics with Adam, combining denoising and a multi-step readout loss.
Training from an NPZ manifest or with a learned command interface uses eager
simulation by default.

SVG can opt into the foot-contact training term used by the SuperMimic recipe
with `agent.wm.contact_reward=0.25 agent.wm.ref_contact_fix=true`. It applies to
imagined and recorded-window learner rewards; simulator rewards and evaluation
stay unchanged. Recorded terminal transitions retain their measured environment
reward because the next replay state belongs to the reset episode. This option
is experimental for SVG and remains off by default.

Pathwise SVG with its default imagined critic can also opt into CUDA graph
updates with `agent.wm.no_sync=true agent.wm.fused_adam=true
agent.wm.update_graphs=true agent.wm.graph_early=true`. Recent replay-window
sampling follows the GPU write head as the buffer fills and wraps. Early
capture uses a different random sampling stream from the ordinary loop.
Recorded critics and REBAR remain unsupported with these options. Captured
SVG is experimental; tracking improvements must be measured before changing
the defaults.

PPO normalizes actor and critic observations for new `imitation-g1-v1` runs.
The task exposes unscaled features; normalization statistics are frozen during
each rollout and optimization cycle, updated afterward, and saved with the
checkpoint. Resuming an existing run restores its original normalization and
network settings.

With `--data` pointing to a packed bank, all interfaces and encoder
pretraining memory-map the full arrays on the CPU. G1 v1 trackers also keep
the compact reference library on the GPU when space permits. Larger banks
use host memory with per-environment windows on the GPU, so the full BONES-SEED
bank need not fit in GPU memory.

```bash
pixi run -e isaaclab imitationlab train \
  --task imitation-g1-v1 --policy supermimic --data "$BANK" --seed 42
```

Select `ppo`, `svg`, or `diffsr` to use a different learner. For a smaller
run, set `--num-envs`. World-model runs also accept `agent.wm.imagine_batch`
and `agent.wm.wm_batch_size`; explicit overrides take precedence over hardware
selection. The full BONES-SEED bank needs about 12.6 GiB of pinned host memory
for the reference library alone when using host placement.

This run uses a frozen FSQ encoder with SVG and loads clips from a manifest.

```bash
pixi run -e isaaclab imitationlab train \
  --task imitation-g1-v1 --policy svg --interface fsq \
  --encoder outputs/encoders/fsq/checkpoints/encoder.pt \
  --data "$MANIFEST" --num-envs 256 --seed 42
```

The command interface and the learner are independent; for example,
`--interface spectral --policy diffsr` pairs the continuous spectral skill
latent with the DiffSR dynamics learner.

Use `--dry-run` to print the resolved trainer command. Use
`agent.collector.total_frames` to set a frame budget and
`agent.save_interval` to set the checkpoint interval. For a short PPO smoke
that writes a checkpoint, set the interval within the frame budget. Runs
default to `outputs/training/<learner>/<task>/<run>/`. The directory holds
`training.log`, the resolved settings in `params/`, and the checkpoints in
`models/`. Set `agent.logger.log_dir` to choose another location.

Resume from a checkpoint or its run directory. The launcher recovers the
original task, learner, interface, encoder, and execution sizes from the saved
settings. It does not repeat hardware selection on resume.

```bash
pixi run -e isaaclab imitationlab train --checkpoint "$RUN" --max-iterations 100
```

## Evaluate a checkpoint

Evaluation reuses the saved network and task settings, pins clips at frame
zero, disables randomization, and uses the task's root height and tilt
termination rule. Pass `--data` to score a separate packed bank or manifest.
For numerical Newton evaluation without a Kit renderer, pass `--assert-kitless`.
The learning-speed benchmark uses this mode on both workstations and ICE.
Use matching data, ranks, and horizons for comparisons. When evaluating
specific ranks, set the environment count to a multiple of the number of
ranks.

```bash
pixi run -e isaaclab imitationlab evaluate \
  --checkpoint "$RUN" --data "$EVAL_BANK" \
  --num-envs 4 --steps 500 --seed 42 \
  --output outputs/evaluation/tracker.json --trajectory_ranks 0 1 2 3
```

New checkpoints carry the launch settings. For an older checkpoint, keep
the original run directory with its `command.txt` or `run.json`. For an
older standalone checkpoint with no recorded settings, pass `--policy`,
`--data`, and the original architecture and encoder overrides.

## Export a checkpoint

Export a native bundle from a run directory or checkpoint. The bundle
contains the training checkpoint, recorded settings, file hashes, and the
encoder weights when applicable. Supply the data separately when evaluating
the bundle on another machine.

```bash
pixi run imitationlab export --checkpoint "$RUN" --output outputs/bundles/g1
pixi run -e isaaclab imitationlab evaluate \
  --checkpoint outputs/bundles/g1 --data "$EVAL_BANK" --num-envs 4
```

See [release verification](release-verification.md) for the first-release
dataset identities and smoke settings.

## Measure learning speed

Use `benchmark run` to train once, save timed checkpoints, and evaluate each
on a fixed motion board. It uses normal architectures and hardware sizing.
The default budget is 30 policy-training minutes, with checkpoints requested
after 1, 2, 5, 10, 15, 20, and 30 minutes. Encoder pretraining and startup
are additional costs. Pass both `--seconds` and `--milestones` for a shorter run.
For latent commands, first use `benchmark pretrain` to record encoder training
time and weights; pass that output directory as `--encoder-run`.

```bash
pixi run -e isaaclab imitationlab benchmark pretrain \
  --dataset lafan1 --interface fsq --data "$BANK" \
  --output outputs/learning-speed/encoders/lafan1-fsq --updates 2000
pixi run -e isaaclab imitationlab benchmark run \
  --dataset lafan1 --policy supermimic --interface fsq --data "$BANK" \
  --encoder-run outputs/learning-speed/encoders/lafan1-fsq \
  --output outputs/learning-speed/cases/lafan1-supermimic-fsq \
  --seconds 1800 --milestones 60 120 300 600 900 1200 1800
pixi run -e isaaclab imitationlab benchmark report \
  --input outputs/learning-speed/cases --output outputs/learning-speed/report
```

Select `bones_seed` for the other bank. Explicit commands need no encoder.
The default board uses up to 256 clips sampled with the run seed (42 by default),
evaluated from frame zero for up to 10 seconds. Thus all 40 LAFAN1 clips are included.
This measures learning on training motions, not held-out generalization.
Reports include success, local/global pose errors, survival time, and first
observed quality thresholds. Startup and encoder costs are separate.

For comparisons across policy training seeds, pass `--eval-seed 42` to keep
the evaluation population and evaluation seed fixed. To reuse an existing
frozen encoder while varying the policy seed or recipe source, explicitly pass
its recorded `--encoder-seed` and `--encoder-source-sha256` from `pretrain.json`.
Without those overrides, both must match the policy seed and current source.
The benchmark always checks the encoder's interface, bank identity and weight
hash, and records its separate pretraining provenance. Render each
recipe/training-seed group separately; duplicate dataset/interface/learner
cases are rejected by the report command.

The clock starts after the first collected batch; compilation after that
point counts toward training. Checkpoints are saved after complete iterations,
so their actual times can exceed requested milestones. Ordinary `train` also
accepts `agent.trainer.checkpoint_seconds=[60,120,300,600]`. Keep all raw
results and checkpoints outside Git, especially for BONES-SEED.
