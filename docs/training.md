# Training and evaluation

This guide covers the full workflow for a G1 motion tracker: choosing a
command interface and a policy learner, pretraining a command encoder,
training, evaluating, exporting, and measuring learning speed. It is for users
who have a working installation and want to train or compare trackers.

**Before you start.** Install the `isaaclab` environment
([setup](setup.md)), prepare a motion bank
([motion data](../data/README.md)), and run the smoke test on it. Run all commands from the repository root.

Terms used in this guide:

| Term | Meaning |
| --- | --- |
| Tracker | The trained policy that follows a reference motion |
| Policy learner (`--policy`) | The algorithm that updates the tracker |
| Command interface (`--interface`) | How the reference motion is presented to the tracker |
| Motion manifest | A JSON file that lists motion clips stored as NPZ files |
| Packed reference bank | A directory of memory-mapped arrays built from a manifest with `imitationlab data pack`; also called a motion bank |
| BPTT | Backpropagation through time |
| FSQ | Finite scalar quantization |

## Choose the interface and the learner

The command interface and the policy learner are independent choices. All
combinations use the `imitation-g1-v1` task on Newton/MJWarp. With no learner
or interface specified, training selects `supermimic` and `explicit`.

### Command interfaces

| `--interface` | Tracker input |
| --- | --- |
| `explicit` (default) | Ten frames of full-body reference features expressed relative to the robot |
| `fsq` | A frozen spectral skill encoder's 64 normalized scalar lattice values and two phase values with the supplied recipe |
| `spectral` | A frozen spectral skill encoder's 64 continuous values and two phase values with the supplied recipe |

`--interface diffsr` is the earlier name of `spectral` and still works.

The `fsq` and `spectral` interfaces are the latent interfaces. Both encoders
use ten 38-value root-and-joint frames at 50 Hz. Which frame these values are
expressed in is set by `env.expert_macro_anchor_mode`:

| Mode | Encoder window | Needed on the robot |
| --- | --- | --- |
| `reference_heading` (supplied recipe) | The explicit goal's root position, root orientation and joint positions. Frames start at the next control step and are expressed in the yaw frame and position of the reference frame the robot is on. Nothing is read from the robot | An IMU but no position estimate, as for the explicit interface |
| `robot_pos_heading` | Frames expressed in the live robot's yaw frame at the robot's x/y, which adds position feedback | Odometry |

Encoders from the first release recorded `robot_pos_heading` as
`robot_heading` and still load. Keep the encoder weights with the tracker
checkpoint.

### Policy learners

| `--policy` | Update |
| --- | --- |
| `supermimic` (default) | SuperMimic-style BPTT through imagined rollouts with a value bootstrap |
| `svg` | Pathwise gradients through the world model along recorded simulator windows, with noise inference and an imagined critic |
| `diffsr` | BPTT through imagined rollouts of a DiffSR spectral dynamics model of the low-level state, using its differentiable mean readout |
| `ppo` | Clipped policy updates from simulator rollouts |

`supermimic`, `svg`, and `diffsr` are the world-model learners. The `diffsr`
policy learner models low-level dynamics; it is separate from the spectral
command encoder.

See [setup](setup.md#official-supermimic) for the optional official mjlab
runner.

## Prepare motion data

Training reads either a motion manifest or a packed reference bank. The
examples below use two shell variables:

| Variable | Value |
| --- | --- |
| `MANIFEST` | A motion manifest, as written by `imitationlab data prepare` |
| `BANK` | A packed reference-bank directory, used for standard training |

The repository distributes no motion data. With LAFAN1 prepared at the
default location:

```bash
MANIFEST=data/lafan1/manifests/g1_lafan1_manifest.json
pixi run imitationlab data validate "$MANIFEST"
```

The [motion data guide](../data/README.md) covers BONES-SEED and LAFAN1
preparation and parallel packing. Use a separate evaluation set.

**Note.** BONES-SEED is proprietary and separately licensed. Obtain it from
Bones Studio and keep it in storage accessible only to authorized users.

## Pretrain a command encoder

This step applies to the `fsq` and `spectral` interfaces only. Set
`INTERFACE` to `fsq` or `spectral`.

```bash
INTERFACE=fsq
pixi run -e isaaclab imitationlab pretrain \
  --interface "$INTERFACE" --data "$MANIFEST" \
  --output-dir "outputs/encoders/$INTERFACE" \
  --num-updates 2000 --wandb-mode disabled
```

The output directory holds two checkpoints:

| File | Use |
| --- | --- |
| `checkpoints/encoder.pt` | Pass to the policy trainer with `--encoder` |
| `checkpoints/latest.pt` | Retain for encoder pretraining state |

With no output path specified, pretraining writes to
`outputs/encoders/<latent_mode>/<timestamp>/`.

Both encoders train the same spectral skill objective
(`experiments/recipes/encoders/g1.args`):

- A bilinear spectral energy φ(s, z) = g(z)ᵀF(s) ties the skill latent z to
  the motion transition.
- A predictor keeps z predictable one chunk ahead with a diffusion head over
  the next chunk's frames.
- SIGReg prevents collapse.

FSQ quantizes z to a normalized lattice; `spectral` keeps it continuous.
Pretraining samples reference windows from `imitation-g1-v1`'s data plane.

## Train a tracker

Train with the default learner and interface on a packed bank:

```bash
pixi run -e isaaclab imitationlab train \
  --task imitation-g1-v1 --policy supermimic --data "$BANK" --seed 42
```

Select `ppo`, `svg`, or `diffsr` to use a different learner.

To train with a latent interface, pass the pretrained encoder. This run uses a
frozen FSQ encoder with SVG and loads clips from a manifest:

```bash
pixi run -e isaaclab imitationlab train \
  --task imitation-g1-v1 --policy svg --interface fsq \
  --encoder outputs/encoders/fsq/checkpoints/encoder.pt \
  --data "$MANIFEST" --num-envs 256 --seed 42
```

Any interface can be paired with any learner; for example,
`--interface spectral --policy diffsr` pairs the continuous spectral skill
latent with the DiffSR dynamics learner.

### Run options

| Option | Effect |
| --- | --- |
| `--dry-run` | Print the resolved trainer command and exit |
| `--num-envs` | Set the environment count, for example for a smaller run |
| `agent.wm.imagine_batch`, `agent.wm.wm_batch_size` | Set the world-model learners' batch sizes |
| `agent.collector.total_frames` | Set a frame budget |
| `agent.save_interval` | Set the checkpoint interval |
| `agent.logger.log_dir` | Choose another output location |
| `--hardware off` | Keep fixed recipe sizes instead of automatic hardware selection |

Explicit overrides take precedence over hardware selection. For a short PPO
smoke that writes a checkpoint, set the checkpoint interval within the frame
budget.

### Run directory

Runs default to `outputs/training/<learner>/<task>/<run>/`. The directory
holds `training.log`, the resolved settings in `params/`, and the checkpoints
in `models/`.

### Resume a run

Resume from a checkpoint or its run directory. The launcher recovers the
original task, learner, interface, encoder, and execution sizes from the saved
settings. It does not repeat hardware selection on resume.

```bash
pixi run -e isaaclab imitationlab train --checkpoint "$RUN" --max-iterations 100
```

### Default configuration

The standard packed-bank configuration uses 4,096 environments and a captured
G1 step for all four learners with explicit commands. Across explicit, FSQ,
and spectral commands, the world-model learners share batches of 2,048, the
same model and optimizer defaults, and zero world-model-only warmup
iterations. Policy updates start once the initial replay is collected. These
learner defaults also apply when training from an NPZ manifest.

Automatic hardware selection adjusts these execution sizes and
reference-library placement to the selected GPU and available host memory.
Use `--hardware off` to keep fixed recipe sizes, or inspect and calibrate a
configuration with the [hardware commands](hardware.md).

SuperMimic adds structured MLPs and captured learner updates. The DiffSR
learner fits its dynamics with Adam, combining denoising and a multi-step
readout loss. Training from an NPZ manifest or with a latent command interface
uses eager simulation by default.

### Reference data in memory

With `--data` pointing to a packed bank, all interfaces and encoder
pretraining memory-map the full arrays on the CPU. G1 v1 trackers also keep
the compact reference library on the GPU when space permits. Larger banks use
host memory with per-environment windows on the GPU, so the full BONES-SEED
bank need not fit in GPU memory. The full BONES-SEED bank needs about 12.6 GiB
of pinned host memory for the reference library alone when using host
placement.

### Learner-specific options

**PPO observation normalization.** PPO normalizes actor and critic
observations for new `imitation-g1-v1` runs. The task exposes unscaled
features; normalization statistics are frozen during each rollout and
optimization cycle, updated afterward, and saved with the checkpoint. Resuming
an existing run restores its original normalization and network settings.

**SVG foot-contact term (experimental).** SVG can opt into the foot-contact
training term used by the SuperMimic recipe with
`agent.wm.contact_reward=0.25 agent.wm.ref_contact_fix=true`. It applies to
imagined and recorded-window learner rewards; simulator rewards and evaluation
stay unchanged. Recorded terminal transitions retain their measured
environment reward because the next replay state belongs to the reset episode.
This option remains off by default.

**SVG CUDA graph updates (experimental).** Pathwise SVG with its default
imagined critic can also opt into CUDA graph updates with
`agent.wm.no_sync=true agent.wm.fused_adam=true agent.wm.update_graphs=true
agent.wm.graph_early=true`. Recent replay-window sampling follows the GPU
write head as the buffer fills and wraps. Early capture uses a different
random sampling stream from the ordinary loop. Recorded critics and REBAR
remain unsupported with these options. Tracking improvements must be measured
before changing the defaults.

## Evaluate a checkpoint

Evaluation reuses the saved network and task settings, pins clips at frame
zero, disables randomization, and uses the task's root height and tilt
termination rule.

```bash
pixi run -e isaaclab imitationlab evaluate \
  --checkpoint "$RUN" --data "$EVAL_BANK" \
  --num-envs 4 --steps 500 --seed 42 \
  --output outputs/evaluation/tracker.json --trajectory-ranks 0 1 2 3
```

| Option | Effect |
| --- | --- |
| `--data` | Score a separate packed bank or manifest instead of the saved one |
| `--trajectory-ranks` | Evaluate specific clips by rank. Set the environment count to a multiple of the number of ranks |
| `--assert-kitless` | Run numerical Newton evaluation without a Kit renderer. The learning-speed benchmark uses this mode |
| `--overwrite` | Replace an existing result file at `--output` |

Use matching data, ranks, and horizons for comparisons.

**Older checkpoints.** New checkpoints carry the launch settings. For an older
checkpoint, keep the original run directory with its `command.txt` or
`run.json`. For an older standalone checkpoint with no recorded settings, pass
`--policy`, `--data`, and the original architecture and encoder overrides.

## Export a checkpoint

Export a native bundle from a run directory or checkpoint. The bundle contains
the training checkpoint, recorded settings, file hashes, and the encoder
weights when applicable. Supply the data separately when evaluating the bundle
on another machine.

```bash
pixi run imitationlab export --checkpoint "$RUN" --output outputs/bundles/g1
pixi run -e isaaclab imitationlab evaluate \
  --checkpoint outputs/bundles/g1 --data "$EVAL_BANK" --num-envs 4
```

See [release verification](release-verification.md) for the first-release
dataset identities and smoke settings.

## Measure learning speed

`imitationlab benchmark` measures tracking quality against training time. It
trains once, saves timed checkpoints, and evaluates each on a board: a fixed
set of clips from the training bank. It uses normal architectures and hardware
sizing.

| Subcommand | Purpose |
| --- | --- |
| `benchmark pretrain` | Pretrain a command encoder and record its training time and weights |
| `benchmark run` | Train a policy and evaluate its milestone checkpoints |
| `benchmark report` | Render aggregate tables and curves from recorded cases |

For latent interfaces, run `benchmark pretrain` first and pass its output
directory as `--encoder-run`. The explicit interface needs no encoder.

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

Select `--dataset bones_seed` for the other bank.

### Budget and milestones

The default budget is 30 policy-training minutes, with checkpoints requested
after 1, 2, 5, 10, 15, 20, and 30 minutes. Encoder pretraining and startup are
additional costs and are reported separately. Pass both `--seconds` and
`--milestones` for a shorter run.

The clock starts after the first collected batch; compilation after that point
counts toward training. Checkpoints are saved after complete iterations, so
their actual times can exceed requested milestones. Ordinary `train` also
accepts `agent.trainer.checkpoint_seconds=[60,120,300,600]`.

### Evaluation board

The default board uses up to 256 clips sampled with the run seed (42 by
default), evaluated from frame zero for up to 10 seconds. Thus all 40 LAFAN1
clips are included. Reports include success, local and global pose errors
(MPJPE-L and MPJPE-G, the mean per-joint position errors), survival time, and
first observed quality thresholds.

**Note.** This measures learning on training motions, not held-out
generalization.

### Comparing seeds and reusing encoders

For comparisons across policy training seeds, pass `--eval-seed 42` to keep
the evaluation population and evaluation seed fixed.

To reuse an existing frozen encoder while varying the policy seed or recipe
source, explicitly pass its recorded `--encoder-seed` and
`--encoder-source-sha256` from `pretrain.json`. Without those overrides, both
must match the policy seed and current source. The benchmark always checks the
encoder's interface, bank identity and weight hash, and records its separate
pretraining provenance.

Render each recipe/training-seed group separately; duplicate
dataset/interface/learner cases are rejected by the report command.

Keep all raw results and checkpoints outside Git, especially for BONES-SEED.

## Next steps

- [Hardware selection](hardware.md): automatic execution sizes and throughput
  calibration.
- [Results](README.md#results): measured reports produced with these commands.
