# G1 learning speed: 30-minute runs

This completed H100 sweep includes all 24 combinations: PPO, SuperMimic (BPTT),
SVG and DiffSR, each with explicit, FSQ and spectral commands on LAFAN1 and
BONES-SEED. No combination is omitted. The [ten-minute study](g1-learning-speed/README.md)
and its follow-ups remain available separately.

All 28 jobs completed: four encoder jobs and 24 policy jobs. The independent
audit passed for all cases and all 144 evaluated checkpoints. See the
[complete timed results and figures](g1-learning-speed-30min/README.md).

Each policy starts from scratch with the current defaults and receives
1,800 cumulative training seconds. Checkpoints are requested at 60, 120, 300,
600, 900, 1,200 and 1,800 seconds. Saves occur at completed iteration boundaries;
tables use actual checkpoint clocks. Learner compilation and checkpoint costs
after the first collected batch count toward training. Policy startup, encoder
pretraining and evaluation are recorded separately; queue time is excluded.

PPO normalizes actor and critic inputs. H100 latent SVG/DiffSR uses the standard
profile; the other workloads retain automatic memory-based sizing. These
settings were selected before this sweep. Normal architectures, horizon 32 for
world-model learners, and the `imitation-g1-v1` task contract are preserved.
The four dataset/interface encoders are freshly pretrained for 2,000 updates
on the same source snapshot and then frozen for their policy runs.

| Setting | Value |
| --- | --- |
| Policy budget | 30 training minutes per combination; 12 GPU-hours total |
| Hardware | One NVIDIA H100 per job; 8 CPU cores; 96 GiB host-memory request |
| Scheduling | A Slurm cluster; 60-minute policy walltime limit |
| Seed | 42 |
| Motion banks | LAFAN1: 40 clips; BONES-SEED: 129,785 clips |
| Evaluation board | All 40 LAFAN1 clips; the same deterministic 256-clip BONES-SEED sample |
| Evaluation horizon | First 500 control steps (10 seconds), or natural clip completion |
| Evaluation behavior | Deterministic actions, frame-zero starts, no randomization |
| Compiler cache | Cold, separate cache for each job |

Success means no root-height or tilt failure through the evaluated prefix:
absolute root-height error at most 0.25 m and absolute root R[2,2] error at
most 0.8. Local/global MPJPE uses valid pretermination transitions across
the board. Interpret errors together with success and survival; a failing
run contributes shorter trajectories. First-observed quality thresholds use
saved checkpoints without interpolation. These are single-seed training-motion
measurements, not held-out generalization or full BONES-SEED coverage.

Time-to-target reporting accounts for float32 success-rate representation error
at exact boundaries, such as 38/40 being stored as 0.949999988 rather than 0.95.
The report recomputes thresholds from unchanged measurements with an absolute
tolerance of 1e-7, retains differing original threshold records, and independently
checks every crossing against integer success counts on the fixed board.
This reporting correction is in commit `d52954d`; the training source below
is unchanged.

The sweep uses commit `cb4194715e64557a24fb88da0313ff4c97995d80`, training-source
fingerprint `9756613231dd12379bf73cf6285026f165f74ce7bd8ba29a913137ba4f021eb2`,
and workspace archive SHA-256
`6e3beca1745d3ee93429af75e58682a332b57e42de263a7d2c776e96982bd2e8`.
The bank manifests, evaluation boards and pinned simulator/runtime match the
[ten-minute protocol](g1-learning-speed.md). Fresh encoder hashes, resolved
execution sizes, checkpoint hashes and actual clocks are retained with results.

All eight jobs accepted before an interrupted submission were retained. The
remaining twenty stages were submitted after verifying job identities,
dependencies and remote script/archive hashes. The recovered submission record
contains all 28 unique job IDs; no accepted job was duplicated or cancelled.
Policy walltime reservations were reduced from 90 to 60 minutes. This retains the full 30-minute training
budget and leaves additional time for startup and checkpoint evaluation.

The report audit requires positive policy/world-model/critic optimizer counters
as applicable, finite checkpoints, matching source/data/encoder identities and
evaluation boards, and completion of the full training budget. Raw motions,
per-motion evaluations and model weights stay outside Git. Only aggregate
results and figures are published. See [reproduction commands](../training.md#measure-learning-speed).
