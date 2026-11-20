# Hardware selection: local verification (2026-11-12)

This is a dated record of checks of [hardware selection](../hardware.md) run on
one workstation. For measurements on four GPU tiers, see the
[G1 release check](g1-release-check.md).

On a workstation with an RTX A4500 (20 GiB VRAM, approximately 62.5 GiB host
RAM), automatic selection recommended `standard` for explicit SuperMimic on
both release banks: GPU-resident LAFAN1 references and host-backed full
BONES-SEED references.
The CPU tests cover all 24 learner/interface/bank-size combinations, simulated
8–80 GiB GPU budgets, low available memory, override precedence, preserved
resume settings, and rejection of stale or unsuccessful calibration results.

A two-candidate SuperMimic calibration on the simulator used
`imitation-g1-v1`, explicit commands, seed 42, and `lafan1@0ff4dce7ae91`
(40 clips, 441,080 frames). It used the normal network sizes, simulator
capture, and compiled/captured learner updates. No encoder was used. Five
warmup iterations preceded each measurement.

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
Supply `--assert-kitless` again when reusing this selection. The evidence is
under ignored `outputs/tuning/2026-11-12-supermimic-lafan1-v4/`; its
selection records source fingerprint
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
