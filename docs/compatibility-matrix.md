# Learner and interface compatibility

This check covers the native ImitationLab learners on `imitation-g1-v1`:
PPO, SuperMimic, SVG, and DiffSR, each with explicit, FSQ, and spectral command
interfaces. Each combination is run on LAFAN1 and BONES-SEED separately.
The optional official mjlab SuperMimic baseline uses a different task and is
outside this matrix.

## Results

All **24/24** learner/interface/dataset combinations passed, as did all four
encoder pretraining jobs. See the [README results](../README.md#results).
All checkpoint hashes were rechecked after the run; the saved learner classes,
task and data identities, execution modes, and optimizer step counters matched
the requested configuration. Evaluation metrics with nonzero sample counts
had finite means and standard deviations.

Evaluation executed 48–56 steps before all four environments terminated,
out of a requested maximum of 100. These are minimally updated smoke policies;
**the pass rate is a compatibility result, not a tracking success rate**.

The final run's commands, logs, checkpoints, evaluation JSON, incremental
`results.json`, and final `audit.json` are stored locally under ignored
`outputs/compatibility-matrix/2026-11-12-verified/`. Initial attempts that exposed
the memory issue remain separate under `outputs/compatibility-matrix/2026-11-12/`.
No proprietary motion data or model weights are included in this report.

## Protocol

Run the matrix with the Isaac Lab environment and two authorized local packed
banks. The runner starts one process at a time on the local GPU, uses the
public CLI, and saves incremental results and logs under `outputs/`.
It passes `--hardware off` to keep the protocol independent of automatic
hardware sizing on the machine running the check.

```bash
pixi run --locked -e isaaclab python scripts/validation/run_training_matrix.py \
  --lafan1 "$LAFAN1_BANK" --bones-seed "$BONES_SEED_BANK" \
  --output outputs/compatibility-matrix/my-run
```

Use a fresh output directory, or add `--resume` to retain verified passes and
retry incomplete cases with the same source commit, seed, and data identities.
Bank paths must contain a
`reference_arrays_manifest.json`. The runner does not download or publish
motion data. BONES-SEED requires separate authorization from Bones Studio;
raw data, checkpoints, per-motion evaluation output, and logs stay outside Git.

Each dataset gets its own FSQ and spectral encoder. Pretraining runs two updates
with batch size 16, four data environments, and one evaluation batch. Both
encoders retain the 64-dimensional command and ten-frame reference window.
For this bounded check, encoder hidden layers are `[64, 64]`, DiffSR feature
and embedding dimensions are 64, and the DiffSR f/g/mu hidden layers are
`[64, 64]`. These are smoke-test overrides; the normal recipes are unchanged.

Each tracker uses `--max-iterations 2` with 16 environments and seed 42.
That flag sets a rollout-frame budget: PPO completes two rollout updates
(768 transitions), while world-model learners complete one learner iteration
after their replay warmup (1,024 transitions).
PPO uses its normal model settings. World-model learners use replay warmup 64,
world-model and imagination batch sizes 32, horizon 4, world-model and actor
hidden layers `[64, 64, 64]`, and critic layers `[64, 64]`. Compilation is not
disabled. The historical smoke set `agent.wm.wm_warmup_iters=0` so latent runs
updated the actor and critic within this short budget. That override masked
a launcher bug: latent runs did not receive the shared fast learner defaults.
The current launcher applies those defaults, including zero model-only warmup,
to every command interface and data format. The launcher selects its execution mode: packed explicit
training uses the captured simulator step, while latent interfaces use eager
simulation. Dataset size is not reduced.

A passing case must finish training, save finite checkpoint tensors with
positive optimizer step counters (including actor and critic), record
the requested task/learner/interface/seed/data identity, and reload through
`imitationlab evaluate`. Latent trackers must record the exact pretrained
encoder hash. Evaluation starts bank ranks 0–3 at frame zero in four
environments, requests up to 100 control steps, and checks that rollout
metrics are finite and valid transitions were collected. Evaluation may
stop early when every environment terminates.

These checks establish that the components work together. A short smoke does not
produce a trained tracker, and early tracking failures are expected. A pass
does not establish motion-tracking quality, full-bank coverage, convergence,
or sustained throughput. Process timings include simulator startup and any
compiler cold start and should not be used to rank learners.

## Recorded inputs

The 2026-11-12 run uses source commit `82ab6c904c6ffda359b07141826e5591128f9b3f`,
an NVIDIA RTX A4500, Python 3.12.13, PyTorch 2.12.0+cu130,
TorchRL 0.13.3, TensorDict 0.13.0, and Newton/MJWarp. The task is
`imitation-g1-v1` and the seed is 42.

| Bank | Persist ID | Clips | Transitions |
| --- | --- | ---: | ---: |
| LAFAN1 | `lafan1@0ff4dce7ae91` | 40 | 441,080 |
| BONES-SEED SONIC selection | `bones_seed_sonic_full_129785@e714bbff` | 129,785 | 47,491,234 |

Bank manifest SHA-256 values:

- `lafan1`: `ae7b212a370e98823592ec0532adb427c948197f5406937f3e79af8d7568167f`
- `bones_seed`: `936799c1fd7621838b8ddbfd6d1230c0d1b4fbb142e3cd61b20843cc197610c5`

Both encoders completed two optimizer updates on each bank. The exported
encoder SHA-256 values below are also checked against each tracker run.

| Bank | Interface | Encoder SHA-256 |
| --- | --- | --- |
| lafan1 | fsq | `a56fa3d45aca8b2534e48261040ec4813292b43d6896017fe8d16ec756f0f915` |
| lafan1 | spectral | `480027a3a20b5b19fc398834e551a5f99bbe93dd1e6763f6c039f8b1b193b5dd` |
| bones_seed | fsq | `41396c85feb05e059ea3917deafaadd1def382d593a153c877f4e2ab8bc43ab4` |
| bones_seed | spectral | `b4e68911e47faf9a8d861efd93b95194a77b3c123da343acef77b2c01e82627d` |

## Issue found during verification

The initial full-bank BONES-SEED encoder check exhausted GPU memory because
the packed data loader tried to copy all 46 GiB of arrays to the GPU. The
unified CLI now memory-maps packed arrays on the CPU for pretraining and
all command interfaces. Packed G1 v1 latent trackers also use the host-backed
reference library and per-environment GPU windows. Explicit training recipes,
the observation/action contract, and checkpoint tensor layouts are unchanged.
Storage overrides remain available to callers.

The short smoke also set world-model warmup to zero and required positive
optimizer step counters. Full-budget testing later found latent runs ending
during the unintended 500-iteration model warmup, without an actor update.
The launcher now shares learner defaults across interfaces, and the learning
benchmark requires positive policy optimizer counters in the final checkpoint.
The historical smoke matrix above is not evidence for the corrected default
configuration's learning speed.

The CLI regression checks passed 50 tests. The runtime/evaluation CPU suite
passed 568 tests with two skips; these suites overlap. Static checks passed.
