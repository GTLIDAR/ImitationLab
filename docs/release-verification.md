# First release verification

The first release provides tooling for LAFAN1 G1 and the SONIC-filtered
BONES-SEED G1 bank. Training data includes [Motion Data by Bones Studio](https://bones.studio/).
See the [motion data guide](../data/README.md#bones-seed-g1-conversion) for
authorized access and local storage.
The checks below exercise motion ingestion, one optimizer iteration, checkpoint
reload, and a short simulator rollout on a local RTX A4500. They measure
integration. Tracking quality needs trained checkpoints and a held-out
evaluation manifest.

| Packed bank | Persist ID | Clips | Transitions | Explicit learners exercised |
| --- | --- | ---: | ---: | --- |
| LAFAN1 G1 from existing NPZs | `lafan1@276b08c82ed5` | 40 | 441,080 | PPO, SuperMimic, SVG |
| LAFAN1 G1 freshly converted | `lafan1@0ff4dce7ae91` | 40 | 441,080 | PPO, SuperMimic, SVG |
| BONES-SEED language subset | `bones_seed_language10_v1@60a5b7a5` | 10 | 5,147 | PPO, SuperMimic, SVG |
| BONES-SEED SONIC selection | `bones_seed_sonic_full_129785@e714bbff` | 129,785 | 47,491,234 | PPO, SuperMimic, SVG |

All 40 LAFAN1 CSVs were converted again from 30 Hz to 50 Hz and packed.
The packer compared the first, middle, and last rows of five trajectories to
their source NPZs. The two LAFAN1 banks had the same clip ranges, body order,
and joint order. Their floating-point values differed slightly, so each bank
was tested under its own identity. The BONES-SEED selector reproduced the 129,785-motion
selection with SHA-256 `e714bbff883dc82e3cf05216cfe6347e8b6a6bc3a4eadd4648c20ee3eb6867d6`.
The BONES-SEED training checks used existing packed banks. The raw G1
archive was unavailable locally. One FSQ encoder update on LAFAN1 and one
continuous DiffSR encoder update on the BONES-SEED subset each produced an
encoder that PPO, SuperMimic, and SVG loaded for a training iteration.

The DiffSR learner completed a training iteration on freshly converted
LAFAN1 with explicit commands and with a newly pretrained DiffSR encoder.
The latter checkpoint reloaded for a 20-step evaluation with the same
encoder. The explicit run also passed with PyTorch compilation enabled.
These runs used 16 environments and reduced model and batch sizes.

## Unified package and CLI verification

The `imitationlab` package migration was checked on the local RTX A4500 with
`imitation-g1-v1`, seed 42, and the freshly converted LAFAN1 bank
`lafan1@0ff4dce7ae91`. All training and encoder recipe files were compared
byte for byte with the source snapshot taken before this migration and were
unchanged.

The default CPU task passed 891 tests with two skips. The Isaac Lab task
suite passed 381 tests. After the final CLI and script-wrapper changes, the
affected launcher and checkpoint tests passed another 369 tests. These
suites overlap and should not be added into a single coverage count. Static
checks and the package wheel build passed. The wheel's CLI and nested recipes
were also checked outside the source checkout.

| Local check | Result |
| --- | --- |
| PPO, SuperMimic, SVG, DiffSR with explicit commands | Each completed an optimizer update and wrote a checkpoint; the three world-model choices also exercised their native public learner classes |
| SuperMimic and SVG with compilation enabled | Each completed an update with the packed graph environment and compiler enabled; SuperMimic used its native public learner class |
| Pre-migration checkpoints | All four learners loaded and ran 20 evaluation steps; PPO resumed to the next rollout iteration |
| FSQ and DiffSR encoder pretraining | Each completed one update; FSQ/PPO and DiffSR/DiffSR trackers trained and reloaded for evaluation |
| Captured-step equivalence | Masked and captured steps matched bit for bit over 210 replayed-physics steps at 256 environments, including resets and the reference window |
| Native export | Preserved checkpoint tensors and resolved a copied encoder after moving the bundle |
| Motion data | Validated the 40-clip LAFAN1 bank and 129,785-clip BONES-SEED bank; repacked two real LAFAN1 clips with two workers and checked source equality |

The encoder artifacts used by the latent tracking checks have SHA-256:

- FSQ: `1d96a10540d1b7fc45462d7aa76ced495cdb83cf78c9afe4c91193d169676c9a`
- DiffSR: `18bba9249824ed22c209a743193b73f8b7571bfef13ad716dd77843b72b66e43`

Commands, run descriptions, evaluation JSON, and test logs are saved locally
under ignored `outputs/reorganization/`. These are integration and compatibility
checks using reduced batches. They do not establish convergence or sustained
training throughput. BONES-SEED validation used the existing authorized local
bank; its raw archive was not re-converted or distributed.

After moving the learners and data tools into ImitationLab, all four
learners completed another `imitation-g1-v1` explicit-command update on
`lafan1@0ff4dce7ae91` with seed 42. The evaluator reloaded a pre-migration
checkpoint for each learner and ran 20 steps. The native DiffSR pretrainer
produced the same encoder file as the pre-migration smoke
(`fff37d4d61a21e6ea9e0d04f5e23a7386e0ee62516d3f480458509bdcc02938e` SHA-256).
The native DiffSR learner also completed a training iteration with that
continuous `z` command. SVG completed a packed training iteration with
PyTorch compilation enabled. PPO resumed from its pre-migration checkpoint.

## Single-task release verification

The single-task release registers only `imitation-g1-v1` and removes the
earlier task designs, their aliases and the ONNX exporter. The
[migration guide](migration.md) lists every removal and rename. The checks
below ran on a local NVIDIA RTX PRO 6000 Blackwell Workstation Edition in the
locked `default`, `isaaclab` and `static` environments, with seed 42, the
LAFAN1 bank `g1_lafan1_40@d972c37c41da` and the BONES-SEED bank
`bones_seed_sonic_full_129785@e714bbff`.

The environment step was compared with the commit before the change. That
commit recorded the physics of 200 steps at 256 environments, about 1,250
resets; the release replayed it. Observations, terminal rows, rewards, dones,
reward parts, histories, reference cursors, sampler state and random-number
states matched bit for bit for the explicit and spectral eager steps, the
masked step, and the captured step. Checkpoints trained before the change
evaluated with the same step counts and with metrics within about 1e-4, the
size of the difference between two evaluations of one checkpoint.

| Check | Result |
| --- | --- |
| `pixi run --locked -e static ci-static` | Passed |
| `git ls-files -z '*.sh' \| xargs -0 -n1 bash -n` | Passed |
| `pixi run --locked test-default` | 1,680 passed, 21 skipped, 1 failed (below) |
| `pixi run --locked -e isaaclab test-isaaclab` | 482 passed, 18 skipped |
| Opt-in step equivalence, `IMITATION_G1_EQUIV_BANK` set | 23 passed; the statistics test also needs `IMITATION_G1_GRAPH_STATS=1` |
| `pixi run -e isaaclab smoke-g1` | Passed |
| Learner and interface smokes | 12 of 12 trained, reloaded, exported and reloaded from the moved bundle |
| [Compatibility matrix](compatibility-matrix.md), both banks | 24 of 24 cases and 4 of 4 encoders passed; one hung evaluation passed on `--resume` |
| Wheel built with `uv build` | Outside the checkout, the installed CLI expanded every training and encoder recipe in `--dry-run` |

`test_the_patch_replaces_cublas_within_tolerance` fails on this GPU before and
after the change, with the same error. CI's CPU runner skips it. `pixi run
test-default` stops after a failing suite, so the other three suites were run
individually.

The learner and interface smokes trained PPO, SuperMimic, SVG and DiffSR on
LAFAN1 with explicit commands and with FSQ and spectral encoders, using the
commands in [Reproduce a training smoke](#reproduce-a-training-smoke) and
[Reproduce a checkpoint reload](#reproduce-a-checkpoint-reload). A latent run
adds `--interface "$INTERFACE" --encoder "$ENCODER"`. Each encoder was
pretrained for two updates with small networks:

```bash
pixi run --locked -e isaaclab imitationlab pretrain --hardware off \
  --interface "$INTERFACE" --data "$BANK" --output-dir "outputs/encoders/$INTERFACE" \
  --num-envs 4 --batch_size 16 --num_updates 2 --eval_batches 1 --seed 42 \
  --wandb_mode disabled --assert-kitless --encoder_hidden_dims 64 64 \
  --diffsr_feature_dim 64 --diffsr_embed_dim 64 --diffsr_f_hidden_dims 64 64 \
  --diffsr_g_hidden_dims 64 64 --diffsr_mu_hidden_dims 64 64
```

Each checkpoint was then exported with `pixi run --locked imitationlab export
--checkpoint "$RUN" --output "$BUNDLE"`, the bundle was moved, and the moved
bundle was evaluated again. Every bundle held exactly the checkpoint's
tensors, and its 20-step evaluation matched the checkpoint's within
run-to-run noise.

On this workstation, about one evaluation in fifteen wrote its JSON and then
did not exit, before and after the change. In every hang inspected, the
evaluator had forked a child that deadlocked before `exec`, and shutdown
waited for that child. When the child was aborted, the evaluator exited with
status 0. Most of the forks came from OmniClient, which relaunched a local
Omniverse Hub install that could not bind its port about eight times a second.
Setting `XDG_DATA_HOME` to an empty directory stops OmniClient from finding the
Hub, and the matrix ran with that setting. One matrix evaluation still hung;
its stuck child had been forked by a Carbonite tasking thread.

### Learning speed

The release was also compared with the commit before the change on
BONES-SEED, with `benchmark run` as described in
[Measure learning speed](training.md#measure-learning-speed). Three cases ran:
explicit SuperMimic, spectral SuperMimic and spectral PPO. Each trained for
30 minutes with seed 42 and was evaluated on the default 256-clip board after
1, 2, 5, 10, 15, 20 and 30 minutes. Both commits used one spectral encoder,
pretrained for 2,000 updates with the earlier commit. Its weights have
SHA-256 `8071d358ee11c74fc37cc76da9061aba65cb93514318614a29f1148778acdc64`.
The release loaded it with `--encoder-seed 42` and that
encoder's `--encoder-source-sha256`. Hardware sizing chose 8,192 environments
for SuperMimic and 4,096 for PPO on both commits. Each case ran twice per
commit. Round 1 ran the earlier commit first in each pair, and round 2 ran the
release first. Another process held about 39 GiB of GPU memory before every
run.

| Case | Round | Commit | Frames (M) | Success (%) | Local error (mm) | Global error (mm) | 50% success (min) | 90% success (min) |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Explicit SuperMimic | 1 | Earlier | 175 | 100.0 | 12.9 | 60.5 | 5 | 5 |
| Explicit SuperMimic | 1 | Release | 181 | 100.0 | 12.7 | 58.6 | 2 | 2 |
| Explicit SuperMimic | 2 | Earlier | 184 | 100.0 | 13.1 | 62.8 | 2 | 2 |
| Explicit SuperMimic | 2 | Release | 186 | 100.0 | 12.7 | 60.0 | 1 | 2 |
| Spectral SuperMimic | 1 | Earlier | 142 | 100.0 | 17.0 | 46.2 | 5 | 5 |
| Spectral SuperMimic | 1 | Release | 145 | 100.0 | 17.3 | 49.5 | 2 | 5 |
| Spectral SuperMimic | 2 | Earlier | 147 | 100.0 | 16.8 | 48.9 | 2 | 2 |
| Spectral SuperMimic | 2 | Release | 148 | 100.0 | 16.6 | 47.4 | 2 | 2 |
| Spectral PPO | 1 | Earlier | 194 | 94.9 | 73.9 | 685.5 | 10 | 15 |
| Spectral PPO | 1 | Release | 194 | 93.8 | 74.1 | 467.3 | 10 | 20 |
| Spectral PPO | 2 | Earlier | 194 | 94.5 | 77.9 | 413.3 | 10 | 20 |
| Spectral PPO | 2 | Release | 193 | 93.8 | 80.6 | 472.0 | 10 | 15 |

Frames, success and pose errors are from the 30-minute checkpoint. A success
time is the first checkpoint that reached the level.

At 30 minutes the two commits track equally well. Success differs by at most
1.1 points. With SuperMimic, local error differs by at most 0.4 mm between the
commits, and one commit moves by up to 0.7 mm between rounds. With spectral
PPO, local error differs by up to 2.7 mm between the commits and by up to
6.5 mm between the rounds of one commit; its global error ranges from 413 to
686 mm over the four runs.

From the 2-minute to the 30-minute checkpoint, the commits collected the same
number of frames in every case and round, within 0.5%. The differences are in
the first two minutes. In round 1 the earlier commit took 49 and 55 seconds to
start SuperMimic, against 20 and 23 seconds for the release, and its first
checkpoint came at 118 and 88 seconds. At 2 minutes it had collected 0.7 and
2.9 million frames, against 6.9 and 6.1 million. Its later success times in
round 1 follow from that delay. In round 2 the earlier commit started in 21 and
23 seconds, so the slow start came from running first, not from the commit.
Both commits then reached 90% success at 2 minutes in both SuperMimic cases.
In round 2 the release still collected more frames in the first minute: 5.5
against 3.5 million with explicit commands, and 4.1 against 2.7 million with
spectral ones, and its explicit run passed 50% success one checkpoint
earlier. These runs do not show whether that first-minute difference comes
from the code or from the compiler cache. The PPO runs reached 90% success at
15 and 20 minutes, in opposite orders in the two rounds.

No run hung, and the exit watchdog that guarded the runs never acted.

## Reproduce a training smoke

Prepare arrays using the [motion data guide](../data/README.md). Set
`BANK` to the packed bank directory. The launcher reads its data identity
from `reference_arrays_manifest.json`.

```bash
POLICY=ppo
TERM=xterm OMNI_KIT_ACCEPT_EULA=YES TORCHDYNAMO_DISABLE=1 \
pixi run --locked -e isaaclab imitationlab train \
  --task imitation-g1-v1 --policy "$POLICY" --data "$BANK" \
  --num-envs 16 --max-iterations 1 --seed 42 --assert-kitless \
  agent.logger.backend= agent.save_interval=1 \
  "agent.logger.log_dir=outputs/release-smoke/$POLICY"
```

For `POLICY=supermimic`, `POLICY=svg`, or `POLICY=diffsr`, append these smaller
model and batch settings to the command.

```text
agent.collector.total_frames=1088 agent.wm.min_replay_steps=64
agent.wm.wm_batch_size=32 agent.wm.imagine_batch=32 agent.wm.horizon=4
agent.wm.wm_hidden=[64,64,64] agent.wm.actor_hidden=[64,64,64]
agent.wm.critic_hidden=[64,64]
```

These world-model runs updated at 1,024 collected transitions. PPO updated at
384. `TORCHDYNAMO_DISABLE=1` avoids the compiler cold start during the smoke.
The launcher still selects the packed training configuration, including the
CUDA graph simulator step.

## Reproduce a checkpoint reload

Set `RUN` to the saved run directory or its `model_step_*.pt` checkpoint.
The [evaluation command](training.md#evaluate-a-checkpoint) recovers the
learner and architecture from the recorded settings.

```bash
pixi run --locked -e isaaclab imitationlab evaluate \
  --checkpoint "$RUN" --data "$BANK" --num-envs 4 --steps 20 --seed 42 \
  --output outputs/evaluation/release-smoke.json --trajectory_ranks 0
```
