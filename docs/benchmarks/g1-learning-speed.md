# G1 learning speed

The [24-case H100 report](g1-learning-speed/README.md) is complete, with the
[original baseline](g1-learning-speed/baseline/README.md) and all 20 follow-up
runs retained separately. All 44 policy runs and four encoder jobs finished
successfully and passed the artifact/provenance audit. Process completion is
not a tracking-quality pass: the report includes weak and regressing cases.

New PPO runs now normalize both networks' inputs. Automatic H100 sizing caps
latent SVG/DiffSR at standard, consistently across both banks. Smaller PPO
batches worsened all six follow-up cases and were rejected. These defaults
were chosen using these training-motion measurements, so the final table is
not an independent test of configuration selection. The changes preserve
networks, encoders, seed, budget, task contract and checkpoint resume settings.

The earlier L40S sweep completed three policy cases and all four encoders.
Its remaining jobs were cancelled when switching hardware; those measurements
remain separate from the H100 comparison.

In the baseline sweep, four encoder jobs and eight explicit-policy jobs ran
independently. Each latent-policy job waited for its matching encoder. The
follow-ups reused those frozen encoders and used explicit overrides for the
settings now selected by default. All runs share the same training source
fingerprint, normal networks and 600-second budget. Encoders received 2,000
updates on H100s.

The shared learner defaults include the corrected zero model-only warmup
for latent commands. The benchmark checks actual policy optimizer updates
as well as finite checkpoint tensors. The separate 24/24 result in the main
README remains an integration check.

| Setting | Value |
| --- | --- |
| Task | `imitation-g1-v1` |
| Learners | PPO, SuperMimic (BPTT), SVG, DiffSR |
| Command interfaces | Explicit, FSQ, DiffSR |
| Banks | LAFAN1 (40 clips), BONES-SEED (129,785 clips) |
| Policy training | 600 cumulative training seconds per combination |
| Snapshots | First completed iteration after 60, 120, 300, 600 seconds |
| Encoders | One per bank/interface; 2,000 updates, frozen for policy learning |
| Settings | Normal networks; shared fast defaults; PPO input normalization; H100 latent SVG/DiffSR capped at standard |
| Hardware | One NVIDIA H100 per job; 8 CPU cores; 96 GiB host-memory request; `gpu` / `gpu` |
| Seed | 42 |
| Evaluation | All 40 LAFAN1 clips; deterministic seed-42 sample of 256 BONES-SEED clips |
| Horizon | First 500 control steps (10 seconds), or natural clip completion if shorter |
| Actions and resets | Deterministic actions; frame-zero starts; no randomization |

Success means no task tracking failure through the evaluated prefix. The
failure rule is absolute root-height error greater than 0.25 m or absolute
root R[2,2] error greater than 0.8. Report MPJPE-L, MPJPE-G, and survival time
alongside success. Pose errors aggregate valid pretermination transitions
across all evaluated clips; failed runs contribute shorter trajectories.
These are training motions, so the benchmark measures acquisition rather
than held-out generalization or full-bank coverage.

Use actual saved checkpoint times. A long iteration can cross several
requested times, producing one measurement for those columns. Thresholds
(50%, 90%, and 95% success, plus 90% with MPJPE-L at most 40 mm) are first
observed crossings, without interpolation. Targets not observed by the last
checkpoint are reported as not reached. One seed cannot establish a reliable
learner ranking.

A completed benchmark must have positive optimizer step counters for the
policy, and for the world model and critic where applicable. Finite weights
or a zero process exit code alone do not establish that policy learning ran.

The training clock starts after the first collected batch. Learner compilation
and checkpoint overhead after that point count toward the budget. Record
startup and encoder process time separately; add them to training time for
the end-to-end figure. Queue and evaluation time are excluded. Each job has
its own cold Torch compiler cache. Checkpoints are evaluated after training
in separate processes.

Both bank manifests match those used in the local compatibility checks:

- LAFAN1: `lafan1@0ff4dce7ae91`, manifest SHA-256
  `ae7b212a370e98823592ec0532adb427c948197f5406937f3e79af8d7568167f`.
- BONES-SEED: `bones_seed_sonic_full_129785@e714bbff`, manifest SHA-256
  `936799c1fd7621838b8ddbfd6d1230c0d1b4fbb142e3cd61b20843cc197610c5`.

The original H100 sweep uses source commit `64d70f422515bbd3e437790a44918c528ef20dc0`
and workspace archive SHA-256
`fa5765703479fe2d4993f87a04002b2261b3e0718563bec1a651c7ae4fb06663`.
The two follow-up snapshots are `e1923ff` and `1ca5909`, whose intervening
changes are documentation only. Every measured run records training source
fingerprint `eea53176a120de7b1595e6401ab54db6dae02c53492ff981b63251505e0c6b1f`.
The final launcher makes the tested overrides default; the selected results
were not rerun after that equivalent default change. CPU checks verify
selection and resume behavior, and the report audit checks that the current
automatic profile resolves to the measured settings for every row.
Follow-up archive hashes and per-row measurement phases are in the aggregate
JSON records. [Audit counts](g1-learning-speed/audit.json) cover all three phases.
The runtime pins Isaac Lab
commit `06128939c6f3264594483a73e4344e4fefcea6d3`, PyTorch 2.12.0+cu130,
TorchRL 0.13.3, TensorDict 0.13.0, Newton 1.6.1, and MuJoCo/MJWarp 3.12.0.
The container SHA-256 is
`3f586458dcf145709edce8c4a83be52fef3e3ee08e79c85346b551e2de68e3c1`.
Resolved sizes, source and weight hashes, GPU model, and measured times are
recorded with each result.

See [training](../training.md#measure-learning-speed) to reproduce and render
the report with `imitationlab benchmark`. BONES-SEED remains proprietary:
raw motions, per-motion output, encoder weights, and policy checkpoints stay
outside Git. Only aggregate results and figures belong in this report.
