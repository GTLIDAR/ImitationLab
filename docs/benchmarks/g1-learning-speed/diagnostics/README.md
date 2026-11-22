# Targeted H100 diagnostics

Ten fresh 600-second runs on the same training-motion boards, seed 42 and
frozen encoders. Source code is unchanged from the baseline; overrides enable
PPO actor/critic input normalization on both banks or select `--hardware standard`
for LAFAN1 latent SVG/DiffSR. The standard profile changes both environments
and learner batches, so this comparison does not isolate either size.

All ten runs passed the same provenance, checkpoint and timing audit. No
result was discarded. These are configuration-development measurements on
training motions; they are not an independent validation set.

Each cell shows **baseline → diagnostic** at the final saved checkpoint.
L/G are local/global MPJPE in mm. Errors use valid pretermination transitions;
a change in survival also changes the error population.

| Dataset | Command | Learner | Success % | L mm | G mm | Survival s | Final train s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bones_seed | spectral | ppo | 62.1 → 66.4 | 81.3 → 81.2 | 391.7 → 456.6 | 4.6 → 4.7 | 600.58 |
| bones_seed | explicit | ppo | 90.2 → 85.5 | 61.2 → 60.9 | 759.6 → 657.2 | 5.7 → 5.6 | 600.38 |
| bones_seed | fsq | ppo | 58.2 → 77.3 | 81.8 → 83.7 | 377.5 → 667.2 | 4.5 → 5.3 | 601.55 |
| lafan1 | spectral | diffsr | 0.0 → 87.5 | 177.7 → 53.2 | 254.7 → 246.2 | 1.0 → 9.7 | 600.57 |
| lafan1 | spectral | ppo | 0.0 → 17.5 | 86.6 → 74.1 | 138.3 → 263.1 | 2.9 → 6.0 | 601.03 |
| lafan1 | spectral | svg | 50.0 → 80.0 | 66.9 → 57.6 | 619.2 → 339.6 | 8.4 → 9.4 | 600.46 |
| lafan1 | explicit | ppo | 0.0 → 62.5 | 64.7 → 54.7 | 164.9 → 394.0 | 4.2 → 8.8 | 600.10 |
| lafan1 | fsq | diffsr | 0.0 → 87.5 | 153.6 → 50.7 | 240.2 → 262.0 | 1.0 → 9.7 | 600.70 |
| lafan1 | fsq | ppo | 0.0 → 10.0 | 96.7 → 74.3 | 162.1 → 225.7 | 2.7 → 6.1 | 600.94 |
| lafan1 | fsq | svg | 50.0 → 72.5 | 67.4 → 59.5 | 511.2 → 419.3 | 8.3 → 9.2 | 600.50 |

Both stalled LAFAN1 DiffSR runs now reach 87.5% success. PPO results improve
on LAFAN1 but remain mixed on BONES-SEED; normalization alone does not make
latent PPO converge in ten minutes. Further profile checks are recorded
separately. Original checkpoints and per-motion evaluations remain outside Git.

See [aggregate records](summary.json) and the [baseline](../baseline/README.md).
