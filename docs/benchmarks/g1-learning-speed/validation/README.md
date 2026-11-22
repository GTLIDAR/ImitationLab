# H100 profile follow-up

Ten fresh 600-second runs with normal networks, seed 42 and the original
frozen encoders and boards. PPO compares normalization-only against
normalization plus standard. BONES-SEED latent SVG/DiffSR compares the original
automatic profile against standard. Every run passed the artifact audit.

All six PPO success rates fell, so this PPO profile was rejected. Standard
latent SVG/DiffSR is retained consistently on both datasets; SVG has a
BONES-SEED regression and DiffSR has mixed error tradeoffs. See the
[selected-defaults report](../README.md). These are training-motion
configuration-development measurements, not held-out validation.

Cells show comparator → standard at the final checkpoint. L/G are MPJPE mm.

| Dataset | Command | Learner | Success % | L mm | G mm | Survival s | Train s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| bones_seed | explicit | ppo | 85.5 → 73.8 | 60.9 → 60.7 | 657.2 → 667.1 | 5.6 → 5.2 | 600.13 |
| lafan1 | explicit | ppo | 62.5 → 10.0 | 54.7 → 50.4 | 394.0 → 174.1 | 8.8 → 5.2 | 600.07 |
| bones_seed | fsq | svg | 88.7 → 85.2 | 53.3 → 55.3 | 239.3 → 226.9 | 5.7 → 5.6 | 600.58 |
| bones_seed | spectral | diffsr | 92.6 → 98.0 | 51.4 → 49.0 | 313.3 → 298.1 | 5.8 → 5.9 | 600.26 |
| bones_seed | spectral | svg | 91.8 → 85.2 | 51.1 → 52.8 | 237.6 → 268.9 | 5.7 → 5.6 | 600.64 |
| bones_seed | spectral | ppo | 66.4 → 28.9 | 81.2 → 71.1 | 456.6 → 161.6 | 4.7 → 3.3 | 600.03 |
| bones_seed | fsq | ppo | 77.3 → 27.0 | 83.7 → 69.3 | 667.2 → 193.9 | 5.3 → 3.2 | 601.30 |
| lafan1 | fsq | ppo | 10.0 → 0.0 | 74.3 → 78.4 | 225.7 → 146.0 | 6.1 → 2.1 | 600.60 |
| bones_seed | fsq | diffsr | 94.9 → 97.3 | 51.4 → 52.3 | 283.1 → 339.9 | 5.8 → 5.9 | 600.43 |
| lafan1 | spectral | ppo | 17.5 → 0.0 | 74.1 → 81.8 | 263.1 → 156.0 | 6.0 → 1.9 | 601.42 |

All measurements, hashes and execution sizes: [aggregate records](summary.json).
