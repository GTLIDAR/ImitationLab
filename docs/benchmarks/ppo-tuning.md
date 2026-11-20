# PPO tuning

The published 30-minute benchmark used normalized vanilla PPO, so it did not
show what a tuned PPO recipe can reach. This study screens three stronger
recipes and confirms the selected one against a fresh default run.

**Better means better tracking.** The comparison uses the final checkpoint
after 1,800 policy-training seconds: tracking success, survival, local pose
error (MPJPE-L), and global pose error (MPJPE-G). More frames or earlier
threshold crossings do not count as an improvement.

## Protocol

All runs use `imitation-g1-v1`, one cluster NVIDIA H100 80 GB per job, automatic
hardware sizing (the `large` profile, 8,192 environments), and 1,800
policy-training seconds. Startup, encoder pretraining, and checkpoint
evaluation are recorded separately. Every run evaluates the same fixed seed-42
boards as the [30-minute benchmark](g1-learning-speed-30min/README.md): all 40
LAFAN1 clips and a fixed 256-clip BONES-SEED board (bank
`bones_seed_sonic_full_129785@e714bbff`), each from frame zero for up to 10
seconds. These are training motions, so the study measures acquisition, not
held-out generalization.

The FSQ and spectral runs reuse the four frozen encoders of the 30-minute
benchmark (seed 42, 2,000 updates, encoder source
`9756613231dd…`). The candidates run from source commit `f38440b` (source
fingerprint `df851c8e…`), the same source as the final SVG/BPTT confirmation.

The selection rule was recorded before any result arrived:

1. A candidate must complete all six dataset/interface cases.
2. In every case, its final success must be at least the default's minus one
   percentage point.
3. Rank the eligible recipes and the default by final MPJPE-L in each case,
   and order them by mean rank. Break ties by mean MPJPE-G rank, then mean
   success. Select one generic recipe; no per-case selection.
4. Confirm the selected recipe against a fresh default run at policy seed 43.
   Adopt it as the PPO default only if it lowers MPJPE-L in at least five of
   the six seed-43 pairs without violating the success guard.

## Candidates

All three use SiLU networks, normalized inputs, γ = 0.99, GAE λ = 0.95,
clip 0.2, an adaptive KL learning-rate schedule stepped once per iteration,
and captured (CUDA-graph) PPO updates. The local wiring check confirmed that
the captured update ran for each recipe instead of falling back to eager
updates.

| Recipe | Networks (actor and critic) | Rollout | Epochs × minibatch | Learning rate | Other |
| --- | --- | --- | --- | --- | --- |
| Default | [512, 256, 128] ELU | 24 steps | 5 × 24,576 | 1e-3, KL 0.01 per minibatch | entropy 0.005 |
| Adaptive minibatch | [1024, 1024, 512] | 24 steps | 3 × 49,152 | 1e-3 (1e-5 to 1e-3), KL 0.02 | entropy 0 |
| Full batch | [2048, 2048, 1024, 1024, 512, 512] | 24 steps | 2 × full rollout | 1e-3 (1e-5 to 1e-3), KL 0.02 | entropy 0, weight decay 0.01 |
| Low noise | [2048, 2048, 1024, 1024, 512, 512] | 32 steps | 5 × 65,536 | actor 2e-5, critic 1e-3, KL 0.01 | entropy 0.01, initial log σ = −3, global advantage normalization |

Full batch adapts an earlier internal full-batch recipe to pure PPO. Low
noise adapts portable PPO settings from NVIDIA's
[SONIC configuration](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/gear_sonic/config/algo/ppo_im_phc.yaml);
it does not reproduce SONIC's architecture, auxiliary losses, or environment.

## Screen (policy seed 42)

The default row is the published 30-minute run (source `9756613231dd…`); the
seed-43 confirmation below repeats the default from the candidates' source.

| Case | Recipe | Success | MPJPE-L (mm) | MPJPE-G (mm) | Survival (s) | Frames |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LAFAN1 explicit | Default | 92.5% | 43.8 | 295 | 9.88 | 284M |
| | Adaptive minibatch | 92.5% | **43.2** | 230 | 9.87 | 310M |
| | Low noise | 32.5% | 123.5 | 384 | 5.81 | 193M |
| | Full batch | 0.0% | 57.2 | 62 | 0.17 | 272M |
| LAFAN1 FSQ | Default | 85.0% | **59.7** | 464 | 9.52 | 193M |
| | Adaptive minibatch | 87.5% | 70.1 | 379 | 9.72 | 211M |
| | Low noise | 0.0% | 146.4 | 232 | 0.94 | 76M |
| | Full batch | Diverged (non-finite at 70.4M frames) | | | | |
| LAFAN1 spectral | Default | 87.5% | 69.6 | 445 | 9.80 | 195M |
| | Adaptive minibatch | 90.0% | **57.3** | 283 | 9.74 | 212M |
| | Low noise | 20.0% | 113.5 | 367 | 4.43 | 154M |
| | Full batch | 0.0% | 56.8 | 61 | 0.20 | 190M |
| BONES-SEED explicit | Default | 96.1% | **51.6** | 674 | 5.88 | 278M |
| | Adaptive minibatch | 96.9% | 59.1 | 610 | 5.88 | 305M |
| | Low noise | 49.6% | 92.5 | 459 | 4.05 | 186M |
| | Full batch | Out of GPU memory | | | | |
| BONES-SEED FSQ | Default | 96.1% | **65.5** | 755 | 5.86 | 193M |
| | Adaptive minibatch | 96.1% | 74.4 | 764 | 5.86 | 166M |
| | Low noise | 60.5% | 96.0 | 564 | 4.38 | 151M |
| | Full batch | Out of GPU memory | | | | |
| BONES-SEED spectral | Default | 94.9% | 68.6 | 730 | 5.80 | 192M |
| | Adaptive minibatch | 96.1% | **66.8** | 665 | 5.84 | 202M |
| | Low noise | 72.7% | 96.2 | 689 | 4.76 | 154M |
| | Full batch | Out of GPU memory | | | | |

Read pose error together with success and survival. Full-batch runs that
completed fell within about 0.2 seconds, so their low pose errors cover only
a few frames each.

**Full batch is not feasible at this size.** Its captured update needs about
25 GB of graph memory for the 196,608-row rollout. All three BONES-SEED runs
exhausted the 80 GB H100 within four minutes, and LAFAN1 FSQ became
non-finite. It fails the eligibility rule.

**Low noise does not learn within the budget.** With an initial action
standard deviation of 0.05 and an actor learning rate of 2e-5, success stays
between 0% and 73%. It fails the success guard in all six cases.

**Adaptive minibatch is selected, by a narrow margin.** It lowers MPJPE-L in
three cases and raises it in three, so its mean MPJPE-L rank ties the
default's (1.50 each). It wins the tie-break with lower MPJPE-G in five of
six cases (mean rank 1.67 versus 2.67). Its success is equal or higher in
every case (mean 93.2% versus 92.0%).

## Confirmation (policy seed 43)

The selected recipe and the current default ran fresh at policy seed 43 from
the same source (`f38440b`), with the same frozen encoders and seed-42
boards. All 12 runs completed.

| Case | Default success | Default L / G (mm) | Adaptive success | Adaptive L / G (mm) |
| --- | ---: | ---: | ---: | ---: |
| LAFAN1 explicit | 92.5% | **47.2** / 337 | 92.5% | 49.1 / 274 |
| LAFAN1 FSQ | 85.0% | 73.2 / 489 | 90.0% | **58.2** / 364 |
| LAFAN1 spectral | 87.5% | **61.2** / 384 | 85.0% | 75.6 / 396 |
| BONES-SEED explicit | 96.5% | **49.2** / 629 | 96.9% | 57.2 / 566 |
| BONES-SEED FSQ | 95.3% | **68.3** / 757 | 96.5% | 70.3 / 750 |
| BONES-SEED spectral | 93.8% | 71.0 / 659 | 96.1% | **70.3** / 737 |

**The selected recipe is not adopted; the PPO default is unchanged.**
Adaptive minibatch lowers MPJPE-L in two of six pairs, short of the five the
rule requires, and LAFAN1 spectral loses 2.5 percentage points of success,
which violates the guard. It lowers MPJPE-G in four of six pairs.

The two seeds disagree case by case. LAFAN1 spectral favored adaptive
minibatch at seed 42 (57.3 versus 69.6 mm) and the default at seed 43 (75.6
versus 61.2 mm); LAFAN1 FSQ did the reverse. The default alone moves by up to
13.5 mm in MPJPE-L between seeds (its seed-42 run is the published one from
the earlier source). At this budget, seed variation is as large
as the recipe effect, so the screen's narrow selection did not repeat.

## Conclusion

Within a 30-minute H100 budget, none of the three stronger recipes improved
PPO tracking reliably. Larger networks with an adaptive learning rate change
global error more than local error; full-batch updates do not fit at 8,192
environments; a SONIC-style low-noise schedule learns too slowly. PPO stays
at 43–76 mm MPJPE-L across both seeds, against 13–20 mm for SuperMimic (BPTT)
in the [30-minute benchmark](g1-learning-speed-30min/README.md). Closing that
gap would need changes outside this study, such as rewards, observations, or
much longer training.

## Limits

This is one screen seed plus one confirmation seed. The comparison is
complete recipes under a fixed wall-clock budget, so faster-per-frame
recipes also collect more frames. All recipes keep the environment rewards,
terminations, observations, and γ fixed; reward or observation changes are
outside this study.
