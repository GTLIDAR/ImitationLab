# SVG tuning against the BPTT baseline

The completed experimental search found no SVG tracking improvement that
repeated across both seeds in any of the six dataset/interface cases. BPTT
remains the default. The published [30-minute benchmark](g1-learning-speed-30min/README.md)
was the starting comparison; the [final SVG/BPTT report](svg-tuning/confirmation/README.md)
contains all twelve matched pairs and their learning curves.

**Outperform means better tracking performance.** The primary comparison
is final tracking success and survival together with local and global pose
error. Higher throughput, more collected frames, or an earlier threshold
crossing does not establish a tracking-quality improvement. The agreed
30-minute training budget remains the comparison condition; this study does
not establish asymptotic performance after convergence.

The [first 32 results](svg-tuning/README.md) are complete: 30 runs completed
successfully and two H1 runs failed numerically. The wider critic is the
strongest common SVG refinement, but BPTT still has lower final local/global
errors on both banks. Eight further contact-term comparisons also favor
BPTT. The eight captured-update cases are now complete: the wider-critic
H32 candidate lowers both errors on BONES-SEED, but increases global error
on LAFAN1 despite improving local error. That frozen finalist was then
evaluated across both banks, all interfaces, and two policy seeds. BPTT
had lower local/global errors in all ten fresh matched pairs; the initial
BONES-SEED explicit improvement did not repeat at seed 43. The four BONES-SEED
latent SVG runs also had lower success and survival. No default changed.

The initial screen uses explicit commands on both LAFAN1 and BONES-SEED,
seed 42, one cluster H100 80 GB per job, and 1,800 policy-training seconds.
Compilation after the first collected batch counts against this budget.
Startup, checkpoint evaluation, and any encoder pretraining are recorded
separately. Each job starts with its own compiler cache.

| Candidate | Changes from default SVG |
| --- | --- |
| H16 | `agent.wm.horizon=16` |
| H8 | `agent.wm.horizon=8` |
| H16 recent | H16, `agent.wm.real_window_recent_steps=8` |
| H16 real critic | H16, `agent.wm.critic_source=real` |
| H16 structured | H16, `agent.wm.wm_structured=true` |
| H16 score correction | H16, `agent.wm.actor_estimator=rebar` |

Fresh default SVG and default SuperMimic controls run on each bank from the
same source snapshot. This is 16 runs, or eight policy-training GPU-hours;
startup and evaluation add to that total. All explicit runs pin the
published xlarge geometry: 16,384 environments, world-model and actor
batches of 8,192, and a GPU-resident compact reference library. Model sizes,
optimizers, task rewards, resets, and evaluation conditions otherwise retain
the published defaults. H changes both actor and critic rollout horizons.

A subsequent horizon extension adds H1 and H4 on each bank: four further
30-minute H100 runs, or two more policy-training GPU-hours. It uses the same
source snapshot and BPTT controls. These runs test whether the speed of
shorter rollouts compensates for increased dependence on the critic's value
gradient. The extension was chosen after observing iteration throughput,
before tracking evaluations were available; it is not a selected recipe.

The first 16 runs completed with BPTT ahead on both pose-error metrics on
both banks. Default H32 had the lowest errors among the tested SVG recipes.
The next round therefore retains H32 while independently reducing action
noise to 0.05, reducing the actor learning rate to 0.0002, reducing world-model
updates to two per iteration, widening the critic to `[512,256]`, or removing
the actor's value bootstrap (the critic is still fit). A sixth candidate
extends the horizon to 64 and collects 128 replay steps before updating.
Each runs on both banks for another six policy-training GPU-hours. These
are hypotheses under evaluation, not promoted defaults.

A contact-term follow-up tests H32 and H64 with
`agent.wm.contact_reward=0.25 agent.wm.ref_contact_fix=true`. SVG now supports
this optional term on recorded windows as well as imagined critic targets.
The term already belongs to the BPTT recipe; it does not change the simulator
reward, tracking metrics, or recorded episode boundaries. Terminal transitions
keep their measured environment reward. Both candidates run on both banks,
alongside fresh BPTT and default-SVG controls from the same updated source:
eight runs and four policy-training GPU-hours. Compare those candidates to
their new controls, keeping source revisions explicit in the report.

The contact comparison found no tracking improvement over BPTT. The final
screen therefore leaves contact shaping off and tests captured SVG updates:
H32, H32 with the wider `[512,256]` critic, and H64 with 128 replay steps
before updating. Each enables `agent.wm.no_sync=true`,
`agent.wm.fused_adam=true`, `agent.wm.update_graphs=true`, and
`agent.wm.graph_early=true`. Fresh BPTT controls from the same updated source
bring this round to eight runs, four policy-training GPU-hours, on the same
two explicit-command cases. Capture is a way to perform more learning within
the fixed budget; only final tracking quality can justify selecting it.
This is the final configuration screen before choosing one common SVG
candidate for confirmation. If no candidate wins, report that result.

The captured path passed 45 CUDA graph tests, including graph/eager update
parity and sampling while replay fills and wraps; the CPU suites passed
1,026 tests with two skips. Local public training and evaluation reload
checks passed for explicit, FSQ, and spectral commands. Those small integration
checks used 16 environments and 1,536 frames, with small smoke encoders for
the latent interfaces; they do not establish tracking quality. The H100
comparison uses the normal networks and frozen benchmark encoders.

The structured variant infers replay residuals in semi-implicit output
coordinates and uses the same integrator as model fitting and imagination.
At the collecting policy, its forward pass reconstructs the recorded
in-episode transition; gradients use the model Jacobian. This does not make
the learned Jacobian exact. The score-correction variant retains the existing
clipping, averaging, and norm cap; it is not claimed to be an unbiased
estimator under these settings.

SuperMimic retains its existing structured model, reference-window cache,
contact reward inside the model, and captured learner updates. SVG retains
its real-window update path, with capture only in the final screen. This
therefore compares complete training
recipes; it does not isolate the effect of the actor-gradient estimator.
The simulator task and evaluation metric definitions stay fixed.

Evaluation uses the same fixed seed-42 boards as the published study: all
40 LAFAN1 clips and 256 BONES-SEED clips, starting at frame zero, for up to
500 control steps. Checkpoints are requested at 1, 2, 5, 10, 15, 20, and
30 minutes; reports use their actual training clocks. These are acquisition
measurements on training motions, not held-out generalization.

Selection prioritizes final tracking quality. For a case to support a clear
improvement, the candidate must preserve tracking success and survival and
reduce both local and global pose error. Otherwise report the tradeoff.
The full learning curves and throughput are diagnostics, not substitutes for
tracking quality. First shortlist variants on the two explicit cases,
then test a single candidate across both banks and all three interfaces,
reusing the four frozen benchmark encoders with their recorded hashes.
Repeat the finalist and BPTT with another training seed while keeping the
board seed fixed. Report tradeoffs per case; claim an improvement only for
the metrics and cases supported by those repeats. Do not pick a different
winner for each cell or replace final checkpoints with best checkpoints.

The confirmation controls started independently of finalist selection: BPTT
on both banks and all three interfaces with policy seed 43, plus both latent
interfaces with policy seed 42. The explicit seed-42 controls are reused
from the contact comparison. These ten additional runs share the contact
source snapshot, fixed seed-42 boards, and the four frozen benchmark encoders;
their weight hashes were checked again before submission. All ten new runs
completed cleanly; the [12 reference cases](svg-tuning/README.md#bptt-reference-repeats)
include 73 evaluated checkpoints. These controls belong to the contact source
group. The selected finalist uses the newer captured-update source, so its
final comparison used matching controls from that source; the earlier results
remain reported separately. Finalist selection used success and survival as
gates, then the geometric mean of local/global error ratios across both banks
to choose a single common candidate. The chosen configuration was run unchanged
across all interfaces and both policy seeds, retaining the fixed evaluation seed.

The final screen selected captured H32 with `agent.wm.critic_hidden=[512,256]`:
its geometric error ratio is 0.9792 with no success or survival regression.
That aggregate improvement does not remove the LAFAN1 global-error tradeoff.
Confirmation kept all four capture flags and the wider critic unchanged.
It reused the four explicit seed-42 cases from the captured screen and added
20 matched SVG/BPTT runs: both latent interfaces at seed 42 and all three
interfaces at seed 43, on both banks. This is another ten policy-training
GPU-hours, with the same frozen encoders, boards, and source archive. All twenty
new jobs completed with Slurm exits of 0:0, finite final checkpoints, and
positive model/actor/critic optimizer counters. The complete comparison has
24 runs and 132 evaluated checkpoints; actual final training clocks are
1,800.27–1,800.77 seconds. The four reused explicit seed-42 runs are selection
data, not independent confirmation samples. No case met the tracking-win
criterion in both seeds.

The BPTT source-compatibility check compares the pre-extension and current
samplers in compiled H32 Muon/EMA updates with the structured model and
contact term. Initialization, model/optimizer state, replay tensors, and
metrics are bitwise equal through early capture, buffer filling, and wraparound
in the synthetic harness. This does not establish deterministic scores in
the real simulator. The captured-screen BPTT control was weaker on LAFAN1
than earlier reference runs. The modest single-seed gains did not survive
the repeated-seed comparison and do not establish a reliable tracking advantage.

Raw BONES-SEED records, motion identifiers, checkpoints, and private cluster
paths stay outside Git. Publish aggregate results and provenance hashes.
