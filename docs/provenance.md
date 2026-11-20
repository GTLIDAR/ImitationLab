# Source provenance

This page records the source revisions ImitationLab was built from and the
upstream components it vendors or depends on. It is for readers who need to
trace a component to its origin or reproduce an earlier design. License terms
are in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## Starting point

ImitationLab started from the curated public export of IsaacLab-Imitation's
`release/public` at `cd03333716e3ed3543e506d9b26f7a7a6358694e`. The
release's original RLOpt and ImitationLearningTools snapshots were
`bc44635c4b2dee8e1b2d7cac2aaf37ca92e5b6a0` and
`ec96905b0488adb8c90bf0d0ad0e51605ab06bbd`.

**Note.** IsaacLab-Imitation, ImitationLearningTools, and the pre-release
ImitationLab repository are predecessors that are not public. Their revisions
are listed on this page for the record and are not linked.

## Component revisions

| Component | Source revision |
| --- | --- |
| G1 tracking environment, tracker entry points and shared task model | IsaacLab-Imitation `b451fd8` |
| RLOpt PPO, encoder and world-model implementation | [RLOpt `533764c5`](https://github.com/fei-yang-wu/RLOpt/commit/533764c5fc79f9b4960efc7b95767f9088a8be67) |
| Motion loading and data tools | ImitationLearningTools `94af348` |
| Official SuperMimic | [HybridRobotics/supermimic `abc124c`](https://github.com/HybridRobotics/supermimic/commit/abc124c1bd5976341034668af3d12c087f2f1f55) |
| Official SuperMimic simulator dependency | [HybridRobotics/mjlab_research `dd51a9f`](https://github.com/HybridRobotics/mjlab_research/commit/dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd) |
| Isaac Lab simulator dependency | [Isaac Lab `0612893`](https://github.com/isaac-sim/IsaacLab/commit/06128939c6f3264594483a73e4344e4fefcea6d3) |

## Task and earlier designs

`imitation-g1-v1` uses the environment classes of the upstream G1 tracking
task at that revision; it is the only registered task. The FSQ recipe uses the
normalized FSQ (finite scalar quantization) lattice and the same heading-frame
observation path as the continuous command.

Earlier work was last present at these revisions of the pre-release
ImitationLab repository:

| Earlier work | Revision |
| --- | --- |
| The earlier task designs | Pre-release ImitationLab `fbe5b63`; see [migration](migration.md) |
| Legacy IPMD, IRL, and off-policy learners | Pre-release ImitationLab `0041b69` |

## Official SuperMimic

The vendored SuperMimic learner source is unchanged. Its dependency lock was
resolved against the pinned mjlab fork and pins mjviser to 0.0.16 for MuJoCo 3.
See [setup](setup.md#official-supermimic) for how to run it.

## Training defaults

The defaults for explicit commands on packed motion data come from the
BONES-SEED hardware-tier training settings of IsaacLab-Imitation `b451fd8`
(`configs/wmbptt_g1_bones`). The three learners present when these defaults
were adopted (PPO, SuperMimic, and SVG) share the simulator settings. The
world-model learners also share the compatible optimizer and batch settings
from that configuration.

## Data tools

BONES-SEED selection, CSV normalization, and batch planning were adapted from
the IsaacLab-Imitation data tools at `b451fd8` (`scripts/data`).
