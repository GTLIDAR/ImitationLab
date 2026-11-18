# Source provenance

ImitationLab started from the curated public export of IsaacLab-Imitation's
`release/public` at `cd03333716e3ed3543e506d9b26f7a7a6358694e`. The
release's original RLOpt and ImitationLearningTools snapshots were
`bc44635c4b2dee8e1b2d7cac2aaf37ca92e5b6a0` and
`ec96905b0488adb8c90bf0d0ad0e51605ab06bbd`.

| Component | Source revision |
| --- | --- |
| G1 tracking environment, tracker entry points and shared task model | [IsaacLab-Imitation `b451fd8`](https://github.com/fei-yang-wu/IsaacLab-Imitation/commit/b451fd8ae89171f0f71b2122e45585e62ceda9ac) |
| RLOpt PPO, encoder and world-model implementation | [RLOpt `533764c5`](https://github.com/fei-yang-wu/RLOpt/commit/533764c5fc79f9b4960efc7b95767f9088a8be67) |
| Motion loading and data tools | [ImitationLearningTools `94af348`](https://github.com/GTLIDAR/ImitationLearningTools/commit/94af348e6a97a5e7381373690d11ce8e73df2244) |
| Official SuperMimic | [HybridRobotics/supermimic `abc124c`](https://github.com/HybridRobotics/supermimic/commit/abc124c1bd5976341034668af3d12c087f2f1f55) |
| Official SuperMimic simulator dependency | [HybridRobotics/mjlab_research `dd51a9f`](https://github.com/HybridRobotics/mjlab_research/commit/dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd) |
| Isaac Lab simulator dependency | [Isaac Lab `0612893`](https://github.com/isaac-sim/IsaacLab/commit/06128939c6f3264594483a73e4344e4fefcea6d3) |

`imitation-g1-v1` uses the environment classes of the upstream G1 tracking
task at that revision; it is the only registered task. The earlier task designs
remain in the history at [`fbe5b63`](https://github.com/fei-yang-wu/ImitationLab/commit/fbe5b63)
(see [migration](migration.md)). Reproduce legacy IPMD, IRL,
and off-policy learners with the [pre-cleanup revision](https://github.com/fei-yang-wu/ImitationLab/commit/0041b69b98e127f2e2cb0b556e53891b122e6461). The FSQ recipe uses the normalized FSQ lattice
and the same heading-frame observation path as the continuous command.

The vendored SuperMimic learner source is unchanged. Its dependency lock was
resolved against the pinned mjlab fork and pins mjviser to 0.0.16 for MuJoCo 3.

The defaults for explicit commands on packed motion data come from the
[BONES-SEED hardware-tier training settings](https://github.com/fei-yang-wu/IsaacLab-Imitation/tree/b451fd8ae89171f0f71b2122e45585e62ceda9ac/configs/wmbptt_g1_bones).
All three learners share the simulator settings. The world-model learners also
share the compatible optimizer and batch settings from that configuration.

BONES-SEED selection, CSV normalization, and batch planning were adapted from
the [IsaacLab-Imitation data tools at `b451fd8`](https://github.com/fei-yang-wu/IsaacLab-Imitation/tree/b451fd8ae89171f0f71b2122e45585e62ceda9ac/scripts/data).
