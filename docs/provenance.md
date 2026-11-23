# Upstream revisions

This page records the revisions of the upstream components ImitationLab
vendors or depends on. It is for readers who need to trace a component to its
source or pin the same versions. License terms are in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

| Component | Source revision |
| --- | --- |
| Official SuperMimic | [HybridRobotics/supermimic `abc124c`](https://github.com/HybridRobotics/supermimic/commit/abc124c1bd5976341034668af3d12c087f2f1f55) |
| Official SuperMimic simulator dependency | [HybridRobotics/mjlab_research `dd51a9f`](https://github.com/HybridRobotics/mjlab_research/commit/dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd) |
| Isaac Lab simulator dependency | [Isaac Lab `f9af6e3`](https://github.com/isaac-sim/IsaacLab/commit/f9af6e3c52ccb43533996713985808fed2cd46db) |

## Official SuperMimic

The vendored SuperMimic learner source is unchanged. Its dependency lock was
resolved against the pinned mjlab fork and pins mjviser to 0.0.16 for MuJoCo 3.
See [setup](setup.md#official-supermimic) for how to run it.
