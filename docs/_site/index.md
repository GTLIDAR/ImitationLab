# ImitationLab

**Train whole-body motion trackers for the Unitree G1 in minutes on one GPU.**

ImitationLab is a research codebase for humanoid motion tracking in
[Isaac Lab](https://github.com/isaac-sim/IsaacLab) with the Newton/MJWarp
physics backend. One task, four policy learners, and three ways to feed the
reference motion to the policy share a single command-line tool, so learners
and interfaces can be compared on equal terms.

![Tracking success and local pose error versus policy training time for SuperMimic, SVG, DiffSR and PPO: median and range over six G1 runs](benchmarks/g1-release-check/overview-light.svg){.only-light}
![Tracking success and local pose error versus policy training time for SuperMimic, SVG, DiffSR and PPO: median and range over six G1 runs](benchmarks/g1-release-check/overview-dark.svg){.only-dark}

The figure shows 30-minute runs on one NVIDIA H100; the
[four-tier release check](benchmarks/g1-release-check.md) has the numbers and
their limits.

## Where to start

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Set up
:link: setup
:link-type: doc

Install the Pixi environments and Isaac Lab, then run the smoke test.
:::

:::{grid-item-card} Prepare motion data
:link: motion-data
:link-type: doc

Convert LAFAN1 or BONES-SEED and pack a reference bank.
:::

:::{grid-item-card} Train and evaluate
:link: training
:link-type: doc

Commands, policy learners, command interfaces, checkpoints and export.
:::

:::{grid-item-card} Choose hardware
:link: hardware
:link-type: doc

How execution sizes follow the GPU, and how to tune throughput locally.
:::

:::{grid-item-card} Read the results
:link: benchmarks/g1-release-check
:link-type: doc

All 24 learner, interface and dataset cases on four GPU tiers.
:::

:::{grid-item-card} Understand the code
:link: architecture
:link-type: doc

Package boundaries and where each kind of code belongs.
:::

::::

## Install

ImitationLab needs Linux x86-64, an NVIDIA GPU, and
[Pixi](https://pixi.sh) 0.83 or newer.

```bash
git clone https://github.com/GTLIDAR/ImitationLab.git
cd ImitationLab
pixi install --locked -e isaaclab
```

No motion data is distributed with the repository. [Setup](setup.md) covers
the environments, and [motion data](motion-data.md) covers preparing a
reference bank to train on.

## Citation

If you use ImitationLab or its spectral skill encoders in your research,
please cite:

```bibtex
@article{wu2026spectral,
  title   = {Learning Expressive and Compositional Motion Representation via Spectral Skills},
  author  = {Wu, Feiyang and Gao, Chenxiao and Yang, Chen and Zhao, Ye and Dai, Bo and Wu, Anqi},
  journal = {arXiv preprint arXiv:2609.37677},
  year    = {2026},
  url     = {https://arxiv.org/abs/2609.37677}
}
```

```{toctree}
:caption: Using ImitationLab
:hidden:

setup
motion-data
training
hardware
```

```{toctree}
:caption: Design and development
:hidden:

architecture
policy-learners
motion-data-tools
Upstream revisions <provenance>
```

```{toctree}
:caption: Results
:hidden:

Release check on four GPU tiers <benchmarks/g1-release-check>
Learning speed, 30-minute runs <benchmarks/g1-learning-speed-30min>
Learning speed, 10-minute runs <benchmarks/g1-learning-speed>
PPO tuning <benchmarks/ppo-tuning>
SVG tuning <benchmarks/svg-tuning>
Hardware selection check <benchmarks/hardware-selection-verification>
Compatibility matrix <compatibility-matrix>
Release verification <release-verification>
```

```{toctree}
:caption: Project
:hidden:

contributing
changelog
security
third-party-notices
```
