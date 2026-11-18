# Setup

Run the commands from the repository root. Pixi installs ImitationLab in
editable mode and installs Isaac Lab from a pinned Git revision. The
`imitationlab` command provides the data, pretrain, train, evaluate, and
export workflows.

## Isaac Lab

Install [Pixi](https://pixi.sh/latest/installation/) 0.83 or newer. The workspace targets
Linux x86-64. Training requires an NVIDIA GPU and a driver compatible with
the CUDA 13.0 PyTorch build. The Isaac Lab environment targets glibc 2.35 or
newer. Allow several tens of gigabytes for the simulation stack and package
cache.

```bash
pixi install --locked -e isaaclab
pixi run -e isaaclab smoke-g1
```

Isaac Sim requires accepting the [NVIDIA Omniverse License Agreement](https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html).
The `smoke-g1`, `test-isaaclab`, `benchmark` and `tune` commands set
`OMNI_KIT_ACCEPT_EULA=YES` for their simulator processes, so running them
accepts it. Other commands that start Isaac Sim, such as motion conversion,
ask in a terminal; set `OMNI_KIT_ACCEPT_EULA=YES` to run them unattended.

The one-iteration smoke uses the bundled Dance102 motion and writes a
checkpoint under `outputs/smoke-g1`. The first simulation launch may fetch
Isaac Sim ground-plane assets and compile Warp kernels. See [training and
evaluation](training.md) for full runs. For CPU tests and data-bank validation,
install the default environment with `pixi install --locked`.

Inspect the local machine before training with
`pixi run -e isaaclab imitationlab hardware --data "$BANK"`.
Fresh runs use automatic hardware sizing by default; see
[hardware selection](hardware.md) for manual settings and optional calibration.

## Official SuperMimic

The official runner uses Python 3.13 and HybridRobotics' mjlab fork. Install
[uv](https://docs.astral.sh/uv/getting-started/installation/), then prepare the
external simulator beside the vendored SuperMimic package.

```bash
git clone https://github.com/HybridRobotics/mjlab_research.git third_party/mjlab_research
git -C third_party/mjlab_research checkout --detach dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd
(cd third_party/supermimic && uv sync --locked)
```

The sibling checkout is ignored by Git. Use a local motion file in mjlab's
tracking format. To convert the bundled Dance102 clip, run from the repository
root after installing the default Pixi environment.

```bash
pixi run python scripts/data/convert_mjlab_motion.py \
  data/unitree/npz/g1/G1_Take_102.bvh_60hz.npz outputs/dance102-mjlab.npz \
  --start 1 --stop 1457
```

Run the official generalized-coordinate task from an output directory so the
runner's relative `logs/` path stays under `outputs/`.

```bash
mkdir -p outputs/official-supermimic
(
  cd outputs/official-supermimic
  runner=../../third_party/supermimic
  "$runner/.venv/bin/python" "$runner/src/supermimic/scripts/train.py" \
    SuperMimic-GenCoord-G1 --motion-file ../dance102-mjlab.npz \
    --runner.run-name g1-baseline
)
```

The official runner also exposes `SuperMimic-MaxCoord-G1` and accepts a W&B
motion artifact. Its task IDs, data format, and checkpoints belong to the
mjlab baseline. See the [upstream README](../third_party/supermimic/README.md).

For first-release dataset setup, see [motion data](../data/README.md). Run
`pixi run --locked test-default` for the learner, data-tool, and script suites,
`pixi run --locked -e isaaclab test-isaaclab` for the G1 environment suite, and
`pixi run --locked -e static ci-static` for lint and compile checks.
