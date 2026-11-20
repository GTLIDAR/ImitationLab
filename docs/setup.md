# Setup

This guide installs ImitationLab and checks that it runs. It is for anyone
running the code for the first time, on a workstation or a cluster node. The
[official SuperMimic runner](#official-supermimic) is optional and has its own
environment.

Run all commands from the repository root. Pixi installs ImitationLab in
editable mode and installs Isaac Lab from a pinned Git revision. The
`imitationlab` command provides the data, pretrain, train, evaluate, and
export workflows.

## Requirements

| Requirement | Details |
| --- | --- |
| Operating system | Linux x86-64. The Isaac Lab environment targets glibc 2.35 or newer |
| GPU | Training requires an NVIDIA GPU and a driver compatible with the CUDA 13.0 PyTorch build |
| Pixi | [Pixi](https://pixi.sh/latest/installation/) 0.83 or newer |
| Disk | Several tens of gigabytes for the simulation stack and package cache |

## Environments

The workspace defines three Pixi environments. Install only the ones you need.

| Environment | Install | Used for |
| --- | --- | --- |
| `isaaclab` | `pixi install --locked -e isaaclab` | Simulation: motion conversion, pretraining, training, evaluation |
| default | `pixi install --locked` | CPU tests and data-bank validation |
| `static` | `pixi install --locked -e static` | Lint and compile checks |

## Isaac Lab

Install the simulation environment:

```bash
pixi install --locked -e isaaclab
```

The repository distributes no motion data, so the smoke test needs a packed
reference bank. Prepare one from the public LAFAN1 retargeting with the
[motion data guide](../data/README.md), then pass its directory as `BANK`:

```bash
BANK=data/reference_arrays/<persist-id> pixi run -e isaaclab smoke-g1
```

The smoke test runs one PPO iteration on that bank and writes a checkpoint
under `outputs/smoke-g1`. It stops with a one-line message when `BANK` is not
set. The first simulation launch may fetch Isaac Sim ground-plane assets and
compile Warp kernels.

**License.** Isaac Sim requires accepting the
[NVIDIA Omniverse License Agreement](https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html).
The `smoke-g1`, `test-isaaclab`, `benchmark` and `tune` commands set
`OMNI_KIT_ACCEPT_EULA=YES` for their simulator processes, so running them
accepts it. Other commands that start Isaac Sim, such as motion conversion,
ask in a terminal; set `OMNI_KIT_ACCEPT_EULA=YES` to run them unattended.

## Check the machine

Inspect the local machine before training. `BANK` is a packed reference bank
from the [motion data guide](../data/README.md).

```bash
pixi run -e isaaclab imitationlab hardware --data "$BANK"
```

Fresh runs use automatic hardware sizing by default; see
[hardware selection](hardware.md) for manual settings and optional
calibration.

## Run the tests

| Command | Covers |
| --- | --- |
| `pixi run --locked test-default` | Learner, data-tool, task-math, runner, and evaluation suites (CPU) |
| `pixi run --locked -e isaaclab test-isaaclab` | G1 environment suite (GPU) |
| `pixi run --locked -e static ci-static` | Lint and compile checks |

## Official SuperMimic

This section is optional. The repository vendors the official SuperMimic
baseline under `third_party/supermimic`. It runs on mjlab rather than Isaac
Lab; its task IDs, data format, and checkpoints belong to the mjlab baseline.

The official runner uses Python 3.13 and HybridRobotics' mjlab fork.

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/),
   then prepare the external simulator beside the vendored SuperMimic
   package. The sibling checkout is ignored by Git.

   ```bash
   git clone https://github.com/HybridRobotics/mjlab_research.git third_party/mjlab_research
   git -C third_party/mjlab_research checkout --detach dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd
   (cd third_party/supermimic && uv sync --locked)
   ```

2. Provide a local motion file in mjlab's tracking format. To convert one
   50 Hz G1 clip produced by `imitationlab data prepare`
   ([motion data](../data/README.md)), run this from the repository root
   after installing the default Pixi environment. Replace the input path
   with your clip; `--start` and `--stop` optionally select a frame range:

   ```bash
   pixi run python scripts/data/convert_mjlab_motion.py \
     data/lafan1/npz/g1/<clip>.npz outputs/motion-mjlab.npz
   ```

3. Run the official generalized-coordinate task from an output directory so
   the runner's relative `logs/` path stays under `outputs/`:

   ```bash
   mkdir -p outputs/official-supermimic
   (
     cd outputs/official-supermimic
     runner=../../third_party/supermimic
     "$runner/.venv/bin/python" "$runner/src/supermimic/scripts/train.py" \
       SuperMimic-GenCoord-G1 --motion-file ../motion-mjlab.npz \
       --runner.run-name g1-baseline
   )
   ```

The official runner also exposes `SuperMimic-MaxCoord-G1` and accepts a W&B
motion artifact. See the
[upstream README](../third_party/supermimic/README.md).

## Next steps

- [Motion data](../data/README.md): download or convert LAFAN1 and BONES-SEED
  and pack a reference bank.
- [Training and evaluation](training.md): pretrain an encoder, train a
  tracker, and evaluate it.
- [Hardware selection](hardware.md): how execution sizes are chosen for your
  GPU.
