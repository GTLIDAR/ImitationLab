# Motion data tools

ImitationLab's data package prepares reference motions and manages replay
storage, trajectory cursors, and reset frames. See the
[motion data guide](../../data/README.md) for dataset access and conversion
commands. Pixi installs the dependencies from the repository root.

| Module | Purpose |
| --- | --- |
| `imitationlab.data_tools.bones_seed` | Normalize raw G1 CSVs and apply the public SONIC motion-name filter |
| `imitationlab.data_tools.reference_arrays.build` | Pack an NPZ motion manifest into validated memory-mapped arrays |
| `imitationlab.data_tools.reference_arrays.publish` | Transfer packed banks through authorized Hugging Face storage |
| `imitationlab.data_tools.lafan1.loader` | Resample CSV or NPZ motions and write a Zarr store |
| `imitationlab.data_tools.common.replay` | Build or reopen a TorchRL replay buffer from a Zarr store |
| `imitationlab.data_tools.common.trajectories` | Track a reference trajectory and frame per environment |
| `imitationlab.data_tools.common.reset_sampling` | Sample start frames, including failure-adaptive SONIC sampling |
| `imitationlab.data_tools.common.registry` | Resolve named dataset loaders, including the `lafan1` and `lafan1_csv` aliases |

## Commands

Run `imitationlab data --help` for preparation, packing, and validation.
Dataset preparation keeps completed batches and can resume from the same
work directory. Packing writes the manifest last, so an interrupted pack
does not appear to be a complete bank.

## LAFAN1 loader

`Lafan1CsvLoader` reads the CSV layout of the retargeted LAFAN1 release for the
Unitree G1 (one row per frame):

- columns `0:3`: root position in the world frame,
- columns `3:7`: root orientation as an `xyzw` quaternion (stored as `wxyz`),
- columns `7:`: joint positions.

NPZ files with `joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w`, ... and an
`fps` entry are accepted as well. Motions are resampled to `control_freq` and
root/joint velocities are computed by finite differences. The motion data
itself is distributed under its own license and is not part of this package.

```python
from imitationlab.data_tools.lafan1.loader import Lafan1CsvLoader

cfg = {
    "dataset": {
        "trajectories": {
            "lafan1_csv": [
                "/path/to/motions/walk1_subject1.csv",
                {"path": "/path/to/motions/dance1_subject2.csv",
                 "input_fps": 30, "frame_range": [1, 1200]},
                {"name": "run_combo",
                 "paths": ["/path/to/motions/run1_subject2.csv",
                           "/path/to/motions/run2_subject1.csv"]},
            ]
        }
    },
    "joint_names": [...],   # CSV column order; defaults to joint_0, joint_1, ...
    "input_fps": 30,        # default source rate for entries without input_fps
    "control_freq": 50,     # output rate written to the store
}
loader = Lafan1CsvLoader(cfg=cfg, build_zarr_dataset=True,
                         zarr_path="/path/to/lafan1.zarr")
print(loader.metadata)      # DatasetMeta: trajectory lengths, dt, joint names, keys
```

Each entry of `lafan1_csv` is a path, a `{"path": ..., "input_fps": ...,
"frame_range": [start, end]}` dict (1-indexed, inclusive), or a
`{"name": ..., "paths": [...]}` group that stores several files as
trajectories of one motion. `cfg` may be a plain mapping or any object with
attribute access (an OmegaConf `DictConfig` works). Set
`canonical_joint_names` to write every motion in one joint order.

The store is laid out as `<dataset>/<motion>/trajectory_<i>/<key>`, with the
per-frame arrays `qpos`, `qvel`, `root_pos`, `root_quat`, `root_lin_vel`,
`root_ang_vel`, `joint_pos`, `joint_vel`, the corresponding `next_*` arrays,
and `body_*_w` arrays when the source provides them. The dataset group's
attributes carry the trajectory manifest, `dt`, `control_freq` and the joint
names.

## Replay buffer and trajectory manager

```python
import torch

from imitationlab.data_tools.common.trajectories import ParallelTrajectoryManager, ResetSchedule
from imitationlab.data_tools.common.reset_sampling import StartFrameSampler
from imitationlab.data_tools.common.replay import make_rb_from

rb, info = make_rb_from(
    "/path/to/lafan1.zarr",
    datasets="lafan1",
    keys=["qpos", "qvel", "joint_pos", "joint_vel", "root_pos", "root_quat"],
    persist_dir="/path/to/lafan1_rb",   # optional memmap copy, reopened on later runs
)

num_envs = 4096
manager = ParallelTrajectoryManager(
    rb=rb,
    traj_info=info,
    num_envs=num_envs,
    reset_schedule=ResetSchedule.RANDOM,   # or SEQUENTIAL, ROUND_ROBIN, CUSTOM
    reference_joint_names=loader.metadata.joint_names,
    target_joint_names=robot_joint_names,  # the articulation's joint order
    reset_generator=torch.Generator().manual_seed(0),
)
start_frames = StartFrameSampler(manager.length, mode="uniform")

env_ids = torch.arange(num_envs)
manager.reset_envs(env_ids)                                     # pick trajectories
steps = start_frames.sample_steps(manager.env_traj_rank[env_ids])
manager.reset_envs(env_ids, steps=steps)                        # pick start frames

reference = manager.sample()      # TensorDict [num_envs] of the current frame
# ... step the simulation; manager.sample() advanced every cursor by one frame
rank = manager.get_traj_rank("lafan1", "walk1_subject1", "trajectory_0")
```

`make_rb_from` fills a CPU memmap (or GPU) `TensorDictReplayBuffer` with every
selected frame and returns, alongside it, `start_index`, `end_index` and
`ordered_traj_list`, which the manager uses to address trajectories by rank.
With `persist_dir`, a later call memory-maps the filled buffer instead of
walking the Zarr store again; pass `persist_rebuild=True` after the store
changes.

`ParallelTrajectoryManager` keeps a `(trajectory_rank, local_step)` cursor per
environment. `sample()` indexes the buffer storage directly with the global
indices of all cursors (one frame per environment) and, by default, advances
them; `sample_slice()` returns a window per environment; `advance_cursors()`
moves cursors without reading storage; `reset_envs()` reassigns trajectories
according to `reset_schedule` and sets the start frame. Both joint-name lists
are required: the manager maps the reference joint order to the target order
and marks joints the reference does not provide.

## Reset sampling

Trajectory selection belongs to the manager. `StartFrameSampler` answers the
second question, which frame to start at, in four modes: `fixed`, `random`
(uniform in a configured range), `uniform` (uniform over each trajectory's own
length) and `adaptive` (proportional to a caller-supplied
`weight_fn(ranks, steps)`).

`SonicAdaptiveResetSampler` is a failure-aware sampler: it keeps per-bin visit
and failure counts, mixes the failure-rate distribution with a uniform floor,
and can be used standalone (`sample(count)` returns ranks and frames jointly)
or as the `weight_fn` of `StartFrameSampler`.

```python
from imitationlab.data_tools.common.reset_sampling import SonicAdaptiveResetSampler, StartFrameSampler

adaptive = SonicAdaptiveResetSampler(manager.length, bin_size=50)
adaptive.record_visits(manager.env_traj_rank, manager.env_step)
adaptive.record_failures(manager.env_traj_rank[failed], manager.env_step[failed])

start_frames = StartFrameSampler(manager.length, mode="adaptive", weight_fn=adaptive)
```

## Tests

```bash
pixi run --locked python -m pytest -q imitationlab/data_tools/tests
```

The tests build small synthetic CSV / NPZ motions in a temporary directory, so
no dataset download is needed.

## Third-party attributions

- The CSV layout read by `Lafan1CsvLoader` and its preprocessing assumptions
  (root position, `xyzw` root quaternion, joint positions; resampling and
  finite-difference velocities) follow `scripts/mimic/csv_to_npz.py` of
  `unitree_rl_lab` (Unitree Robotics), as noted in
  `lafan1/loader.py`.
- `SonicAdaptiveResetSampler` reimplements the failure-rate-weighted
  motion/frame reset sampling of SONIC's public motion library.

## License

MIT, see [LICENSE](LICENSE).
