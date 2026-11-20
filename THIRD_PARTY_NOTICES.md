# Third-party notices

## NDM Gym, BlinkMimic and InstantTrack

The G1 execution and world-model optimizations port code from
[NDM Gym](https://github.com/Frank-ZY-Dou/ndm-gym) @ `05ea9f3`, which ports it
from [BlinkMimic](https://github.com/fuyichen0725/blinkmimic) @ `d264658`, itself
derived from [InstantTrack](https://github.com/Huy-Quang-Dao/instantTrack). All
are under the MIT License in `third_party/licenses/ndm_gym.txt`. The native
learners preserve their upstream notices in `imitationlab/algorithms/THIRD_PARTY_NOTICES.md`.

| file here | source in NDM Gym |
|---|---|
| `imitationlab/envs/isaaclab/task_model/g1_math.py` (`reward_kernel_scale`) | `--reward_std_scale` and `training_reward_cfg` in `ndm_gym/train.py` |
| `imitationlab/envs/isaaclab/task_model/g1_math.py` (`drift_weight`, `semi_implicit_drift`, `structured_targets`, `structured_integrate`) | `SemiImplicitIncrementWorldModel` (`DRIFT_W`, `drift`, `targets`, `integrate`) in `ndm_gym/worldmodels/semi_implicit.py`, from BlinkMimic |
| `imitationlab/envs/isaaclab/task_model/g1_math.py` (`mirror_joint_map`, `G1Mirror`) | `Mirror` in `ndm_gym/algos/blinkmimic.py`, from BlinkMimic, built by joint name for this repository's joint order |
| `imitationlab/envs/isaaclab/task_model/g1_task.py` (`reference_setpoint`) | `reference_setpoints` in `ndm_gym/algos/common.py`, `SuperMimic.setpoint` in `ndm_gym/algos/supermimic.py` (`--sm_residual`) |
| `imitationlab/envs/isaaclab/task_model/g1_task.py` (`perturb_start`) | `perturb_start` in `ndm_gym/algos/common.py` (`--imag_start_perturb`) |
| `imitationlab/envs/isaaclab/task_model/g1_contact.py` | `ndm_gym/tasks/contact.py` (`--contact_reward`, `--ref_contact_fix`), from InstantTrack and BlinkMimic; its per-clip quantile loop as one sort |
| `imitationlab/envs/isaaclab/task_model/g1_math.py` (`G1MDP.reward`'s `contact`) and `g1_task.py` (`reward_and_termination`) | `reward(..., contact=...)` in `ndm_gym/tasks/tracking.py`, `SuperMimic._reward_term` in `ndm_gym/algos/supermimic.py` |
| `imitationlab/envs/isaaclab/contracts/g1_reset_sampling.py` (`fail_share`) | `previous_bins`, `bin_ema` and `mix_failures` in `ndm_gym/sim/rigid/reset_stats.py`; the update order in `ndm_gym/sim/rigid/isaaclab_mjwarp.py` |
| `imitationlab/envs/isaaclab/contracts/g1_reset_sampling.py` (`*_masked`) | `bin_ema_w` in `ndm_gym/sim/rigid/reset_stats.py`; the gated `sample_frames(N, gate)` in `ndm_gym/sim/rigid/isaaclab_mjwarp.py` |
| `imitationlab/envs/isaaclab/envs/graph_step/g1.py` | `reset_masked` (all-environment draws, masked writes) in `ndm_gym/sim/rigid/isaaclab_mjwarp.py` |
| `imitationlab/envs/isaaclab/contracts/g1_reset_sampling.py` (`enable_focus`, `set_focus`, `_mix_focus`) | `mix_focus` in `ndm_gym/sim/rigid/reset_stats.py`; `set_focus` and `use_static_focus` in `ndm_gym/sim/rigid/isaaclab_mjwarp.py` (`--eprobe_share`), from BlinkMimic |
| `imitationlab/evaluation/world_model/g1_eval_probe.py` | `ndm_gym/algos/eprobe.py` (`probe_tiles`, `sweep_frames`, `next_bins`, `FailureFocus`, `ProbeSweep`, `EvalProbe`, `worker`; `--eprobe_*`), from BlinkMimic, on (rank, local frame) and this repository's bins |
| `imitationlab/evaluation/grouped_eval.py` (`grouped_rank_assignment`, `BlockPolicy`, `rollout`) | `evaluate_group` in `ndm_gym/evaluate.py` (`--group`, `--repeats`): the world layout `(m * repeats + r) * C + c` and each checkpoint's policy on its own block |
| `imitationlab/evaluation/replica_readout.py` (`mean_of_repeats`) | `mean_of_repeats` in `ndm_gym/evaluate.py` |

## Official SuperMimic

The code in [third_party/supermimic](third_party/supermimic) is from
[HybridRobotics/supermimic](https://github.com/HybridRobotics/supermimic).
It retains its [Apache-2.0 license](third_party/supermimic/LICENSE).

## Unitree G1 and the earlier SuperMimic replica

The G1 assets retain their bundled upstream notices under
[imitationlab/envs/isaaclab/assets/unitree](imitationlab/envs/isaaclab/assets/unitree).
The `imitation-g1-v1` environment credits the MIT-licensed
[supermimic_vibed](https://github.com/zita-ch/supermimic_vibed) replica.
See [its license](third_party/licenses/supermimic_vibed.txt).

## Isaac Lab

The following files are derived from
[Isaac Lab](https://github.com/isaac-sim/IsaacLab) and carry its BSD-3-Clause
license header:

- `imitationlab/data_tools/common/isaac_batch.py`
- `imitationlab/data_tools/lafan1/hub.py`
- `imitationlab/data_tools/lafan1/manifest.py`
- `imitationlab/data_tools/lafan1/prepare.py`
- `imitationlab/data_tools/lafan1/setup.py`
- `imitationlab/envs/isaaclab/__init__.py`
- `imitationlab/envs/isaaclab/assets/robots/unitree.py`
- `imitationlab/envs/isaaclab/contracts/__init__.py`
- `imitationlab/envs/isaaclab/contracts/command_channels.py`
- `imitationlab/envs/isaaclab/envs/imitation_interface.py`
- `imitationlab/envs/isaaclab/envs/torchrl_adapter.py`
- `imitationlab/envs/isaaclab/tasks/__init__.py`
- `imitationlab/envs/isaaclab/tasks/manager_based/__init__.py`
- `imitationlab/envs/isaaclab/tasks/manager_based/imitation/_resolve.py`
- `imitationlab/envs/isaaclab/tasks/manager_based/imitation/command_components.py`
- `imitationlab/envs/isaaclab/tasks/manager_based/imitation/command_interface.py`
- `imitationlab/envs/isaaclab/tasks/manager_based/imitation/config/g1/common/actions.py`
- `imitationlab/envs/isaaclab/tasks/manager_based/imitation/config/g1/common/constants.py`
- `imitationlab/envs/isaaclab/tasks/manager_based/imitation/mdp/commands/actor.py`
- `imitationlab/envs/isaaclab/tasks/manager_based/imitation/mdp/commands/reference.py`
- `imitationlab/envs/isaaclab/tasks/manager_based/imitation/motion_data.py`
- `imitationlab/evaluation/tests/test_backend_determinism.py`
- `imitationlab/evaluation/validation/backend_determinism.py`
- `imitationlab/runners/zero_agent.py`
- `scripts/list_envs.py`
- `scripts/random_agent.py`
