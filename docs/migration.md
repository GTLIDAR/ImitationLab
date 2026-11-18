# Migrating to the release layout

ImitationLab installs one package and one command. Use `imitationlab train`,
`pretrain`, `evaluate`, `export`, and `data` through the documented Pixi
environment. `python -m imitationlab` invokes the same CLI.

## One task: `imitation-g1-v1`

`imitation-g1-v1` is the only registered task. Its observations, actions,
rewards, termination rules, reset sampling, control rate, physics, agent
defaults, and checkpoint tensors are unchanged. Checkpoints trained on
`imitation-g1-v1` load and evaluate as before. The explicit and latent
command interfaces, the eager and captured control steps, and the pinned-reset
evaluation are settings of this one task:

| Previous task ID | Now |
| --- | --- |
| `Isaac-Imitation-G1-v3`, `Spectra-G1-v1` | `imitation-g1-v1` |
| `Isaac-Imitation-G1-Latent-v3`, `Spectra-G1-Latent-v1` | `imitation-g1-v1` with `--interface fsq` or `spectral` (`env.skill_encoder_path`) |
| `Isaac-Imitation-G1-Graph-v3` | `imitation-g1-v1` with `env.step_mode=graph` (or `masked`) |
| `Isaac-Imitation-G1-RootBoard-v3`, `Spectra-G1-Board-v1` | `imitationlab evaluate`: `env.pinned_reset=true`, no reset or observation noise, no context randomization |

The earlier designs are not registered and have no alias: the `-v0`, old
`-v1`, and `-v2` task families, the SONIC baseline tasks, the SONIC failure-rule
boards (`Board-v3`), and the NDM Gym recipe tasks (`BlinkMimic-v3`,
`FastSuperMimic-v3`). Their configurations and environment classes remain in
the history of this repository at
[`fbe5b63`](https://github.com/fei-yang-wu/ImitationLab/commit/fbe5b63).

## Renamed modules, classes, and settings

| Previous name | Current name |
| --- | --- |
| `envs.imitation_rl_env_v3` (`G1V3RLEnv`, `G1V3GraphRLEnv`, `G1V3LatentRLEnv`) | `envs.g1_env` (`G1RLEnv`, `G1GraphRLEnv`, `G1LatentRLEnv`) |
| `envs.imitation_rl_env_v2` (`ImitationRLEnv`) | `envs.imitation_rl_env` (`ImitationRLEnv`) |
| `envs.graph_step.g1_v3` | `envs.graph_step.g1` |
| `task_model.g1_v3_task`, `g1_v3_math`, `g1_v3_contact` | `task_model.g1_task`, `g1_math`, `g1_contact` |
| `G1V3TrackingTask`, `G1V3LatentTrackingTask`, `G1V3MDP`, `G1V3ReferenceLibrary` | `G1TrackingTask`, `G1LatentTrackingTask`, `G1MDP`, `G1ReferenceLibrary` |
| `contracts.g1_v3_reset_sampling` (`G1V3ResetSampler`) | `contracts.g1_reset_sampling` (`G1ResetSampler`) |
| `config.g1.imitation_g1_v1` (`ImitationG1V1EnvCfg`) | `config.g1.imitation_g1_env` (`ImitationG1EnvCfg`) |
| `agents.rlopt_ppo_v3_cfg` (`G1V3RLOptPPOConfig`) | `agents.rlopt_ppo_cfg` (`G1RLOptPPOConfig`) |
| `agents.rlopt_wmbptt_v3_cfg` (`ImitationG1V1WMBPTTConfig`) | `agents.rlopt_wmbptt_cfg` (`G1RLOptWMBPTTConfig`) |
| `mdp.g1_v3` | `mdp.g1` |
| `evaluation.world_model.g1_v3_eval_probe` | `evaluation.world_model.g1_eval_probe` |
| `env.v3_library_host_mapped`, `_host_shared`, `_shared_file`, `_managed`, `_env_window` | `env.library_host_mapped`, `_host_shared`, `_shared_file`, `_managed`, `_env_window` |
| `env.v3_graph_reference_source` | `env.graph_reference_source` |
| W&B reward parts `V3/<term>` | `Reward/<term>` |

Module paths are relative to `imitationlab.envs.isaaclab` (configs under
`tasks.manager_based.imitation`). The recipes in `experiments/recipes/` use the
current setting names. The world-model goal bindings recorded in checkpoints
(`g1-v3`, `g1-v3-latent`) keep their values, so existing checkpoints still bind.

These environment settings did nothing on `imitation-g1-v1` and were removed:
`static_collector`, `replay_reference`, `replay_only`,
`enable_termination_curriculum`, `termination_curriculum_start_frames`,
`termination_curriculum_end_frames`, `enable_reward_input_observations`,
`enable_visualizers`, `visualize_reference_arrows`, `print_reference_velocity`,
`print_reference_velocity_every`, and `_debug_rewards`.

## Removed tools

The ONNX exporter (`imitationlab export --format onnx`) supported only
observation layouts of earlier task designs; `imitationlab export` writes the
native bundle. The trainer flags `--termination_window*`,
`--match-sonic-release-overrides`, and `--train_anchor_override`, the
reference-replay visualizer, and the world-model dataset and dynamics-comparison
tools of the `-v2` world-model task are removed with the designs they served.

## Encoder pretraining

`imitationlab pretrain` samples reference windows on `imitation-g1-v1`. The
windows are the same as before (the same clips, frames, and 38-wide macro
frames). The environment's construction draws a different number of random
values first, so a pretraining run with a given seed produces a different
encoder than earlier releases did.

## Imports and entry points

New code uses the `imitationlab` package. The compatibility packages
`algorithms`, `data_tools`, `isaaclab_imitation`, and `imitation_experiments`
and their import aliases were removed:

| Previous location | Current owner |
| --- | --- |
| `algorithms.ppo`, `algorithms.supermimic` | `imitationlab.algorithms.ppo`, `imitationlab.algorithms.supermimic` |
| Shared world-model machinery under `algorithms.supermimic` | `imitationlab.algorithms.common.world_model` |
| `algorithms.common.skill_encoder` | `imitationlab.encoders.skill_encoder` |
| `algorithms.diffsr.pretrain` | `imitationlab.encoders.diffsr.pretrain` |
| `data_tools` | `imitationlab.data_tools` |
| `isaaclab_imitation` | `imitationlab.envs.isaaclab` |
| `imitation_experiments.world_model` task math | `imitationlab.envs.isaaclab.task_model` |
| `imitation_experiments.evaluation` | `imitationlab.evaluation` |
| Experiment launch and distributed utilities | `imitationlab.runners` |

The scripts under `scripts/` run the current modules. Importing the
environment package is lightweight; import `imitationlab.envs.isaaclab.tasks`
or call `register_tasks()` to register the Gym task.
`imitationlab.algorithms.base.BaseAlgorithm` is the small learner contract.

## Checkpoints and data

New training runs write `run.json` and embed the same run description in
checkpoints. It records the task, learner, interface, seed, data identity,
encoder hash, resolved arguments, and execution settings. Resume restores
the saved settings; explicit trailing overrides take precedence.

```bash
pixi run -e isaaclab imitationlab train \
  --checkpoint /path/to/run --max-iterations 10000
pixi run -e isaaclab imitationlab evaluate \
  --checkpoint /path/to/run --data /path/to/evaluation-bank
```

Runs without `run.json` recover their launch settings from `command.txt` and
their seed from `params/agent.yaml`; keep those files with the checkpoints. A
standalone checkpoint without a run description needs explicit learner, data,
and architecture settings. Native export bundles preserve checkpoint tensors
and package the encoder with verified identity. Motion banks remain separate,
including proprietary BONES-SEED assets.

Evaluation uses pinned starts and eager stepping, and disables adaptive training
reset sampling. Training keeps its captured execution defaults. Data replacement
clears the previous source and cache selection so a new manifest cannot silently
reuse the prior bank.
