# Policy learners

| Location | Responsibilities |
| --- | --- |
| [base.py](base.py) | Learner contract for training, prediction, and checkpoints |
| [ppo/](ppo) | Policy and value updates with GAE, optional recurrent policies, and distributed PPO |
| [supermimic/](supermimic) | Backpropagation through imagined trajectories |
| [svg/](svg) | Real-window rollouts with inferred action noise and dynamics residuals |
| [diffsr/](diffsr) | Policy updates through a differentiable spectral dynamics readout |
| [common/](common) | Configuration, collection, network utilities, and value targets |

[The training runtime](../runners/algorithm_runtime.py) handles collection,
metrics, checkpointing, and time budgets. The world-model learners share
[model fitting and replay](common/world_model). The
[task protocol](common/world_model/task.py) lets the task supply the same
observation, action, reward, and reset definitions to simulated and imagined
trajectories.

Latent interfaces and encoder pretraining live in [encoders/](../encoders).
Use the [training guide](../../docs/training.md) for launcher commands. The CPU
algorithm tests live in [tests/](tests). See [LICENSE](LICENSE) and the
[third-party notices](THIRD_PARTY_NOTICES.md) for the code's provenance.
