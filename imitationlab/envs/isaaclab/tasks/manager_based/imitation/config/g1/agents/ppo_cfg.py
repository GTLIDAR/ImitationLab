"""PPO settings for ``imitation-g1-v1``.

Both command interfaces publish one flat ``features`` term per group, so the
policy reads ``("policy", "features")`` and the critic ``("critic", "features")``.
The packed-data recipe (``experiments/recipes/g1/ppo.args``) adds running input
normalization on top of these defaults.
"""

from isaaclab.utils.configclass import configclass

from imitationlab.algorithms.ppo import PPOTrainerConfig


@configclass
class G1PPOConfig(PPOTrainerConfig):
    """PPO for G1 tracking: five epochs of four minibatches per 24-step rollout."""

    def __post_init__(self):
        super().__post_init__()

        assert isinstance(self, PPOTrainerConfig)
        assert self.value_function is not None, (
            "Value function configuration must be provided."
        )

        self.policy.input_keys = [("policy", "features")]
        self.value_function.input_keys = [("critic", "features")]

        self.collector.init_random_frames = 0
        self.collector.frames_per_batch = 24
        self.replay_buffer.size = 4096 * 24

        self.loss.epochs = 5
        self.loss.mini_batch_size = 4096 * 24 // 4
        self.loss.loss_critic_type = "l2"

        self.ppo.clip_epsilon = 0.2
        self.ppo.gae_lambda = 0.95
        self.ppo.entropy_coeff = 0.005
        self.ppo.critic_coeff = 1.0
        self.ppo.clip_value = True
        self.ppo.normalize_advantage = True
        self.ppo.clip_log_std = False
        self.ppo.log_std_init = 0.0

        self.optim.lr = 1.0e-3
        self.optim.max_grad_norm = 1.0
        self.optim.scheduler = "adaptive"
        self.optim.desired_kl = 0.01

        self.loss.gamma = 0.99

        self.policy.num_cells = [512, 256, 128]
        self.value_function.num_cells = [512, 256, 128]

        self.collector.total_frames = 10_000_000_000
        self.save_interval = 5_000_000  # samples
        self.compile.compile = False
        self.collector.no_cuda_sync = True
        self.trainer.progress_bar = True
        self.trainer.log_interval = 10_000_000  # samples
