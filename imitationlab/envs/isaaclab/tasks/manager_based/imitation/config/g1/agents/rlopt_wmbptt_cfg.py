"""World-model learner settings for ``imitation-g1-v1`` (learned model + BPTT).

The agent reads only the env's clean ``wm`` observation group and builds its
policy features, reward and termination itself through the task model
(``imitationlab.envs.isaaclab.task_model.g1_task``). The SuperMimic, SVG and
DiffSR learners share these defaults; the launcher and the recipes in
``experiments/recipes/g1/`` select the learner and its packed-data settings.
"""

from isaaclab.utils.configclass import configclass

from imitationlab.algorithms.supermimic import WMBPTTTrainerConfig

EXPLICIT_TASK = "imitationlab.envs.isaaclab.task_model.g1_task:G1TrackingTask"
LATENT_TASK = "imitationlab.envs.isaaclab.task_model.g1_task:G1LatentTrackingTask"


@configclass
class G1RLOptWMBPTTConfig(WMBPTTTrainerConfig):
    """SuperMimic recipe: k = 8, H = 32, 4096 imagined rollouts, gamma 0.99."""

    _command_interface = None
    command_space: str = "world_model"

    def __post_init__(self):
        super().__post_init__()
        self.policy.input_keys = [
            ("wm", "root_state"),
            ("wm", "joint_state"),
            ("wm", "last_action"),
            ("wm", "reference_cursor"),
            ("wm", "sim_context"),
            ("wm", "proprio_history"),
        ]
        self.value_function = None
        self.loss.gamma = 0.99
        self.collector.total_frames = 10_000_000_000
        self.collector.frames_per_batch = 1
        self.save_interval = 50_000_000  # frames
        self.trainer.progress_bar = False
        self.wm.task_model = EXPLICIT_TASK
        self.wm.goal = "explicit"
        self.wm.action_clip = 20.0

    def sync_input_keys(self) -> None:
        """The agent reads the fixed ``wm`` group; nothing to derive."""
        return

    def bind_env_config(self, env_cfg):
        """A frozen command encoder on the env selects the latent task model."""
        if env_cfg.skill_encoder_path and self.wm.task_model == EXPLICIT_TASK:
            self.wm.task_model = LATENT_TASK
            self.wm.goal = "z"
