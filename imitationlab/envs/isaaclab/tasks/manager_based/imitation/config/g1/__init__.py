"""Gym registration of the public G1 tracking task, ``imitation-g1-v1``."""

import gymnasium as gym

from . import agents, common

__all__ = ["agents", "common"]

gym.register(
    id="imitation-g1-v1",
    entry_point="imitationlab.envs.isaaclab.envs.g1_env:make_imitation_g1_v1",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.imitation_g1_env:ImitationG1EnvCfg",
        "agent_cfg_entry_point": f"{agents.__name__}.ppo_cfg:G1PPOConfig",
        "ppo_cfg_entry_point": f"{agents.__name__}.ppo_cfg:G1PPOConfig",
        "wmbptt_cfg_entry_point": f"{agents.__name__}.wmbptt_cfg:G1WMBPTTConfig",
    },
)
