# ruff: noqa: E402

"""Script to play a checkpoint of a PPO agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher
from imitationlab.cli_args import add_kebab_aliases
from imitationlab.runners.launch_compat import add_launcher_args

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Play a checkpoint of an ImitationLab agent."
)
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during play."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default="agent_cfg_entry_point",
    help="ImitationLab agent configuration entry point.",
)
parser.add_argument(
    "--algo",
    "--algorithm",
    dest="algorithm",
    type=str.upper,
    default="PPO",
    choices=["PPO"],
    help="ImitationLab algorithm (must match the checkpoint).",
)
parser.add_argument(
    "--checkpoint", type=str, default=None, help="Path to model checkpoint (.pt)."
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Optional log/video output directory for this play run.",
)
parser.add_argument(
    "--seed", type=int, default=None, help="Seed used for the environment."
)
parser.add_argument(
    "--real-time",
    action="store_true",
    default=False,
    help="Run in real-time, if possible.",
)
# append launcher cli args
add_launcher_args(parser)
add_kebab_aliases(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import random
import time

import gymnasium as gym
import torch
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)
from imitationlab.envs.isaaclab.envs.torchrl_adapter import (
    IsaacLabWrapper,
    attach_terminal_obs_reader,
)
from imitationlab.runners.launch_compat import hydra_task_config
from imitationlab.evaluation.video_capture import attach_video_capture
from imitationlab.algorithms import PPO
from imitationlab.runners.registry import resolve_agent_cfg_entry_point
from torchrl.envs import Compose, RewardClipping, RewardSum, StepCounter, TransformedEnv
from torchrl.envs.utils import set_exploration_type, step_mdp
from tensordict.nn import InteractionType

import isaaclab_tasks  # noqa: F401
import imitationlab.envs.isaaclab.tasks  # noqa: F401

ALGORITHM_CLASS_MAP = {
    "PPO": PPO,
}


agent_entry_point = resolve_agent_cfg_entry_point(
    args_cli.task, args_cli.algorithm, args_cli.agent
)


@hydra_task_config(args_cli.task, agent_entry_point)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg,
):
    """Play with an ImitationLab agent."""
    # Same binding as training: the environment's command interface decides the
    # agent's input keys, so a checkpoint is replayed against the contract it
    # was trained on.
    from imitationlab.envs.isaaclab.tasks.manager_based.imitation.command_interface import (
        bind_command_interface,
    )

    if bind_command_interface(agent_cfg, env_cfg) is None:
        sync_input_keys = getattr(agent_cfg, "sync_input_keys", None)
        if callable(sync_input_keys):
            sync_input_keys()

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = (
        args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    )
    agent_cfg.env.num_envs = env_cfg.scene.num_envs
    agent_cfg.env.env_name = args_cli.task
    agent_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_cfg.seed
    agent_cfg.collector.frames_per_batch *= env_cfg.scene.num_envs
    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )

    # validate checkpoint
    if args_cli.checkpoint is None:
        raise ValueError("--checkpoint is required for play.")
    checkpoint_path = os.path.abspath(args_cli.checkpoint)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if args_cli.output_dir is None:
        log_dir = os.path.dirname(checkpoint_path)
    else:
        log_dir = os.path.abspath(args_cli.output_dir)
        os.makedirs(log_dir, exist_ok=True)
    env_cfg.log_dir = log_dir
    # Playback is an inference surface, not a training run.  Keep it local so
    # loading a checkpoint cannot create an unrelated W&B run from the agent
    # config's training defaults.
    agent_cfg.logger.backend = "csv"
    agent_cfg.logger.log_dir = log_dir
    agent_cfg.logger.video = False

    # The env records video itself from ``env_cfg.video_recorders``; configure it
    # before the env exists. One clip starts at the first step.
    video_capture = None
    if args_cli.video:
        video_capture = attach_video_capture(
            env_cfg,
            output_dir=os.path.join(log_dir, "videos", "play"),
            video_length=args_cli.video_length,
        )
        print(f"[INFO] Recording videos during play to: {video_capture.directory}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    if isinstance(env.unwrapped, DirectMARLEnv):
        raise NotImplementedError(
            "DirectMARLEnv is not supported for ImitationLab play."
        )

    env = IsaacLabWrapper(env)
    env = attach_terminal_obs_reader(env)
    env = TransformedEnv(
        base_env=env,
        transform=Compose(
            RewardSum(),
            StepCounter(1000),
            RewardClipping(-10.0, 5.0),
        ),
    )

    # create agent and load checkpoint
    agent_class = ALGORITHM_CLASS_MAP[args_cli.algorithm]
    agent = agent_class(env=env, config=agent_cfg)

    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    agent.load_model(checkpoint_path)

    # switch to eval / inference mode
    collector_policy = getattr(agent, "deployment_policy", None)
    if collector_policy is None:
        collector_policy = agent.collector_policy

    collector_policy.eval()

    dt = getattr(env, "step_dt", None)
    if dt is None:
        # fall back: try to get it from the unwrapped env
        base = env
        while hasattr(base, "env") or hasattr(base, "base_env"):
            base = getattr(base, "env", None) or getattr(base, "base_env", None)
            if base is None:
                break
            if hasattr(base, "step_dt"):
                dt = base.step_dt
                break

    # reset environment
    td = env.reset()
    timestep = 0
    print("[INFO] Starting inference loop. Press Ctrl+C to stop.")

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        with (
            torch.inference_mode(),
            set_exploration_type(InteractionType.DETERMINISTIC),
        ):
            td = collector_policy(td)
            td = env.step(td)
            td = step_mdp(
                td, exclude_reward=True, exclude_done=False, exclude_action=True
            )

        timestep += 1
        if args_cli.video and timestep >= args_cli.video_length:
            break

        # time delay for real-time evaluation
        if args_cli.real_time and dt is not None:
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    # close the simulator
    env.close()
    if video_capture is not None:
        # ``env.close()`` writes the last clip. Print every retained path.
        video_capture.report()


if __name__ == "__main__":
    main()
    simulation_app.close()
