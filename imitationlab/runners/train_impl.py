"""ImitationLab training implementation loaded after simulation runtime selection."""

# ruff: noqa: E402
import argparse
import json
import logging
import os
import re
import signal
import sys
import uuid
import warnings
from pathlib import Path

import torch


_sigint_seen = False


def cleanup_pbar(signum, _frame):
    """Handle Ctrl+C / SIGTERM quickly and safely.

    Keep the handler minimal to avoid exceptions inside unrelated callback
    contexts (e.g. Isaac Sim GC hooks) and ensure the first signal stops
    training. The first signal raises ``KeyboardInterrupt``; the training loop
    then writes its resume checkpoint. Repeated SIGTERMs are ignored from then
    on: Slurm signals every process of the job step, so the launcher (pixi,
    apptainer) forwards a second copy right behind the first, and that copy
    used to interrupt the checkpoint write (measured 2026-10-10: TERM to python
    and pixi together left no checkpoint). A second Ctrl+C still aborts.
    """
    global _sigint_seen
    if not _sigint_seen:
        _sigint_seen = True
        name = "SIGTERM" if signum == signal.SIGTERM else "Ctrl+C"
        print(f"\n[INFO] {name} received. Stopping training...")
        # Restore default behavior for any subsequent interrupt during shutdown.
        signal.signal(signal.SIGINT, signal.default_int_handler)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raise KeyboardInterrupt


# disable KeyboardInterrupt override
signal.signal(signal.SIGINT, cleanup_pbar)
# Slurm walltime expiry delivers SIGTERM (early, when sbatch passes
# --signal=TERM@<sec>). Route it through the same interrupt path so the
# training loop can write a final checkpoint before the job dies -- segmented
# runs resume from that checkpoint instead of losing everything since the
# last save_interval boundary.
signal.signal(signal.SIGTERM, cleanup_pbar)

import random
import time
from datetime import datetime

import gymnasium as gym
import numpy as np
import wandb
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)
from isaaclab.utils.io import dump_yaml
from imitationlab.envs.isaaclab.contracts.g1_reset_sampling import (
    install_reset_stats_persistence,
)
from imitationlab.envs.isaaclab.envs.torchrl_adapter import (
    IsaacLabWrapper,
    attach_terminal_obs_reader,
)
from imitationlab.evaluation.video_capture import attach_video_capture
from imitationlab.algorithms import PPO, WMBPTT
from imitationlab.algorithms.common.world_model.speed_options import (
    apply_compile_options,
)
from imitationlab.algorithms.common.config import AlgorithmConfig, TrainerConfig
from torchrl.envs import (
    Compose,
    RewardSum,
    StepCounter,
    TransformedEnv,
)

torch.set_float32_matmul_precision("high")

# Suppress known third-party deprecations until upstream packages update.
warnings.filterwarnings(
    "ignore",
    message=r"Read the `app_url` setting from the appropriate Settings object\.",
    category=DeprecationWarning,
    module=r"wandb\.analytics\.sentry",
)
warnings.filterwarnings(
    "ignore",
    message=r"The `Scope\.user` setter is deprecated in favor of `Scope\.set_user\(\)`\.",
    category=DeprecationWarning,
    module=r"wandb\.analytics\.sentry",
)

logger = logging.getLogger(__name__)
logging.getLogger("imitationlab.data_tools").setLevel(logging.WARNING)

ALGORITHM_CLASS_MAP = {"PPO": PPO, "WMBPTT": WMBPTT}


def _render_frame_to_numpy(frame):
    """Convert render outputs to contiguous CPU uint8 arrays for video logging."""
    if isinstance(frame, list):
        if len(frame) == 0:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        frame = frame[-1]
    if isinstance(frame, torch.Tensor):
        frame = frame.detach()
        if frame.is_cuda:
            frame = frame.to("cpu")
        if frame.dtype != torch.uint8:
            frame = frame.to(torch.uint8)
        frame = frame.numpy()
    else:
        frame = np.asarray(frame)
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8, copy=False)
    return np.ascontiguousarray(frame)


def _infer_render_fps(env: object, default_fps: int = 30) -> int:
    """Infer render FPS from env metadata (falls back to default_fps)."""
    stack: list[object] = [env]
    visited: set[int] = set()
    while len(stack) > 0:
        current = stack.pop()
        obj_id = id(current)
        if obj_id in visited:
            continue
        visited.add(obj_id)

        metadata = getattr(current, "metadata", None)
        if isinstance(metadata, dict):
            fps = metadata.get("render_fps")
            try:
                if fps is not None and float(fps) > 0:
                    return max(1, int(round(float(fps))))
            except (TypeError, ValueError):
                logger.debug("Ignoring non-numeric render_fps %r", fps)

        for attr_name in ("base_env", "env", "_env", "unwrapped"):
            try:
                next_obj = getattr(current, attr_name, None)
            except (AttributeError, RuntimeError, KeyError) as exc:
                logger.debug("Cannot read %s from %r: %s", attr_name, current, exc)
                continue
            if next_obj is None:
                continue
            if isinstance(next_obj, (list, tuple)):
                stack.extend(next_obj)
            else:
                stack.append(next_obj)
    return max(1, int(default_fps))


def _imitation_env_with_cfg(env: object) -> object:
    """Walk the wrapper chain to the env whose cfg carries the macro anchor mode."""
    current = env
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        cfg = getattr(current, "cfg", None)
        if cfg is not None and hasattr(cfg, "expert_macro_anchor_mode"):
            return current
        current = next(
            (
                candidate
                for candidate in (
                    getattr(current, attr, None)
                    for attr in ("unwrapped", "base_env", "env", "_env")
                )
                if candidate is not None and candidate is not current
            ),
            None,
        )
    raise TypeError("no wrapped environment exposes cfg.expert_macro_anchor_mode")


def _enable_wandb_video_sync(agent: object, *, video_folder: str, base_dir: str):
    """Enable WandB video sync and return a callable that logs newly completed videos."""
    logger_obj = getattr(agent, "logger", None)
    wandb_run = getattr(logger_obj, "experiment", None) if logger_obj else None
    if (
        wandb_run is None
        or not hasattr(wandb_run, "save")
        or not hasattr(wandb_run, "log")
    ):
        print("[INFO] WandB run not available; videos will remain local only.")
        return None

    video_pattern = os.path.join(video_folder, "*.mp4")
    video_step_pattern = re.compile(r"(?:step-|clip_)(\d+)")
    video_dir = Path(video_folder)
    last_uploaded_name: str | None = None
    # Tracks the highest WandB step seen through training metrics. Videos are
    # logged at this step so they cannot push WandB ahead of scalar metrics.
    max_step_seen = 0

    def _video_sort_key(path: Path) -> tuple[int, str]:
        match = video_step_pattern.search(path.stem)
        if match is not None:
            return int(match.group(1)), path.name
        return int(1e12), path.name

    if video_dir.exists():
        existing_videos = sorted(video_dir.glob("*.mp4"), key=_video_sort_key)
        if len(existing_videos) > 0:
            # Start from files created after the latest file seen at startup.
            last_uploaded_name = existing_videos[-1].name

    def _log_pending_videos(step_hint: int | None = None) -> None:
        nonlocal last_uploaded_name, max_step_seen

        if not video_dir.exists():
            return

        all_videos = sorted(video_dir.glob("*.mp4"), key=_video_sort_key)
        if len(all_videos) == 0:
            return

        if last_uploaded_name is None:
            new_videos = all_videos
        else:
            last_idx = next(
                (i for i, p in enumerate(all_videos) if p.name == last_uploaded_name),
                None,
            )
            if last_idx is None:
                new_videos = all_videos
            else:
                new_videos = all_videos[last_idx + 1 :]

        if len(new_videos) == 0:
            return

        if step_hint is not None:
            max_step_seen = max(max_step_seen, int(step_hint))

        wandb_step = getattr(wandb_run, "step", None)
        if wandb_step is not None:
            max_step_seen = max(max_step_seen, int(wandb_step))

        uploads_this_call = 0
        for video_path in new_videos:
            try:
                if video_path.stat().st_size <= 0:
                    continue
            except OSError:
                continue

            try:
                wandb_run.log(
                    {
                        "videos/train": wandb.Video(
                            str(video_path),
                            format="mp4",
                        )
                    },
                    step=max_step_seen,
                )
                uploads_this_call += 1
                last_uploaded_name = video_path.name
            except (OSError, ValueError, TypeError, wandb.Error) as exc:
                # If a file is still being finalized, retry on the next periodic call.
                logger.debug("Deferred W&B upload of %s: %s", video_path, exc)
                continue

            if uploads_this_call >= 1:
                # Bound upload overhead per metrics call.
                break

    try:
        # Keep local logging layout and stream only generated videos to WandB.
        wandb_run.save(video_pattern, base_path=base_dir, policy="live")
    except Exception as exc:
        print(f"[WARNING] Failed to enable WandB video sync: {exc}")
    return _log_pending_videos


def _collect_fastpath_state() -> dict[str, object]:
    """Resolved opt-in collection fast-path switches, env and ImitationLab side."""
    from imitationlab.envs.isaaclab.envs.collect_fastpath import describe as env_state
    from imitationlab.algorithms.common.collect_fastpath import (
        describe as learner_state,
    )

    return {"imitationlab.envs.isaaclab": env_state(), "algorithms": learner_state()}


def _resolve_checkpoint_path(checkpoint: str) -> str:
    """Resolve ``--checkpoint`` to a file, accepting a directory.

    A multi-segment cluster run cannot name its predecessor's checkpoint at
    submission time: the file name carries the step it was written at, and the
    logger nests it under a per-run ``<timestamp>_wandb-<id>`` directory that
    does not exist yet. Passing the checkpoint *tree* instead resolves it at
    launch, so every segment of a chain can be frozen with the same argument.

    Selection is by modification time, NOT by step. Each segment restarts its
    own step counter at zero, so after segment 2 writes ``model_step_5e8.pt``
    the highest-numbered file in the tree is still segment 1's
    ``model_step_25e8.pt`` -- resuming by step would walk the chain backwards.
    Checkpoints are staged through ``$TMPDIR`` and moved into place, so a
    visible file is always complete.

    A tree with no checkpoint is an error: an ``afterok`` predecessor that
    produced none did not do its job, and silently restarting from scratch
    would burn a whole segment.
    """
    path = os.path.abspath(checkpoint)
    if not os.path.isdir(path):
        return path
    found: list[tuple[float, int, str]] = []
    for root, _dirs, names in os.walk(path):
        for name in names:
            match = re.fullmatch(r"model_step_(\d+)\.pt", name)
            if match:
                full = os.path.join(root, name)
                found.append((os.path.getmtime(full), int(match.group(1)), full))
    if not found:
        msg = (
            f"--checkpoint points at a tree with no model_step_<N>.pt file: "
            f"{path}. A resumed segment must find its predecessor's checkpoint."
        )
        raise FileNotFoundError(msg)
    _mtime, step, resolved = max(found)
    print(
        f"[INFO] Resuming from the newest of {len(found)} checkpoints under "
        f"{path} (segment-local step {step}): {resolved}"
    )
    return resolved


def _load_training_checkpoint(agent, checkpoint_path: str) -> None:
    """Restore a full checkpoint or a model-only training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "actor_critic" in checkpoint:
        agent.load(checkpoint_path)
    else:
        agent.load_model(checkpoint_path)


def train(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: AlgorithmConfig,
    args_cli: argparse.Namespace,
) -> None:
    """Train an ImitationLab agent inside an already-selected simulation lifecycle."""
    # The environment's declared command interface is the authority on every
    # command input key and on whether the actor consumes a latent. Binding the
    # agent to it makes the historical env/agent mismatch impossible rather than
    # validated after the fact; `sync_input_keys` runs as part of the binding.
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
    if agent_cfg.trainer is None:
        agent_cfg.trainer = TrainerConfig()
    if args_cli.log_interval is not None:
        agent_cfg.trainer.log_interval = max(1, int(args_cli.log_interval))
    if args_cli.profile_iterations:
        agent_cfg.trainer.profile_iterations = True
    max_train_seconds = getattr(args_cli, "max_train_seconds", None)
    if max_train_seconds is not None:
        if float(max_train_seconds) <= 0.0:
            raise ValueError(
                f"--max-train-seconds must be > 0, got {max_train_seconds}."
            )
        agent_cfg.trainer.max_train_seconds = float(max_train_seconds)
    agent_cfg.collector.frames_per_batch *= env_cfg.scene.num_envs
    # Keep the on-policy rollout buffer and minibatching consistent when num_envs
    # or the per-env horizon (collector.frames_per_batch) differ from the config
    # defaults. PPO uses a single-rollout buffer (buffer == one
    # collected batch), but the config sizes replay_buffer.size / mini_batch_size
    # with a literal 4096-env assumption. Rescale the buffer to the actual batch
    # and keep the configured minibatch SIZE (so per-gradient-step memory is
    # constant; the number of minibatches grows with the batch). The default
    # 4096-env / horizon-24 configuration is unchanged by this.
    if args_cli.algorithm == "PPO":
        scaled_frames_per_batch = int(agent_cfg.collector.frames_per_batch)
        replay_buffer_cfg = getattr(agent_cfg, "replay_buffer", None)
        if replay_buffer_cfg is not None and getattr(replay_buffer_cfg, "size", 0):
            if int(replay_buffer_cfg.size) != scaled_frames_per_batch:
                logger.warning(
                    "Rescaling replay_buffer.size %d -> %d to match the on-policy "
                    "rollout batch (num_envs=%d x horizon).",
                    int(replay_buffer_cfg.size),
                    scaled_frames_per_batch,
                    int(env_cfg.scene.num_envs),
                )
            replay_buffer_cfg.size = scaled_frames_per_batch
        loss_cfg = getattr(agent_cfg, "loss", None)
        if loss_cfg is not None and getattr(loss_cfg, "mini_batch_size", 0):
            loss_cfg.mini_batch_size = min(
                int(loss_cfg.mini_batch_size), scaled_frames_per_batch
            )
    # max_iterations is expressed in rollout iterations, so override total_frames
    # after scaling frames_per_batch to the actual number of simulated envs.
    if args_cli.max_iterations is not None:
        agent_cfg.collector.total_frames = (
            args_cli.max_iterations * agent_cfg.collector.frames_per_batch
        )
    # TorchRL collectors warn and over-collect when total_frames is not divisible by
    # frames_per_batch. Align to an exact number of rollout batches.
    frames_per_batch = int(agent_cfg.collector.frames_per_batch)
    total_frames = int(agent_cfg.collector.total_frames)
    if frames_per_batch > 0:
        aligned_total_frames = max(
            frames_per_batch,
            (total_frames // frames_per_batch) * frames_per_batch,
        )
        if aligned_total_frames != total_frames:
            logger.warning(
                "Adjusting collector.total_frames from %d to %d to match frames_per_batch=%d.",
                total_frames,
                aligned_total_frames,
                frames_per_batch,
            )
            agent_cfg.collector.total_frames = aligned_total_frames
    if not agent_cfg.logger.log_dir:
        learner = args_cli.algorithm.lower()
        if args_cli.algorithm == "WMBPTT":
            if agent_cfg.wm.dynamics_model == "diffsr":
                learner = "diffsr"
            elif agent_cfg.wm.actor_source == "real":
                learner = "svg"
            else:
                learner = "supermimic"
        agent_cfg.logger.log_dir = str(
            Path("outputs/training") / learner / args_cli.task
        )
    distributed_metadata = None
    if getattr(args_cli, "distributed", False):
        import torch.distributed as dist
        from imitationlab.runners.distributed.runtime import shard_config

        distributed_metadata = shard_config(
            args_cli,
            env_cfg,
            agent_cfg,
            world=dist.get_world_size(),
            rank=dist.get_rank(),
        )
    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )
    # Applied before `dump_yaml` below so `params/env.yaml` records the
    # protocol the run actually used, not the task id's default.

    # directory for logging into
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger_backend = str(getattr(agent_cfg.logger, "backend", "") or "").lower()
    if logger_backend.rsplit(".", maxsplit=1)[-1] == "wandb":
        # Allocate the W&B ID before constructing the local run directory and
        # export it so TorchRL/W&B reuse the exact same ID during logger init.
        # A project identifies a collection of runs and is not unique; the run
        # ID is the collision-proof component needed when concurrent jobs begin
        # within the same second.
        wandb_run_id = os.environ.get("WANDB_RUN_ID") or wandb.util.generate_id()
        os.environ.setdefault("WANDB_RUN_ID", wandb_run_id)
        run_suffix = f"wandb-{wandb_run_id}"
    else:
        # Preserve the same uniqueness guarantee for non-W&B and offline jobs.
        scheduler_job_id = os.environ.get("SLURM_JOB_ID")
        run_suffix = (
            f"slurm-{scheduler_job_id}"
            if scheduler_job_id
            else f"run-{uuid.uuid4().hex[:8]}"
        )
    run_info = f"{run_timestamp}_{run_suffix}"
    # Explicit roots can point to a persistent bind mount for cluster runs.
    log_root_path = str(Path(agent_cfg.logger.log_dir).expanduser().resolve())
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence,
    # do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {run_info}")
    log_dir = os.path.join(log_root_path, run_info)
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    collect_fastpath = _collect_fastpath_state()
    print(f"[INFO] Collection fast paths: {collect_fastpath}")
    dump_yaml(
        os.path.join(log_dir, "params", "collect_fastpath.yaml"), collect_fastpath
    )
    agent_cfg.logger.log_dir = log_dir
    if distributed_metadata is not None:
        dump_yaml(
            os.path.join(log_dir, "params", "distributed.yaml"), distributed_metadata
        )
    # log command used to run the script
    command = " ".join(sys.orig_argv)
    (Path(log_dir) / "command.txt").write_text(command)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # The env records video itself from ``env_cfg.video_recorders``; configure it
    # before the env exists.
    video_capture = None
    if args_cli.video:
        camera_kwargs = {}
        if args_cli.video_width is not None:
            camera_kwargs["window_width"] = int(args_cli.video_width)
        if args_cli.video_height is not None:
            camera_kwargs["window_height"] = int(args_cli.video_height)
        video_capture = attach_video_capture(
            env_cfg,
            output_dir=os.path.join(log_dir, "videos", "train"),
            video_length=args_cli.video_length,
            video_interval=args_cli.video_interval,
            **camera_kwargs,
        )
        print(f"[INFO] Recording videos during training to: {video_capture.directory}")

    # WMBPTT's process-wide compile options come first: env helpers compile.
    if args_cli.algorithm == "WMBPTT":
        apply_compile_options(agent_cfg.wm)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # Construction resolves runtime bindings (including frozen encoder hashes).
    # Keep the saved environment contract equal to the one actually initialized.
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env.unwrapped.cfg)
    isaac_env = env.unwrapped

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        raise NotImplementedError(
            "DirectMARLEnv is not supported for ImitationLab training yet."
        )
    start_time = time.time()

    env = IsaacLabWrapper(env)  # type: ignore
    env = attach_terminal_obs_reader(env)
    env = TransformedEnv(
        base_env=env,
        transform=Compose(
            RewardSum(),  # type: ignore
            StepCounter(500),  # type: ignore
            # RewardClipping(-10.0, 5.0),  # type: ignore
        ),
    )

    agent_class = ALGORITHM_CLASS_MAP[args_cli.algorithm]
    if args_cli.algorithm == "WMBPTT":
        from imitationlab.algorithms import DiffSR, SVG, SuperMimic

        agent_class = (
            DiffSR
            if agent_cfg.wm.dynamics_model == "diffsr"
            else SVG
            if agent_cfg.wm.actor_source == "real"
            else SuperMimic
        )
    agent_kwargs = {}
    if getattr(args_cli, "distributed", False):
        if args_cli.algorithm != "PPO":
            raise ValueError("Distributed training requires PPO.")
        from imitationlab.algorithms.ppo.distributed import DistributedPPO

        agent_class = DistributedPPO
        agent_kwargs["max_training_seconds"] = args_cli.max_training_seconds
    agent = agent_class(
        env=env,
        config=agent_cfg,  # type: ignore
        **agent_kwargs,
    )
    from imitationlab.runners.artifacts import training_run_spec

    agent._run_metadata = training_run_spec(
        args_cli, env_cfg, agent_cfg, collect_fastpath
    )
    agent._run_metadata["learner_class"] = (
        f"{type(agent).__module__}:{type(agent).__name__}"
    )
    Path(log_dir, "run.json").write_text(
        json.dumps(agent._run_metadata, indent=2) + "\n"
    )
    install_reset_stats_persistence(agent, isaac_env)
    if wandb.run is not None:
        # ImitationLab records its own switch at W&B init; add the env side too.
        wandb.run.config.update(
            {"collect_fastpath": collect_fastpath}, allow_val_change=True
        )
    if getattr(args_cli, "distributed", False):
        # Every rank must leave the loop at the same iteration boundary: a
        # signal only raises a flag, which DistributedPPO pools across ranks.
        def request_distributed_stop(_signum, _frame):
            agent.stop_requested = True

        signal.signal(signal.SIGTERM, request_distributed_stop)
        signal.signal(signal.SIGINT, request_distributed_stop)
    budget = getattr(agent_cfg.trainer, "max_train_seconds", None)
    if budget is not None and float(budget) > 0.0:
        if not getattr(agent, "supports_train_time_budget", False):
            raise ValueError(
                f"{type(agent).__name__}.train does not honour "
                "trainer.max_train_seconds (--max-train-seconds). Use PPO for wall-clock-limited training."
            )
        print(
            f"[INFO] Training wall-clock budget: {float(budget):.0f} s "
            "(starts after the first collected batch)."
        )

    video_media_logger = None
    if video_capture is not None:
        video_media_logger = _enable_wandb_video_sync(
            agent,
            video_folder=str(video_capture.directory),
            base_dir=log_dir,
        )
        if video_media_logger is not None:
            original_log_metrics = agent.log_metrics

            def _log_metrics_with_video(*args, **kwargs):
                step = kwargs.get("step")
                try:
                    step_hint = int(step) if step is not None else None
                except (TypeError, ValueError):
                    step_hint = None
                result = original_log_metrics(*args, **kwargs)
                video_media_logger(step_hint)
                return result

            agent.log_metrics = _log_metrics_with_video

    if args_cli.checkpoint is not None:
        if getattr(args_cli, "distributed", False):
            # Rank zero resolves the newest checkpoint; every rank loads that one.
            checkpoint_paths = [
                _resolve_checkpoint_path(args_cli.checkpoint)
                if dist.get_rank() == 0
                else None
            ]
            dist.broadcast_object_list(checkpoint_paths, src=0)
            checkpoint_path = checkpoint_paths[0]
        else:
            checkpoint_path = _resolve_checkpoint_path(args_cli.checkpoint)
        print(f"[INFO] Loading checkpoint: {checkpoint_path}")
        _load_training_checkpoint(agent, checkpoint_path)

    # run training
    interrupted = False
    try:
        agent.train()
        if getattr(agent, "stop_reason", None) == "signal":
            # Distributed ranks stop on a pooled flag instead of raising
            # KeyboardInterrupt; the run has not reached its budget either way.
            interrupted = True
    except KeyboardInterrupt:
        interrupted = True
        print("\n[INFO] Training interrupted by user.")
    finally:
        if video_media_logger is not None:
            video_media_logger(None)
        env.close()
        if video_capture is not None:
            # ``env.close()`` writes the last clip. Print every retained path.
            video_capture.report(require_clip=False)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")
    if interrupted:
        # An interrupted run has NOT reached its frame budget, so it must not
        # look like a success. Exiting 0 here satisfies an `afterok` dependency
        # and lets a chained segment resume from a truncated predecessor: on
        # 2026-08-15 a SIGINT stopped group_vq64_hold10 at 220M of 2.5B, Slurm
        # recorded COMPLETED, and the next segment launched. That one was
        # caught only because no checkpoint exists before save_interval; an
        # interrupt at 1.2B would have resumed from the 1B checkpoint and
        # quietly trained 8.8B of a 10B budget.
        if wandb.run is not None:
            wandb.finish(exit_code=1)
        msg = (
            "Training was interrupted before reaching its frame budget; "
            "failing so dependent stages do not run."
        )
        raise RuntimeError(msg)
    success_marker = os.environ.get("ISAACLAB_WORKLOAD_SUCCESS_MARKER")
    if success_marker:
        Path(success_marker).touch()
        print("[INFO] ImitationLab training workload success marker written.")

    if wandb.run is not None:
        wandb.finish(exit_code=0)
