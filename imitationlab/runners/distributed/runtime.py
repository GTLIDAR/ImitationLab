"""Single-node torchrun integration; CLI batch sizes remain global."""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path


def require_smoke_success(root, workers):
    """Gate a queued production stage on all workers' completed smoke evidence."""
    import math

    from imitationlab.runners.distributed.qualification import read_health

    rows = read_health(Path(root), workers)
    if any(
        row["cumulative_env_frames"] <= row["initial_frames"]
        or row["optimizer_updates_this_segment"] <= 0
        or row["stop_reason"] != "time_budget"
        or not all(math.isfinite(lr) and lr > 0 for lr in row["learning_rates"])
        for row in rows
    ):
        raise RuntimeError(
            "Distributed smoke did not complete finite synchronized training"
        )
    print(f"[DISTRIBUTED] verified {workers}-worker smoke at {root}", flush=True)


def isolate_worker_temporary_directory(rank):
    """Prevent concurrent USD downloads from overwriting memory-mapped assets."""
    import tempfile

    root = (
        Path(tempfile.gettempdir())
        / f"rlopt-job-{os.environ.get('SLURM_JOB_ID', os.getppid())}"
        / f"rank-{rank}"
    )
    root.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(root)
    # tempfile caches its first lookup, often before runtime configuration.
    tempfile.tempdir = str(root)
    print(f"[DISTRIBUTED] rank={rank} temporary_directory={root}", flush=True)
    return root


def configure_runtime(args, env_cfg, agent_cfg):
    """Select devices before simulator startup, then initialize NCCL."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if not getattr(args, "distributed", False):
        if world > 1:
            raise ValueError(
                "torchrun requires --distributed; refusing independent trainers"
            )
        return
    if args.algorithm != "PPO":
        raise ValueError("--distributed requires PPO")
    if world < 2:
        raise ValueError("--distributed requires torchrun with at least two workers")
    if int(os.environ.get("LOCAL_WORLD_SIZE", world)) != world:
        raise ValueError("Only single-node distributed training is qualified")
    smoke = getattr(args, "require_distributed_smoke", None)
    if smoke:
        require_smoke_success(smoke, world)
    if args.max_training_seconds is not None and args.max_training_seconds <= 0:
        raise ValueError("--max_training_seconds must be positive")
    if args.video:
        raise ValueError("Distributed qualification does not support training video")
    if args.seed == -1:
        raise ValueError("Distributed training requires an explicit reproducible seed")
    import faulthandler

    # Preserve Python stacks for native simulator crashes (e.g. SIGBUS).
    faulthandler.enable(all_threads=True)
    import torch
    import torch.distributed as dist

    rank = int(os.environ["LOCAL_RANK"])
    backend = str(getattr(args, "distributed_backend", None) or "nccl")
    if backend not in ("nccl", "gloo"):
        raise ValueError(f"--distributed_backend must be nccl or gloo, got {backend!r}")
    isolate_worker_temporary_directory(rank)
    print(
        f"[DISTRIBUTED] before CUDA init: rank={rank} backend={backend} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"CUDA_DEVICE_ORDER={os.environ.get('CUDA_DEVICE_ORDER')}",
        flush=True,
    )
    device_index = worker_device_index(rank, backend, torch.cuda.device_count)
    torch.cuda.set_device(device_index)
    device = f"cuda:{device_index}"
    args.device = env_cfg.sim.device = agent_cfg.device = device
    agent_cfg.env.device = device
    if hasattr(env_cfg, "distributed_reference_sampling"):
        env_cfg.distributed_reference_sampling = True
    data = getattr(env_cfg, "data", None)
    for name in ("macro_cache_device", "runtime_cache_device"):
        value = getattr(data, name, None)
        if isinstance(value, str) and value.startswith("cuda"):
            setattr(data, name, device)
    for key in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "WARP_CACHE_PATH"):
        root = os.environ.get(key, str(Path("/tmp") / f"rlopt-{key.lower()}"))
        os.environ[key] = str(
            Path(root)
            / f"job-{os.environ.get('SLURM_JOB_ID', os.getppid())}"
            / f"rank-{rank}"
        )
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
    # Newton uses implicit Warp devices for capture and temporary arrays.
    # Task imports can initialize Warp on cuda:0 before Torch selects this rank.
    import warp as wp

    wp.set_device(device)
    print(f"[DISTRIBUTED] rank={rank} warp_device={wp.get_device()}", flush=True)
    if backend == "nccl":
        dist.init_process_group(
            "nccl", timeout=timedelta(minutes=10), device_id=torch.device(device)
        )
    else:
        dist.init_process_group("gloo", timeout=timedelta(minutes=10))
    print(f"[DISTRIBUTED] rank={rank}/{world} device={device}", flush=True)


def worker_device_index(rank, backend, device_count):
    """The CUDA device of a worker: its local rank, one GPU per worker.

    NCCL refuses two ranks on one GPU, so only the Gloo backend (local
    functional tests) may share devices: a rank beyond the visible GPUs runs
    on cuda:0, which is also the device Isaac Lab's launcher
    (``_resolve_distributed_device``) gives its simulation. ``device_count``
    is called only on that path.
    """
    if backend == "nccl":
        return int(rank)
    count = int(device_count())
    if count < 1:
        raise RuntimeError("Distributed workers need at least one visible CUDA device")
    return int(rank) if int(rank) < count else 0


def shard_config(args, env_cfg, agent_cfg, *, world, rank):
    """Called after CLI scaling: partition global batches and preserve budget."""
    fields = {
        "num_envs": int(env_cfg.scene.num_envs),
        "frames_per_batch": int(agent_cfg.collector.frames_per_batch),
        "mini_batch_size": int(agent_cfg.loss.mini_batch_size),
    }
    for name, value in fields.items():
        if value < world or value % world:
            raise ValueError(
                f"Global {name}={value} must be divisible by world_size={world}"
            )
    if fields["frames_per_batch"] % fields["mini_batch_size"]:
        raise ValueError("Global rollout must contain complete minibatches")
    env_cfg.scene.num_envs = agent_cfg.env.num_envs = fields["num_envs"] // world
    # Env-side frame schedules (termination anneal, ratio ramp, graph capture
    # gate) count global frames, like the trainer's budget.
    if hasattr(env_cfg, "data_parallel_world_size"):
        env_cfg.data_parallel_world_size = int(world)
    agent_cfg.collector.frames_per_batch = fields["frames_per_batch"] // world
    agent_cfg.replay_buffer.size = agent_cfg.collector.frames_per_batch
    agent_cfg.loss.mini_batch_size = fields["mini_batch_size"] // world
    # Model tensors are broadcast from rank zero after initialization; random
    # rollout and minibatch streams must differ across workers. The env seed
    # (train_impl: env_cfg.seed = agent_cfg.seed) keys each rank's reset
    # generator and the graph step's counter-based reset randomness, so ranks
    # draw distinct streams. Rank r of seed s shares its streams with rank 0
    # of seed s + r: give repeat seeds of one arm a stride >= world_size.
    agent_cfg.seed += rank
    if rank:
        agent_cfg.logger.backend = ""
        agent_cfg.logger.log_dir = str(
            Path(agent_cfg.logger.log_dir) / f"worker-{rank}"
        )
    return {
        **fields,
        "world_size": world,
        "rank": rank,
        "total_frames": int(agent_cfg.collector.total_frames),
    }
