"""Environment frames as the frame-based schedules count them."""

from __future__ import annotations

from typing import Any


def schedule_frames(env: Any) -> int:
    """Control steps x environments, summed over the data-parallel ranks.

    The trainer's frame budget is global, so the env-side schedules expressed
    in frames (the termination-threshold anneal, the SONIC uniform-ratio ramp,
    the graph step's capture gate) count global frames too: under data-parallel
    training each rank steps ``num_envs`` of the global environments and
    ``cfg.data_parallel_world_size`` (set by the distributed entrypoint, 1
    otherwise) scales its local count. With equal shards every rank computes
    the same value, so the schedules stay in step across ranks.
    """
    world = int(getattr(getattr(env, "cfg", None), "data_parallel_world_size", 1) or 1)
    return int(env.common_step_counter) * int(env.num_envs) * world


__all__ = ["schedule_frames"]
