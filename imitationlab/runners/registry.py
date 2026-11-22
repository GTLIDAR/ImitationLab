"""Task identifiers and agent-config lookup shared by every entry point.

Importing this module does not import the simulator. The gym registry is read
only by :func:`resolve_agent_cfg_entry_point`, after the caller has registered
the tasks.
"""

from __future__ import annotations

from imitationlab.runners.errors import UserError

DEFAULT_TASK = "imitation-g1-v1"

#: Command interfaces. ``spectral`` is the continuous spectral skill latent;
#: ``fsq`` is the same spectral skill objective with a quantized latent.
INTERFACES = ("explicit", "fsq", "spectral")
LATENT_INTERFACES = INTERFACES[1:]


#: The registry kwarg that names an algorithm-independent agent config. A run
#: that leaves ``--agent`` at this value gets the algorithm-specific key below.
GENERIC_AGENT_ENTRY_POINT = "agent_cfg_entry_point"

#: Gym-registry kwarg keys of the algorithm-specific agent configs. A run
#: record stores the key it used (``--agent``).
ENTRY_POINT_ALGORITHM_MAP = {
    "ppo_cfg_entry_point": "PPO",
    "wmbptt_cfg_entry_point": "WMBPTT",
}


def is_eager_task(task: str) -> bool:
    """Return whether ``task`` is the public task, whose evaluation forces the eager step.

    ``imitationlab evaluate`` applies neutral reset-sampling settings and the
    eager control step to it, because pinned resets require ``step_mode=eager``.
    """
    return task.split(":")[-1] == DEFAULT_TASK


def agent_entry_point_key(algorithm: str) -> str:
    """The registry kwarg key of the agent config for ``algorithm``."""
    return f"{algorithm.lower()}_cfg_entry_point"


def resolve_agent_cfg_entry_point(
    task_name: str | None,
    algorithm: str,
    agent_entry_point: str | None = GENERIC_AGENT_ENTRY_POINT,
) -> str:
    """Return the registry kwarg key of the agent config to load.

    An explicit ``agent_entry_point`` other than the generic key wins. Otherwise
    the task must register the algorithm-specific key.
    """
    if agent_entry_point and agent_entry_point != GENERIC_AGENT_ENTRY_POINT:
        return agent_entry_point
    key = agent_entry_point_key(algorithm)
    if task_name is None:
        return key
    import gymnasium as gym

    task_id = task_name.split(":")[-1]
    try:
        spec = gym.spec(task_id)
    except gym.error.Error as exc:
        raise UserError(f"Task {task_id!r} is not registered.") from exc
    if spec.kwargs.get(key) is not None:
        print(f"[INFO] Using agent config entry point: {key}")
        return key
    supported = sorted(
        name
        for entry, name in ENTRY_POINT_ALGORITHM_MAP.items()
        if spec.kwargs.get(entry) is not None
    )
    raise UserError(
        f"Task {task_id!r} has no agent config for algorithm {algorithm!r}. "
        f"Algorithms with an agent config for this task: {supported}."
    )
