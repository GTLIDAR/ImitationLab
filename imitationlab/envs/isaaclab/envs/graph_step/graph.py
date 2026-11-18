"""CUDA-graph replay of the masked control step.

:class:`GraphStepper` runs :class:`.masked_step.MaskedStepper`'s segments
through three CUDA graphs per control step::

    G_pre    (main stream)  cursor advance, action processing
    G_gather (side stream)  next-frame rows from the host-mapped store
    physics                 Newton's own graph (``env.sim.step``), eager call
    G_post   (main stream)  scene update, terminations, rewards, masked
                            reset, commands, pushes, next frame, observations

The host issues a handful of launches per step, so the step no longer waits on
Python. Between the graphs it only orders streams (side stream after G_pre,
main stream after G_gather) and raises Newton's FK/solver-reset flag.

The graphs record the masked step's own kernels, so replay is bit-identical to
the eager masked step; ``tests/test_graph_step.py`` checks it. Capture waits
until state that the graphs would freeze is constant:

- a curriculum writes Python values into term parameters, so capture never
  starts while one is configured (the task has none);
- the first ``warmup_steps`` eager steps compile and allocate lazily created
  kernels and buffers outside capture.

Everything the next step reads from this step's outputs lives in persistent
buffers written in place (observations, terminal observations, the reference
frame, staging rows); capture-time Python side effects (counters, lazy-buffer
timestamps) are replayed only as kernels.
"""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING, Any

import torch
import warp as wp

from imitationlab.envs.isaaclab.envs.collect_fastpath import ENV_VAR, fastpath_enabled
from imitationlab.envs.isaaclab.envs.graph_step.masked_step import (
    MaskedStepper,
    UnsupportedMaskedStep,
)

logger = logging.getLogger(__name__)

#: Opt-in fast paths the capture cannot run without. With them off the captured
#: segment would call ``nonzero`` (the command timer) and copy from pageable
#: memory (the reward offsets), which CUDA forbids during capture, and the step
#: would have no device-addressable reference store.
CAPTURE_FAST_PATHS = ("device_gather", "never_resample", "reward_offsets")

if TYPE_CHECKING:
    from imitationlab.envs.isaaclab.envs.imitation_rl_env import ImitationRLEnv


def _nccl_group_initialized() -> bool:
    """Whether this process belongs to an initialized NCCL process group."""
    import torch.distributed as dist

    return bool(
        dist.is_available() and dist.is_initialized() and dist.get_backend() == "nccl"
    )


@contextlib.contextmanager
def lazy_properties_allowed(*objects: Any):
    """Lift Isaac Lab's ``capture_unsafe`` guard on these objects' classes.

    The guard refuses lazily recomputed properties (``projected_gravity_b``
    and friends) during capture because their recompute is gated by a Python
    timestamp that replay cannot see. In this graph step every segment that
    reads them starts with ``scene.update``, which advances that timestamp, so
    each read in the captured segment records its recompute and replays it
    every step. Inside the context the guarded properties call their
    originals; they are restored on exit.
    """
    from isaaclab.utils.warp.utils import capture_unsafe

    guard_code = capture_unsafe()(lambda: None).__code__
    patched: list[tuple[type, str, property]] = []
    try:
        for obj in objects:
            for cls in type(obj).__mro__:
                for name, attr in list(vars(cls).items()):
                    fget = getattr(attr, "fget", None)
                    if (
                        isinstance(attr, property)
                        and fget is not None
                        and getattr(fget, "__code__", None) is guard_code
                    ):
                        patched.append((cls, name, attr))
                        setattr(
                            cls,
                            name,
                            property(
                                fget.__wrapped__, attr.fset, attr.fdel, attr.__doc__
                            ),
                        )
        yield
    finally:
        for cls, name, attr in reversed(patched):
            setattr(cls, name, attr)


def _clone_structure(obs: Any) -> Any:
    if isinstance(obs, dict):
        return {key: _clone_structure(value) for key, value in obs.items()}
    return obs.clone()


def _copy_structure(dst: Any, src: Any) -> None:
    if isinstance(dst, dict):
        for key, value in dst.items():
            _copy_structure(value, src[key])
        return
    dst.copy_(src)


def _groups(env: Any, obs: dict) -> dict:
    """The observation manager's own groups of ``obs`` (wrappers may add keys)."""
    names = list(env.observation_manager.active_terms.keys())
    return {name: obs[name] for name in names}


class GraphStepper(MaskedStepper):
    """:class:`MaskedStepper` replayed as CUDA graphs once state is static."""

    def __init__(self, env: ImitationRLEnv, *, warmup_steps: int = 3) -> None:
        super().__init__(env)
        self.warmup_steps = int(warmup_steps)
        self._eager_steps = 0
        self.static_action: torch.Tensor | None = None
        self.static_obs: Any = None
        self.static_terminal_obs: Any = None
        self.graphs: dict[str, torch.cuda.CUDAGraph] | None = None
        self.pool = None
        self.captured_at_step: int | None = None

    @staticmethod
    def validate(env: ImitationRLEnv) -> None:
        """:meth:`MaskedStepper.validate` plus the fast paths capture needs."""
        missing = [name for name in CAPTURE_FAST_PATHS if not fastpath_enabled(name)]
        if missing:
            raise UnsupportedMaskedStep(
                f"The graph step needs the collection fast paths {missing}: set "
                f"{ENV_VAR}=1 before the process starts."
            )
        MaskedStepper.validate(env)

    # ------------------------------------------------------------------
    def _store_obs(self, obs: Any) -> None:
        if self.static_obs is None:
            self.env.obs_buf = obs
            return
        _copy_structure(self.static_obs, _groups(self.env, obs))

    def _snapshot_terminal_obs(self) -> None:
        if self.static_terminal_obs is None:
            super()._snapshot_terminal_obs()
            return
        _copy_structure(self.static_terminal_obs, _groups(self.env, self.env.obs_buf))

    def curriculum_done(self) -> bool:
        """True when no curriculum term can write a parameter the graphs would freeze.

        ``imitation-g1-v1`` has no curriculum. A curriculum term writes Python
        values into term parameters, so any term holds capture back.
        """
        # CurriculumManager has no get_term_cfg; its terms are parallel lists.
        return not self.env.curriculum_manager._term_cfgs

    def capture_ready(self) -> bool:
        return self._eager_steps >= self.warmup_steps and self.curriculum_done()

    # ------------------------------------------------------------------
    def step(self, action: torch.Tensor):
        env = self.env
        self.ensure_frame()
        previous_obs = env.obs_buf
        if self.graphs is None and self.capture_ready():
            self._capture(action)
        if self.graphs is None:
            self._eager_steps += 1
            return super().step(action)
        self.static_action.copy_(action)
        self._replay()
        # The eager step publishes the previous step's observation dict, which
        # TorchRL has since extended in place (reward, done, ...): keep those
        # keys, with the graph's pre-reset group buffers.
        self._terminal_obs = {
            **{k: v for k, v in previous_obs.items() if k not in self.static_obs},
            **self.static_terminal_obs,
        }
        # Fresh tensors in a fresh dict each step, as the eager compute
        # returns: TorchRL adds keys to the dict it receives and the wrapper
        # keeps references to the tensors across steps, while replay
        # overwrites the static buffers in place.
        env.obs_buf = _clone_structure(self.static_obs)
        self._publish_terminal_obs()
        env.extras["log"] = self.log
        return (
            env.obs_buf,
            env.reward_buf,
            env.reset_terminated,
            env.reset_time_outs,
            env.extras,
        )

    def replay_static(self, action: torch.Tensor) -> None:
        """One control step for the static collector: replay only.

        No observation clone and no indexed terminal publish; the collector
        reads :attr:`static_obs`, :attr:`static_terminal_obs`,
        :attr:`reset_mask` and the env's reward / termination buffers.
        """
        self.ensure_frame()
        self.static_action.copy_(action)
        self._replay()
        self.env.extras["log"] = self.log

    def _replay(self) -> None:
        graphs = self.graphs
        assert graphs is not None
        main = torch.cuda.current_stream(self.device)
        graphs["pre"].replay()
        side = self.gather_stream
        side.wait_stream(main)
        with torch.cuda.stream(side):
            graphs["gather"].replay()
            self.gather_done.record(side)
        for k in range(self.substeps):
            self.physics(k)
            if k < self.substeps - 1:
                graphs["between"].replay()
        main.wait_event(self.gather_done)
        graphs["post"].replay()
        # The captured post-physics segment advanced this Python counter only
        # once, at capture; keep it counting control steps as the eager step
        # does (its readers are host-side: schedules, logs).
        self.env.common_step_counter += 1

    # ------------------------------------------------------------------
    def _capture(self, action: torch.Tensor) -> None:
        """Record the three graphs without executing them.

        Capture records kernels but runs the Python once, so Python-side
        counters advanced by the captured code are restored afterwards. The
        step that triggered the capture then runs by replay.
        """
        env = self.env
        device = self.device
        self.static_action = action.detach().clone()
        groups = _groups(env, env.obs_buf)
        self.static_obs = _clone_structure(groups)
        self.static_terminal_obs = _clone_structure(groups)
        env.obs_buf = dict(self.static_obs)
        python_counters = (env.common_step_counter, env._sim_step_counter)
        torch.cuda.synchronize(device)

        names = ("pre", "gather", "between", "post")
        graphs = {name: torch.cuda.CUDAGraph() for name in names}
        pool = torch.cuda.graph_pool_handle()
        capture_stream = torch.cuda.Stream(device=device)
        capture_stream.wait_stream(torch.cuda.current_stream(device))
        warp_stream = wp.stream_from_torch(capture_stream)
        segments = (
            ("pre", lambda: self.pre_physics(self.static_action)),
            ("gather", self.gather_segment),
            ("between", self.between_substeps),
            ("post", self.post_physics),
        )
        guarded = [self.robot.data, *env.scene.sensors.values()]
        guarded += [sensor.data for sensor in env.scene.sensors.values()]
        # Under data-parallel training the NCCL watchdog thread polls events of
        # finished collectives; a "global" capture would forbid that in every
        # thread. "thread_local" keeps the same prohibitions in this (the
        # capturing) thread, and the recorded kernels are the same.
        mode = "thread_local" if _nccl_group_initialized() else "global"
        for name, segment in segments:
            with (
                lazy_properties_allowed(*guarded),
                wp.ScopedStream(warp_stream),
                torch.cuda.graph(
                    graphs[name],
                    pool=pool,
                    stream=capture_stream,
                    capture_error_mode=mode,
                ),
            ):
                segment()
        torch.cuda.synchronize(device)
        env.common_step_counter, env._sim_step_counter = python_counters
        self.graphs = graphs
        self.pool = pool
        self.captured_at_step = int(env.common_step_counter)
        logger.warning(
            "Captured the graph step at control step %d (%d envs, %d physics "
            "sub-steps per control step).",
            self.captured_at_step,
            self.num_envs,
            self.substeps,
        )


__all__ = ["GraphStepper"]
