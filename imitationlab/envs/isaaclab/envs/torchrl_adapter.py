# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Mapping

import gymnasium as gym
import torch
from torchrl.data.tensor_specs import Composite, Unbounded
from torchrl.envs.transforms import VecGymEnvTransform
from torchrl.envs.libs.gym import (
    GymWrapper,
    _gym_to_torchrl_spec_transform,
    terminal_obs_reader,
)
from tensordict import TensorDict
from tensordict.base import TensorDictBase

from .collect_fastpath import fastpath_enabled

logger = logging.getLogger(__name__)

_FAST_SPEC_ZERO = fastpath_enabled("spec_zero")
_FAST_TERMINAL_SWAP = fastpath_enabled("terminal_swap")
_FAST_READ_DONE = fastpath_enabled("read_done")


class IsaacLabWrapper(GymWrapper):
    """A wrapper for IsaacLab environments.

    Args:
        env (isaaclab.envs.ManagerBasedRLEnv or equivalent): the environment instance to wrap.
        categorical_action_encoding (bool, optional): if ``True``, categorical
            specs will be converted to the TorchRL equivalent (:class:`torchrl.data.Categorical`),
            otherwise a one-hot encoding will be used (:class:`torchrl.data.OneHot`).
            Defaults to ``False``.
        allow_done_after_reset (bool, optional): if ``True``, it is tolerated
            for envs to be ``done`` just after :meth:`reset` is called.
            Defaults to ``False``.

    For other arguments, see the :class:`torchrl.envs.GymWrapper` documentation.

    Refer to `the Isaac Lab documentation <https://isaac-sim.github.io/IsaacLab/>`_.

    Example:
        >>> # This code block ensures that the Isaac app is started in headless mode
        >>> from scripts_isaaclab.app import AppLauncher
        >>> import argparse

        >>> parser = argparse.ArgumentParser(description="Train an RL agent with TorchRL.")
        >>> AppLauncher.add_app_launcher_args(parser)
        >>> args_cli, hydra_args = parser.parse_known_args(["--headless"])
        >>> app_launcher = AppLauncher(args_cli)

        >>> # Imports and env
        >>> import gymnasium as gym
        >>> import isaaclab_tasks  # noqa: F401
        >>> from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg
        >>> from torchrl.envs.libs.isaac_lab import IsaacLabWrapper

        >>> env = gym.make("Isaac-Ant-v0", cfg=AntEnvCfg())
        >>> env = IsaacLabWrapper(env)

    """

    def __init__(
        self,
        env: ManagerBasedRLEnv,  # noqa: F821
        *,
        categorical_action_encoding: bool = False,
        allow_done_after_reset: bool = True,
        convert_actions_to_numpy: bool = False,
        device: torch.device | None = None,
        **kwargs,
    ):
        if device is None:
            device = torch.device("cuda:0")
        super().__init__(
            env,
            device=device,
            categorical_action_encoding=categorical_action_encoding,
            allow_done_after_reset=allow_done_after_reset,
            convert_actions_to_numpy=convert_actions_to_numpy,
            **kwargs,
        )
        # Keep only the latest log payload to avoid retaining large per-step
        # info dicts (often CUDA tensors) across long rollouts.
        self.log_infos = deque(maxlen=1)

    def _base_isaac_env(self):
        env = getattr(self, "_env", None)
        return getattr(env, "unwrapped", env)

    def _reference_transition_metrics(self) -> dict[str, object]:
        """Read imitation health scalars without affecting other task types."""

        base_env = self._base_isaac_env()
        reference = getattr(base_env, "reference_command", None)
        reader = getattr(reference, "transition_metrics", None)
        if not callable(reader):
            return {}
        return dict(reader())

    def _log_payload(self, info: Mapping[str, object]) -> dict[str, object]:
        """Copy reset logs and add live scalar views without per-step CUDA sync."""

        payload = self._extract_log_info(dict(info))
        payload.update(self._reference_transition_metrics())
        return payload

    @property
    def required_log_entries(self) -> tuple[str, ...]:
        """The ``extras["log"]`` entries this env guarantees, for the trainer's check."""

        return tuple(getattr(self._base_isaac_env(), "required_log_entries", ()))

    def live_log_metrics(self) -> dict[str, object]:
        """Scalars for an agent that steps the env directly, past ``log_infos``."""

        return self._reference_transition_metrics()

    @property
    def imitation_interface(self):
        """The wrapped environment's imitation capability surface.

        The learner resolves this by walking the wrapper stack; forwarding it here
        keeps the wrapper free of per-capability delegators.
        """
        return self._base_isaac_env().imitation_interface

    @property
    def _is_batched(self) -> bool:
        return True

    def seed(self, seed: int | None):
        self._set_seed(seed)

    @staticmethod
    def _extract_log_info(
        info: dict[str, torch.Tensor | float | int | str | bool | None],
    ) -> dict[str, object]:
        """Detach log scalars without forcing a per-step device sync.

        ``log_infos`` keeps only the newest step (``deque(maxlen=1)``) and the
        trainer converts the surviving payload to Python floats once per
        iteration (``imitationlab.algorithms.common.utils.log_info`` calls
        ``.item()`` on tensors), so a per-step ``.cpu().item()`` here was one
        pipeline stall per scalar per step for values that were then
        discarded. Every entry in the
        ``log`` subtree is freshly allocated by its producer (manager reset
        means, transition-metric ratios), so retaining the detached reference
        until the iteration boundary is safe.
        """

        for key, value in info.items():
            if isinstance(value, torch.Tensor):
                info[key] = value.detach()
            elif isinstance(value, Mapping):
                info[key] = IsaacLabWrapper._extract_log_info(dict(value))
        return info

    def _build_env(
        self,
        env,
        from_pixels: bool = False,
        pixels_only: bool = False,
    ) -> gym.core.Env:  # noqa: F821
        env = super()._build_env(
            env,
            from_pixels=from_pixels,
            pixels_only=pixels_only,
        )
        env.autoreset_mode = "SameStep"
        return env

    def _make_specs(self, env: gym.Env, batch_size=None) -> None:  # noqa: F821
        # Build specs from IsaacLab's unbatched spaces to preserve observation keys.
        if batch_size is None:
            batch_size = self.batch_size
        env_unwrapped = getattr(env, "unwrapped", env)

        action_space = getattr(env_unwrapped, "single_action_space", None)
        action_needs_batch = action_space is not None
        action_space = action_space if action_space is not None else env.action_space
        action_spec = _gym_to_torchrl_spec_transform(
            action_space,
            device=self.device,
            categorical_action_encoding=self._categorical_action_encoding,
        )
        if action_needs_batch:
            action_spec = action_spec.expand(*batch_size, *action_spec.shape)  # type: ignore
        obs_space = getattr(env_unwrapped, "single_observation_space", None)
        obs_needs_batch = obs_space is not None
        obs_space = obs_space if obs_space is not None else env.observation_space
        observation_spec = _gym_to_torchrl_spec_transform(
            obs_space,
            device=self.device,
            categorical_action_encoding=self._categorical_action_encoding,
        )
        if obs_needs_batch:
            observation_spec = observation_spec.expand(
                *batch_size, *observation_spec.shape
            )  # type: ignore
        if not isinstance(observation_spec, Composite):
            if self.from_pixels:
                observation_spec = Composite(pixels=observation_spec, shape=batch_size)  # type: ignore
            else:
                observation_spec = Composite(
                    observation=observation_spec, shape=batch_size
                )  # type: ignore

        reward_space = self._reward_space(env)
        if reward_space is not None:
            reward_spec = _gym_to_torchrl_spec_transform(
                reward_space,
                device=self.device,
                categorical_action_encoding=self._categorical_action_encoding,
            )
        else:
            reward_spec = Unbounded(shape=[1], device=self.device).expand(
                *batch_size, 1
            )  # type: ignore
        if reward_space is not None:
            reward_spec = reward_spec.expand(*batch_size, *reward_spec.shape)  # type: ignore

        self.done_spec = self._make_done_spec()  # type: ignore
        self.action_spec = action_spec  # type: ignore
        self.reward_spec = reward_spec  # type: ignore
        self.observation_spec = observation_spec  # type: ignore

    def _terminal_swap_is_out_of_place(self) -> bool:
        """Whether a reader hands terminal rows to the out-of-place swap.

        True when an `IsaacLabTerminalObsReader` in hand-off mode feeds
        `IsaacLabVecGymEnvTransform`, which copies instead of writing into
        the step's observation tensors.
        """
        return any(
            isinstance(reader, IsaacLabTerminalObsReader) and reader.hand_off_batched
            for reader in self.info_dict_reader
        )

    def read_done(self, terminated=None, truncated=None, done=None):  # type: ignore[override]
        terminated, truncated, done, do_break = super().read_done(
            terminated=terminated, truncated=truncated, done=done
        )
        if (
            _FAST_READ_DONE
            and int(getattr(self, "wrapper_frame_skip", 1)) == 1
            and isinstance(do_break, torch.Tensor)
        ):
            # `GymLikeEnv._step` only uses `do_break` to leave its frame-skip
            # loop, which runs once here and ends either way; taking the
            # tensor's truth value would block the host on the device queue.
            do_break = False
        return terminated, truncated, done, do_break

    def _output_transform(self, step_outputs_tuple):  # type: ignore
        # IsaacLab will modify the `terminated` and `truncated` tensors
        #  in-place. We clone them here to make sure data doesn't inadvertently get modified.
        # The variable naming follows torchrl's convention here.
        observations, reward, terminated, truncated, info = step_outputs_tuple
        self.log_infos.append(self._log_payload(info["log"]))

        done = terminated | truncated

        # IsaacLab emits Gymnasium-style keys: final_obs / final_info.
        # Keep only terminal entries to avoid introducing scalar log info into
        # the tensordict info path.
        if isinstance(info, dict) and "final_obs" in info:
            info = {"final_obs": info["final_obs"]}
            if self._terminal_swap_is_out_of_place():
                # `IsaacLabVecGymEnvTransform` never writes into these
                # tensors (see its `_step`), so they need no defensive copy.
                return (
                    observations,
                    reward.unsqueeze(-1),
                    terminated.clone().to(dtype=torch.bool),
                    truncated.clone().to(dtype=torch.bool),
                    done.clone().to(dtype=torch.bool),
                    info,
                )
            # On a step with resets, TorchRL's VecGymEnvTransform writes the
            # terminal observation IN PLACE into the done rows of ``next``. The
            # tensors returned here are the env's own ``obs_buf``, so without a
            # copy that write also lands in ``obs_buf``; an episode that ends
            # on its first step then publishes the PREVIOUS episode's terminal
            # observation as its own (measured: every length-1 episode on
            # the explicit G1 task). Only reset steps pay the copy.
            observations = CloneObsBuf(observations)
            return (
                observations,
                reward.unsqueeze(-1),
                terminated.clone().to(dtype=torch.bool),
                truncated.clone().to(dtype=torch.bool),
                done.clone().to(dtype=torch.bool),
                info,
            )
        else:
            return (
                observations,
                reward.unsqueeze(-1),
                terminated.clone().to(dtype=torch.bool),
                truncated.clone().to(dtype=torch.bool),
                done.clone().to(dtype=torch.bool),
                {},
            )

    def _reset_output_transform(self, reset_data):
        """Transform the output of the reset method."""
        observations, info = reset_data
        self.log_infos.append(self._log_payload(info["log"]))
        return (observations, {})

    @staticmethod
    def _normalize_nested_batch_sizes(td: TensorDictBase) -> None:
        """Recursively align nested TensorDict batch sizes with their parent."""
        parent_batch = td.batch_size
        for key in td.keys():
            value = td.get(key)
            if not isinstance(value, TensorDictBase):
                continue
            current = value
            if current.batch_size != parent_batch:
                current = current.clone()
                current.batch_size = parent_batch
                td.set(key, current)
            IsaacLabWrapper._normalize_nested_batch_sizes(current)

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        # WMBPTT steps the Isaac env directly; everything that comes through
        # here reads the groups `wm_only_observations` no longer computes.
        if getattr(
            getattr(self._base_isaac_env(), "cfg", None), "wm_only_observations", False
        ):
            raise ValueError(
                "env.wm_only_observations removes the policy and critic groups "
                "this wrapper returns; it is a WMBPTT-only option."
            )
        td_out = super()._step(tensordict)
        self._normalize_nested_batch_sizes(td_out)
        return td_out

    def _reset(self, tensordict: TensorDictBase | None = None, **kwargs) -> TensorDict:
        td_out = super()._reset(tensordict, **kwargs)
        self._normalize_nested_batch_sizes(td_out)
        return td_out


def CloneObsBuf(
    obs_buf: torch.Tensor | Mapping[str, object] | TensorDictBase,
) -> torch.Tensor | dict[str, object]:
    """Clone nested observation structures while normalizing TensorDict groups to dicts."""
    if isinstance(obs_buf, torch.Tensor):
        return obs_buf.clone()
    if isinstance(obs_buf, TensorDictBase):
        # Convert TensorDict groups to plain dicts so GymWrapper read_obs() can
        # re-encode them with Composite spec batch semantics.
        return {k: CloneObsBuf(obs_buf.get(k)) for k in obs_buf.keys()}
    if isinstance(obs_buf, Mapping):
        return {k: CloneObsBuf(v) for k, v in obs_buf.items()}
    return obs_buf


def CheckObsBufForNaN(
    obs_buf: dict[str, torch.Tensor | dict], prefix: str = ""
) -> None:
    """Recursively check nested observation dicts for NaNs."""
    for k, v in obs_buf.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            CheckObsBufForNaN(v, key)
        elif isinstance(v, torch.Tensor) and torch.isnan(v).any():
            first_row = v[0] if v.ndim > 0 else v
            logger.error(
                "NaN values found in observation %s during step. First row: %s",
                key,
                first_row,
            )
            raise ValueError(
                f"NaN values found in observation {key} during step. "
                "This is likely due to an error in the environment or the model."
            )


def _zero_from_spec(spec: object) -> object:
    """``spec.zero()`` without ``TensorDict.from_dict`` on Composite specs.

    ``Composite.zero`` builds its container through ``TensorDict.from_dict``,
    which dispatches every leaf through ``TensorDict.from_any``; for a plain
    tensor that falls through to ``importlib.util.find_spec("pandas")``, a
    ``sys.path`` walk per leaf per step (about 1 ms per control step for the
    G1 observation groups). Leaves still come from their own ``zero()``, and
    the container gets the same keys, batch size, device and dim names.
    """
    if (
        not _FAST_SPEC_ZERO
        or not isinstance(spec, Composite)
        or spec.data_cls is not None
    ):
        return spec.zero()
    try:
        device = spec.device
    except RuntimeError:
        device = spec._device
    return TensorDict(
        {
            key: _zero_from_spec(spec[key])
            for key in spec.keys()
            if spec[key] is not None
        },
        batch_size=spec._safe_shape,
        device=device,
        names=spec._td_dim_names,
    )


class IsaacLabTerminalObsReader(terminal_obs_reader):
    """A terminal observation reader for IsaacLab environments.

    This reader extracts the terminal observation from the environment's info dictionary.
    It is used to read the terminal observation when the environment is reset."""

    def __init__(self, observation_spec: Composite, backend, name: str = "final"):
        super().__init__(observation_spec=observation_spec, backend=backend, name=name)
        # Set by `attach_terminal_obs_reader` when it installs
        # `IsaacLabVecGymEnvTransform`: the batched terminal rows are then
        # handed to that transform instead of being scattered into full-size
        # ``final`` buffers that it would gather back.
        self.hand_off_batched = False
        self.pending_terminal_obs: tuple[torch.Tensor, object] | None = None
        # Avoid recursive final/final specs when the wrapped observation spec
        # already exposes a terminal group.
        if name in self._obs_spec.keys():
            self._obs_spec.pop(name)
        # Provide info specs upfront to avoid dummy rollouts in set_info_dict_reader.
        self._info_spec = Composite({self.name: self._obs_spec.clone()}, shape=[])

    @staticmethod
    def _extract_nested_value(obs: object, key_path: tuple[str, ...]) -> object | None:
        cur = obs
        for key in key_path:
            # Observation groups arrive as plain dicts or, after TorchRL's
            # read_obs, as TensorDicts. Both must be walked.
            if isinstance(cur, (dict, TensorDictBase)):
                if key not in cur.keys():
                    return None
                cur = cur[key]
            else:
                # Scalar/tensor observation: only valid for single-key paths.
                return cur if len(key_path) == 1 else None
        return cur

    def _build_spec_buffer(
        self,
        spec: object,
        per_env_obs: list[object | None],
        key_path: tuple[str, ...],
    ) -> object:
        if isinstance(spec, Composite):
            td = spec.zero()
            for subkey in spec.keys():
                child = self._build_spec_buffer(
                    spec[subkey], per_env_obs, (*key_path, subkey)
                )
                td.set(subkey, child)
            return td

        buf = spec.zero()
        device = buf.device
        num_envs = len(per_env_obs)
        for i in range(num_envs):
            obs = per_env_obs[i]
            if obs is None:
                continue
            val = self._extract_nested_value(obs, key_path)
            if val is None:
                continue
            if isinstance(val, torch.Tensor):
                buf[i] = val.to(device=device)
            else:
                buf[i] = torch.as_tensor(val, device=device)
        return buf

    def _build_spec_buffer_batched(
        self,
        spec: object,
        env_ids: torch.Tensor,
        obs: object,
        key_path: tuple[str, ...],
    ) -> object:
        if isinstance(spec, Composite):
            td = _zero_from_spec(spec)
            for subkey in spec.keys():
                td.set(
                    subkey,
                    self._build_spec_buffer_batched(
                        spec[subkey], env_ids, obs, (*key_path, subkey)
                    ),
                )
            return td

        buf = spec.zero()
        val = self._extract_nested_value(obs, key_path)
        if isinstance(val, torch.Tensor):
            ids = env_ids.to(device=buf.device)
            buf[ids] = val.to(device=buf.device, dtype=buf.dtype)
        return buf

    def __call__(self, info_dict, tensordict):
        # ImitationRLEnv publishes final_obs batched:
        # {"_env_ids": LongTensor[k], "obs": nested dict of [k, ...] tensors}.
        # Other Gymnasium envs publish the object array
        # np.ndarray(num_envs, dtype=object) with per-env dicts; both are
        # consumed here.
        backend_key = self.backend_key[self.backend]
        final_obs = info_dict.pop(backend_key, None)
        info_dict.pop(self.backend_info_key[self.backend], None)

        # We intentionally skip parent terminal_obs_reader.__call__ here to
        # avoid recursive `final` spec self-merging. The IsaacLab wrapper
        # passes through only terminal observation info for this reader.

        if self.hand_off_batched:
            # `IsaacLabVecGymEnvTransform` consumes the batched rows directly;
            # a step without published terminal rows hands off nothing.
            if final_obs is not None and not isinstance(final_obs, dict):
                raise RuntimeError(
                    "The terminal-observation hand-off needs the batched "
                    "`final_obs` of ImitationRLEnv ({'_env_ids', 'obs'}); got "
                    f"{type(final_obs).__name__}. attach_terminal_obs_reader "
                    "enables the hand-off only for ImitationRLEnv."
                )
            self.pending_terminal_obs = (
                None if final_obs is None else (final_obs["_env_ids"], final_obs["obs"])
            )
            return tensordict

        if final_obs is None:
            # No reset this step: only the declared zero buffers, no fill.
            for key in self.info_spec[self.name].keys():
                tensordict.set(
                    (self.name, key), _zero_from_spec(self.info_spec[self.name, key])
                )
            return tensordict

        if isinstance(final_obs, dict):
            env_ids = final_obs["_env_ids"]
            obs = final_obs["obs"]
            for key in self.info_spec[self.name].keys():
                spec = self.info_spec[self.name, key]
                buf = self._build_spec_buffer_batched(spec, env_ids, obs, (key,))
                tensordict.set((self.name, key), buf)
            return tensordict

        num_envs = len(final_obs)
        per_env_obs: list[object | None] = [None] * num_envs
        for i in range(num_envs):
            per_env_obs[i] = final_obs[i]

        for key in self.info_spec[self.name].keys():
            spec = self.info_spec[self.name, key]
            buf = self._build_spec_buffer(spec, per_env_obs, (key,))
            tensordict.set((self.name, key), buf)

        return tensordict


class IsaacLabVecGymEnvTransform(VecGymEnvTransform):
    """``VecGymEnvTransform`` with an index-based terminal-observation swap.

    The parent writes ``next[done] = final[done]``: a boolean-mask gather and a
    boolean-mask scatter per observation leaf, each running its own
    ``nonzero`` (a host synchronization), on top of the full-size ``final``
    buffers the reader scatters into first, and it needs the wrapper's and its
    own defensive copies of every observation. Here the reader hands over the
    env's batched terminal rows (``final_obs["_env_ids"]`` and the ``[k, ...]``
    observations, the same rows in the same ascending env order), and each
    ``next`` leaf becomes ``leaf.index_copy(0, env_ids, rows)``: a new tensor,
    so the env's buffers stay untouched and serve as the saved reset
    observations without a copy. The ``next`` observations hold the parent's
    values: done rows without a published terminal row (a reset the env's
    visualizer requested after `_publish_terminal_obs`) and leaves the
    terminal obs lack get zeros, as the zero ``final`` buffers gave them.
    The saved reset observations are kept on every step, so a partial reset
    always finds them; the parent keeps them only on steps with a done row,
    which it learns through a host synchronization.

    One deliberate difference from the parent: it writes the terminal rows
    into ``next`` in place, so on a step whose done rows have no published
    terminal row (the wrapper then hands over the env's ``obs_buf`` without a
    copy) it also zeroes those rows of the env's own buffer. This path leaves
    the env's buffer untouched.

    The reader is found through the container's base env on every step:
    TorchRL clones transforms when it re-wraps an env, so a stored reference
    could point at a copy.
    """

    def _terminal_reader(self) -> IsaacLabTerminalObsReader | None:
        # A test may pin the reader without building an env around it.
        pinned = self.__dict__.get("_terminal_reader_ref", None)
        if pinned is not None:
            return pinned
        base = getattr(self.container, "base_env", None)
        for candidate in getattr(base, "info_dict_reader", None) or []:
            if (
                isinstance(candidate, IsaacLabTerminalObsReader)
                and candidate.hand_off_batched
            ):
                return candidate
        return None

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        reader = None if self.native_autoreset else self._terminal_reader()
        if reader is None:
            if (
                not self.native_autoreset
                and self.final_name not in next_tensordict.keys()
            ):
                # A hand-off reader suppressed ``final`` but is not reachable
                # from this transform (e.g. wrapped); the parent would then
                # write its missing-observation value into every done row.
                raise RuntimeError(
                    "IsaacLabVecGymEnvTransform found no terminal-observation "
                    f"reader in hand-off mode and no {self.final_name!r} entry; "
                    "the reader must be the env's own IsaacLabTerminalObsReader."
                )
            return super()._step(tensordict, next_tensordict)
        done = False
        for done_key in self.done_keys:
            done = done | next_tensordict.get(done_key)
        if done is False:
            raise RuntimeError(
                f"Could not find any done signal in tensordict:\n{tensordict}"
            )
        self._memo["done"] = done
        next_tensordict.pop(self.final_name, None)
        pending = reader.pending_terminal_obs
        reader.pending_terminal_obs = None
        final_spec = reader.info_spec[reader.name]
        leaf_keys = list(final_spec.keys(True, True))
        # The wrapper hands over the env's own observation buffers (it skips
        # `CloneObsBuf`), and nothing below writes into them: they are the
        # saved reset observations as they are, and each ``next`` leaf becomes
        # a new tensor. One copy per leaf instead of the parent's two.
        saved_next = next_tensordict.select(*leaf_keys)
        done_rows = done.reshape(done.shape[0], -1).any(-1)
        if pending is None:
            env_ids = terminal_obs = None
            unpublished = done_rows
        else:
            env_ids, terminal_obs = pending
            env_ids = env_ids.to(device=done_rows.device, dtype=torch.long)
            # Done rows without a published terminal row; device-side, no sync.
            unpublished = done_rows.index_fill(0, env_ids, False)
        for key in leaf_keys:
            dst = next_tensordict.get(key)
            new = dst
            if env_ids is not None:
                ids = env_ids.to(device=dst.device)
                src = reader._extract_nested_value(
                    terminal_obs, key if isinstance(key, tuple) else (key,)
                )
                if isinstance(src, torch.Tensor):
                    src = src.to(device=dst.device, dtype=dst.dtype)
                else:
                    src = torch.zeros(
                        (ids.numel(), *dst.shape[1:]),
                        device=dst.device,
                        dtype=dst.dtype,
                    )
                new = dst.index_copy(0, ids, src)
            mask = unpublished.to(device=dst.device).reshape(
                -1, *([1] * (dst.ndim - 1))
            )
            new = torch.where(
                mask, torch.zeros((), device=dst.device, dtype=dst.dtype), new
            )
            next_tensordict.set(key, new)
        self._memo["saved_next"] = saved_next
        return next_tensordict


def _publishes_batched_final_obs(env) -> bool:
    """Whether the Isaac Lab env under a TorchRL wrapper chain is an ``ImitationRLEnv``.

    Only :class:`~imitationlab.envs.isaaclab.envs.ImitationRLEnv` publishes the
    batched ``final_obs`` the hand-off needs; other Gymnasium envs publish an
    object array.
    """
    from .imitation_rl_env import ImitationRLEnv

    current = env
    for _ in range(8):
        inner = current.__dict__.get("_env") if hasattr(current, "__dict__") else None
        if inner is not None:
            return isinstance(getattr(inner, "unwrapped", inner), ImitationRLEnv)
        current = getattr(current, "base_env", None)
        if current is None:
            return False
    return False


def attach_terminal_obs_reader(env):
    """Register the terminal-observation reader and return the OUTER env.

    ``IsaacLabWrapper(...)`` returns ``TransformedEnv(wrapper, VecGymEnvTransform)``.
    ``set_info_dict_reader`` is forwarded to the inner wrapper and returns it, so
    ``env = env.set_info_dict_reader(...)`` drops the transform. Without it a
    partial reset returns no observation and TorchRL zero-fills the first
    observation of every new episode. Keep the outer env.

    With the ``terminal_swap`` collection fast path on and an
    :class:`~imitationlab.envs.isaaclab.envs.ImitationRLEnv` underneath (the env that
    publishes batched ``final_obs``), the ``VecGymEnvTransform`` that the
    wrapper attached is replaced by :class:`IsaacLabVecGymEnvTransform` (same
    final name, missing-observation value and autoreset mode), and the reader
    hands it the batched terminal rows. Every other env keeps the stock path.
    """
    reader = IsaacLabTerminalObsReader(
        observation_spec=env.observation_spec, backend="gymnasium"
    )
    env.set_info_dict_reader(reader)  # type: ignore
    transform = getattr(env, "transform", None)
    if (
        _FAST_TERMINAL_SWAP
        and type(transform) is VecGymEnvTransform
        and not transform.native_autoreset
        and _publishes_batched_final_obs(env)
    ):
        env.transform = IsaacLabVecGymEnvTransform(
            final_name=transform.final_name,
            missing_obs_value=transform.missing_obs_value,
            native_autoreset=False,
        )
        reader.hand_off_batched = True
        logger.info("terminal_swap fast path: IsaacLabVecGymEnvTransform installed.")
    elif _FAST_TERMINAL_SWAP:
        logger.warning(
            "terminal_swap fast path requested but not installed (needs "
            "ImitationRLEnv and the stock VecGymEnvTransform); using the stock swap."
        )
    return env
