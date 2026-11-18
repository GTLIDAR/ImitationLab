"""Persistent current-reference frame for the masked step.

The eager data plane rebinds ``current_expert_frame`` to a freshly built
TensorDict at every refresh (prefetch slot views plus derived fields). A CUDA
graph cannot follow a rebinding: it would keep reading the tensors that were
current at capture. :class:`ReferenceFrameState` instead owns one packed
``[N, W]`` row buffer in the runtime-field layout and a fixed TensorDict whose
leaves are views of it or persistent derived buffers recomputed in place, so
the frame changes by value only. The derived fields are computed with the same
ops as ``TrajectoryManager.attach_reference_fields`` and
``_convert_reference_quats_to_xyzw``, so every value equals the eager frame.

Rows are filled in two ways:

- :meth:`gather_masked`: rows for the masked environments from the reference
  backing (the host-mapped packed store, read in one Triton kernel that skips
  unmasked rows, or any CUDA tensor);
- :meth:`select_into`: ``where(keep, self, other)`` against a same-layout
  packed buffer (the next-frame prefetch slot).
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from tensordict import TensorDict

try:  # Triton ships with CUDA builds of torch.
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU-only installs
    triton = None
    tl = None

# The packed reference store keeps quaternions scalar-first; Isaac Lab 3.0 is
# scalar-last (see expert_data_plane._convert_reference_quats_to_xyzw).
_WXYZ_TO_XYZW = (1, 2, 3, 0)


if triton is not None:

    @triton.jit
    def _masked_row_gather_kernel(
        src, idx, mask, dst, width, BLOCK: tl.constexpr
    ):  # pragma: no cover - exercised on GPU
        row = tl.program_id(0)
        if tl.load(mask + row) == 0:
            return
        src_row = tl.load(idx + row).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        keep = cols < width
        values = tl.load(src + src_row * width + cols, mask=keep)
        tl.store(dst + row.to(tl.int64) * width + cols, values, mask=keep)


def masked_row_gather_(
    dst: torch.Tensor, src: torch.Tensor, indices: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """``dst[i] = src[indices[i]]`` for rows with ``mask[i]``; others untouched.

    ``src`` may be a CUDA alias of host-mapped memory: only masked rows cross
    PCIe. Values are copied bit for bit.
    """
    if dst.ndim != 2 or src.ndim != 2 or dst.shape[1] != src.shape[1]:
        raise ValueError("masked_row_gather_ needs [N, W] dst and [R, W] src.")
    if not (dst.is_contiguous() and src.is_contiguous()):
        raise ValueError("masked_row_gather_ needs contiguous buffers.")
    if triton is None or not dst.is_cuda:
        gathered = torch.index_select(src, 0, indices.to(torch.long))
        torch.where(mask.unsqueeze(-1), gathered, dst, out=dst)
        return dst
    width = int(dst.shape[1])
    block = max(16, triton.next_power_of_2(width))
    _masked_row_gather_kernel[(int(dst.shape[0]),)](
        src,
        indices.to(torch.long),
        mask.to(torch.uint8),
        dst,
        width,
        BLOCK=block,
    )
    return dst


class ReferenceFrameState:
    """Fixed-address current reference frame (values change, tensors do not)."""

    def __init__(
        self,
        *,
        num_envs: int,
        device: torch.device,
        field_shapes: Mapping[str, tuple[int, ...]],
        field_order: tuple[str, ...],
        nan_joint_template: torch.Tensor,
        ref_to_target_map: torch.Tensor,
        target_to_ref_map: torch.Tensor,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.field_order = tuple(field_order)
        self.field_shapes = {name: tuple(field_shapes[name]) for name in field_order}
        width = sum(math.prod(self.field_shapes[name]) for name in field_order)
        self.packed = torch.zeros(
            (self.num_envs, width), dtype=dtype, device=self.device
        )
        self.raw: dict[str, torch.Tensor] = {}
        offset = 0
        for name in self.field_order:
            shape = self.field_shapes[name]
            size = math.prod(shape)
            self.raw[name] = self.packed[:, offset : offset + size].view(
                self.num_envs, *shape
            )
            offset += size
        for required in ("qpos", "qvel"):
            if required not in self.raw:
                raise KeyError(f"The packed reference layout lacks {required!r}.")

        self._template = nan_joint_template.to(device=self.device, dtype=dtype)
        self._ref_to_target = ref_to_target_map.to(device=self.device)
        self._target_to_ref = target_to_ref_map.to(device=self.device)
        num_joints = int(self._template.shape[-1])
        self._quat_index = torch.tensor(_WXYZ_TO_XYZW, device=self.device)
        self.joint_pos = torch.zeros(
            (self.num_envs, num_joints), dtype=dtype, device=self.device
        )
        self.joint_vel = torch.zeros_like(self.joint_pos)
        self.root_quat = torch.zeros(
            (self.num_envs, 4), dtype=dtype, device=self.device
        )
        self.body_quat_w: torch.Tensor | None = None
        if "body_quat_w" in self.raw:
            self.body_quat_w = torch.zeros_like(self.raw["body_quat_w"])

        qpos, qvel = self.raw["qpos"], self.raw["qvel"]
        leaves: dict[str, torch.Tensor] = {}
        for name in self.field_order:
            leaves[name] = self.raw[name]
        if self.body_quat_w is not None:
            leaves["body_quat_w"] = self.body_quat_w
        leaves["root_pos"] = qpos[:, 0:3]
        leaves["root_quat"] = self.root_quat
        leaves["root_lin_vel"] = qvel[:, 0:3]
        leaves["root_ang_vel"] = qvel[:, 3:6]
        leaves["joint_pos"] = self.joint_pos
        leaves["joint_vel"] = self.joint_vel
        self.frame = TensorDict(leaves, batch_size=[self.num_envs], device=self.device)

    # ------------------------------------------------------------------
    def refresh_derived(self) -> None:
        """Recompute the derived leaves from the packed rows, in place."""
        qpos, qvel = self.raw["qpos"], self.raw["qvel"]
        if self.body_quat_w is not None:
            torch.index_select(
                self.raw["body_quat_w"], -1, self._quat_index, out=self.body_quat_w
            )
        torch.index_select(qpos[:, 3:7], -1, self._quat_index, out=self.root_quat)
        for out, src in (
            (self.joint_pos, qpos[:, 7:]),
            (self.joint_vel, qvel[:, 6:]),
        ):
            joint_src = src.index_select(-1, self._target_to_ref)
            filled = self._template.expand(out.shape).clone()
            filled[..., self._ref_to_target] = joint_src
            out.copy_(filled)

    def gather_masked(
        self, source: torch.Tensor, indices: torch.Tensor, mask: torch.Tensor
    ) -> None:
        """Rows of ``source`` at ``indices`` for the masked envs, then derived."""
        masked_row_gather_(self.packed, source, indices, mask)
        self.refresh_derived()

    def select_into(self, keep: torch.Tensor, other_packed: torch.Tensor) -> None:
        """``packed = where(keep, packed, other_packed)``, then derived."""
        torch.where(keep.unsqueeze(-1), self.packed, other_packed, out=self.packed)
        self.refresh_derived()

    def adopt(self, frame: TensorDict) -> None:
        """Copy an eager frame's raw fields in and check every leaf matches."""
        for name in self.field_order:
            value = frame.get(name)
            if name == "body_quat_w":
                # The eager frame holds the converted (xyzw) quaternions.
                inverse = torch.argsort(self._quat_index)
                value = value.index_select(-1, inverse)
            self.raw[name].copy_(value)
        self.refresh_derived()
        mismatched = [
            key
            for key in self.frame.keys()
            if key in frame.keys()
            and not torch.equal(
                torch.nan_to_num(self.frame.get(key), nan=1e30),
                torch.nan_to_num(frame.get(key), nan=1e30),
            )
        ]
        missing = sorted(set(frame.keys()) - set(self.frame.keys()))
        if mismatched or missing:
            raise RuntimeError(
                "The persistent reference frame does not reproduce the eager "
                f"frame: mismatched {mismatched}, missing {missing}."
            )


__all__ = ["ReferenceFrameState", "masked_row_gather_"]
