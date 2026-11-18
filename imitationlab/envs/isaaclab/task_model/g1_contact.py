# Ported from NDM Gym (https://github.com/Frank-ZY-Dou/ndm-gym @ 05ea9f3, ndm_gym/tasks/contact.py), which ports it from
# BlinkMimic (https://github.com/fuyichen0725/blinkmimic @ d264658, src/blinkmimic/mdp.py and motion.py), under the MIT
# License:
#
# Copyright (c) 2026 NDM Gym Contributors
# Copyright (c) 2026 ETH Zurich, Chong Zhang
# Copyright (c) 2026 Fuyi Chen
# Copyright (c) 2026 Huy-Quang-Dao (InstantTrack, from which this software is derived)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""The foot-contact term of the imagined reward (``wm.contact_reward``; ``wm.ref_contact_fix``).

InstantTrack's contact reward, with BlinkMimic's reference contact from each clip's own
floor, as NDM Gym ports them: a term of the world model's reward only, which matches a
smooth contact indicator of each foot to the reference's contact state and rewards a
low horizontal foot speed while the reference foot is down. The env's reward, which
evaluation scores, is unchanged.

A foot is the ankle-roll link origin. On ``lafan1_corrected_40@d972c37c`` (2026-11-07,
CPU, FK of every bank row) its height has a 5th percentile of 0.036-0.037 m, the
per-clip floors lie in 0.033-0.046 m (4 of 40 clips have a foot floor above the fixed
0.045 m), and 28% of foot-frames are in contact by the fixed rule against 44% by the
calibrated one. The constants are NDM Gym's, set on its own retarget.
"""

from __future__ import annotations

import torch
from torch import Tensor

FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
CONTACT_Z = 0.045
"""Ankle-roll link origin height (m) below which a foot counts as in contact."""
CONTACT_WIDTH = 0.005
"""Width (m) of the sigmoid of the model's contact indicator."""
SLIP_SPEED = 0.3
"""Horizontal foot speed (m/s) of the slip kernel."""
FLOOR_QUANTILE = 0.05
"""``wm.ref_contact_fix``: a clip's floor is this quantile of each foot's height in the clip."""
FLOOR_MARGIN = 0.015
"""``wm.ref_contact_fix``: a reference foot is down below its clip's floor plus this (m)..."""
STANCE_SPEED = 0.5
"""... and slower than this horizontally (m/s)."""


SORT_ROWS = 4_194_304
"""``clip_quantile`` sorts this many rows at a time (whole clips): the one-sort form over
the 47.5M-row BONES-SEED bank peaked at 8.4 GiB of GPU memory for its index arrays."""


def clip_quantile(
    values: Tensor,
    start: Tensor,
    length: Tensor,
    q: float,
    *,
    sort_rows: int = SORT_ROWS,
) -> Tensor:
    """``torch.quantile(values[s : s + n], q, dim=0)`` (linear interpolation) for every
    clip of contiguous rows (``start``, ``length``), in one sort per group of whole clips
    of at most ``sort_rows`` rows (or one clip) instead of one call per clip: a bank has
    up to 129,785 clips. The per-clip result does not depend on the grouping. ``values``
    is (rows, F); returns (clips, F)."""
    out = values.new_empty(length.numel(), values.shape[-1])
    ends = (start + length).tolist()
    starts = start.tolist()
    first = 0
    while first < len(starts):
        last = first
        while last + 1 < len(starts) and ends[last + 1] - starts[first] <= sort_rows:
            last += 1
        lo, hi = starts[first], ends[last]
        out[first : last + 1] = _clip_quantile_sorted(
            values[lo:hi], start[first : last + 1] - lo, length[first : last + 1], q
        )
        first = last + 1
    return out


def _clip_quantile_sorted(
    values: Tensor, start: Tensor, length: Tensor, q: float
) -> Tensor:
    """One stable sort for the clips that make up ``values`` (``start`` relative to it)."""
    rank = torch.repeat_interleave(
        torch.arange(length.numel(), device=values.device), length
    )
    by_value, order = values.sort(dim=0, stable=True)
    # A stable sort by clip keeps each clip's rows in value order, at its own rows.
    within = by_value.gather(0, rank[order].argsort(dim=0, stable=True))
    # torch.quantile's ranks: q (in the values' dtype) times the last index, truncated
    # below, rounded up above, and the fraction as the weight.
    pos = torch.tensor(q, dtype=values.dtype, device=values.device) * (length - 1).to(
        values.dtype
    )
    below, above = pos.long(), pos.ceil().long()
    weight = (pos - below.to(values.dtype)).unsqueeze(-1)
    return torch.lerp(within[start + below], within[start + above], weight)


@torch.no_grad()
def reference_contact(library, calibrated: bool, chunk: int = 65536) -> Tensor:
    """Each bank row's reference foot contact (rows, 2), bool: the foot below
    ``CONTACT_Z``; with ``calibrated`` (``wm.ref_contact_fix``) below its clip's floor
    (``FLOOR_QUANTILE`` of that foot's height in the clip) plus ``FLOOR_MARGIN`` and
    slower than ``STANCE_SPEED`` horizontally, since the retargeted feet of some clips
    sit about 1 cm higher. Heights and speeds are FK link-origin values of the
    reference rows (``library.gather``)."""
    feet = library.body_index(FEET)
    rank, frame = library._bank_rows()
    z, speed = [], []
    for lo in range(0, library.num_rows, chunk):
        ref = library.gather(rank[lo : lo + chunk], frame[lo : lo + chunk])
        z.append(ref.body_pos[:, feet, 2])
        speed.append(ref.body_lin_vel[:, feet, :2].norm(dim=-1))
    height = torch.cat(z)
    if not calibrated:
        return height < CONTACT_Z
    floor = clip_quantile(height, library.start, library.length, FLOOR_QUANTILE)
    return (height < floor[rank] + FLOOR_MARGIN) & (torch.cat(speed) < STANCE_SPEED)


def contact_term(foot_pos: Tensor, foot_vel: Tensor, reference: Tensor) -> Tensor:
    """Per sample, the mean over the feet (positions and linear velocities (..., feet, 3),
    world frame; reference contact (..., feet) in {0, 1}) of the contact match
    1 - (c - c_ref)^2, c = sigmoid((``CONTACT_Z`` - z) / ``CONTACT_WIDTH``), plus
    c_ref exp(-|v_xy|^2 / ``SLIP_SPEED``^2). It lies in [0, 2]."""
    c = torch.sigmoid((CONTACT_Z - foot_pos[..., 2]) / CONTACT_WIDTH)
    slip = foot_vel[..., :2].square().sum(-1)
    return (
        (1 - (c - reference).square()) + reference * torch.exp(-slip / SLIP_SPEED**2)
    ).mean(-1)
