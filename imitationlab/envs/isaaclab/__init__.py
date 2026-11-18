# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab environments for ImitationLab.

Importing this package leaves Gym registration lazy so the task model can be
used without Isaac Lab. Import :mod:`imitationlab.envs.isaaclab.tasks` or call
:func:`register_tasks` before resolving a task ID.
"""

from __future__ import annotations

from importlib import import_module


def register_tasks() -> None:
    """Register the imitation tasks in Gymnasium."""
    import_module(f"{__name__}.tasks")
