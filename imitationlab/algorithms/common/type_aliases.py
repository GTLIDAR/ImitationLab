"""Optimizer and learning-rate scheduler class annotations."""

from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

OptimizerClass = type[Optimizer]
SchedulerClass = type[LRScheduler]
