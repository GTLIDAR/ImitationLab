"""Opt-in timing at completed optimizer iterations, after warmup and capture."""

from __future__ import annotations

import json
import math
from pathlib import Path
import time


class CalibrationClock:
    def __init__(self, output: str, seconds: float, warmup: int):
        if not math.isfinite(seconds) or seconds <= 0 or warmup < 1:
            raise ValueError(
                "Calibration needs positive measured seconds and warmup iterations."
            )
        self.output = Path(output)
        self.seconds = seconds
        self.warmup = warmup
        self.iterations = 0
        self.start = None
        self.start_frames = 0
        self.done = False

    def step(self, frames: int, *, now: float | None = None) -> bool:
        if self.done:
            return True
        self.iterations += 1
        now = time.monotonic() if now is None else now
        if self.iterations < self.warmup:
            return False
        if self.start is None:
            self.start, self.start_frames = now, frames
            return False
        elapsed = now - self.start
        measured_frames = frames - self.start_frames
        if (
            elapsed >= self.seconds
            and measured_frames > 0
            and self.iterations - self.warmup >= 2
        ):
            self.output.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "warmup_iterations": self.warmup,
                "measured_iterations": self.iterations - self.warmup,
                "measured_seconds": elapsed,
                "measured_frames": measured_frames,
                "frames_per_second": measured_frames / elapsed,
            }
            self.output.write_text(json.dumps(payload, indent=2) + "\n")
            self.done = True
        return self.done
