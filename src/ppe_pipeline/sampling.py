"""Deterministic source-index sampling with original video timestamps."""
from dataclasses import dataclass
import math


@dataclass
class FrameSampler:
    source_fps: float
    target_fps: float | None = None
    next_t: float = 0.0

    def __post_init__(self):
        if not math.isfinite(self.source_fps) or self.source_fps <= 0:
            raise ValueError("source FPS must be positive and finite")
        if self.target_fps is not None and (not math.isfinite(self.target_fps) or self.target_fps <= 0):
            raise ValueError("target FPS must be positive and finite")
        self.rate = min(self.source_fps, self.target_fps or self.source_fps)

    def keep(self, index):
        t = index / self.source_fps
        if t + 1e-9 < self.next_t:
            return False
        self.next_t += 1 / self.rate
        return True
