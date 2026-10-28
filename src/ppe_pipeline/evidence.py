"""Bounded annotated pre-roll and post-roll clip recording, owned by the writer thread."""
from collections import deque
from pathlib import Path
import cv2


class ClipRecorder:
    def __init__(self, fps, shape, seconds=3.0, max_width=960):
        self.fps, self.seconds = fps, seconds
        h, w = shape
        scale = min(1, max_width / w)
        self.size = (max(2, int(w * scale) // 2 * 2), max(2, int(h * scale) // 2 * 2))
        self.frames = deque(maxlen=max(1, int(fps * seconds) + 2))
        self.active = []
        self.paths = set()

    def push(self, t, canvas, requests):
        small = cv2.resize(canvas, self.size)
        for path, event_t in requests:
            if path in self.paths:
                continue
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.size)
            if not writer.isOpened():
                raise OSError(f"cannot write evidence clip: {path}")
            for old_t, old in self.frames:
                if old_t >= event_t - self.seconds - 1e-6:
                    writer.write(old)
            self.paths.add(path)
            self.active.append((event_t + self.seconds, writer))
        pending = []
        for end, writer in self.active:
            if t <= end + 1e-6:
                writer.write(small)
                pending.append((end, writer))
            else:
                writer.release()
        self.active = pending
        self.frames.append((t, small))

    def close(self):
        for _, writer in self.active:
            writer.release()
        self.active.clear()
