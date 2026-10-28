"""ByteTrack (Ultralytics implementation, configured from bytetrack.yaml) over person detections only."""

from __future__ import annotations

import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml


class PersonTracker:
    """One instance per video, so IDs and Kalman state never leak between videos."""

    def __init__(self, tracking_cfg: dict, fps: float, frame_shape: tuple):
        args = YAML.load(check_yaml(tracking_cfg["tracker"]))
        args.update(tracking_cfg.get("overrides") or {})
        if tracking_cfg.get("track_buffer_seconds") is not None:
            args["track_buffer"] = max(1, int(round(tracking_cfg["track_buffer_seconds"] * fps)))
        if args.get("tracker_type") != "bytetrack":
            raise SystemExit(f"expected a bytetrack config, got {args.get('tracker_type')}")
        self.args = args
        self.shape = frame_shape
        # Pinned Ultralytics version consumes track_buffer in update calls.
        self.tracker = BYTETracker(IterableSimpleNamespace(**args))

    def update(self, person_rows: np.ndarray, frame=None) -> list[tuple[int, int]]:
        """person_rows: (N, 6) x1, y1, x2, y2, conf, cls for person detections only.

        Returns (tracker_id, detection_index) for tracks matched to a real detection this frame.
        Lost tracks are not returned.
        """
        boxes = Boxes(np.asarray(person_rows, dtype=np.float32).reshape(-1, 6), self.shape)
        tracks = self.tracker.update(boxes, frame)
        return [(int(row[4]), int(row[7])) for row in np.asarray(tracks).reshape(-1, 8)]

    def lost(self) -> dict[int, tuple]:
        """Kalman-predicted boxes of tracks currently lost but still recoverable."""
        return {int(t.track_id): tuple(float(v) for v in t.xyxy) for t in self.tracker.lost_stracks}

    def alive_ids(self) -> set[int]:
        active = {int(t.track_id) for t in self.tracker.tracked_stracks if t.is_activated}
        return active | {int(t.track_id) for t in self.tracker.lost_stracks}
