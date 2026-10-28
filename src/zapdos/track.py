"""Person tracking with ByteTrack (ultralytics implementation).

ByteTrack over DeepSORT: no per-crop ReID network (DeepSORT's cost grows with people on screen and
CCTV crops are too small for good appearance features), and its second association pass keeps
low-score boxes, which carries tracks through partial occlusion behind machines.
"""
from types import SimpleNamespace

import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker


class PersonTracker:
    def __init__(self, cfg, process_fps):
        args = SimpleNamespace(
            track_high_thresh=cfg["track_high_thresh"],
            track_low_thresh=cfg["track_low_thresh"],
            new_track_thresh=cfg["new_track_thresh"],
            match_thresh=cfg["match_thresh"],
            fuse_score=cfg["fuse_score"],
            track_buffer=int(round(cfg["track_buffer_s"] * process_fps)),
        )
        self.tracker = BYTETracker(args)

    def __call__(self, persons):
        """Returns (N, 6): x1, y1, x2, y2, track_id, conf for tracks matched in this frame."""
        out = self.tracker.update(persons)
        if len(out) == 0:
            return np.zeros((0, 6))
        return out[:, :6]

    def hidden(self, max_frames):
        """Tracks lost for <= max_frames that the tracker still holds: (N, 5) Kalman-predicted x1, y1, x2,
        y2, track_id. For drawing only; nothing is read from a box nobody can see."""
        t = self.tracker
        rows = [[*s.xyxy, s.track_id] for s in t.lost_stracks if t.frame_id - s.end_frame <= max_frames]
        return np.array(rows, float).reshape(-1, 5)
