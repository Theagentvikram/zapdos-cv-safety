"""Restricted zones: per-camera floor polygons tested against each person's ground point.

The ground point is the bottom-centre of the box, i.e. the feet, because the polygon is drawn on the
floor; a box overlapping the polygon from 1 m in front is not inside it. When the feet are hidden
(box much shorter than the track's recent full height, or cut by the bottom frame edge) the point is
estimated as top + remembered full height. Entering needs the point a margin inside the polygon and
leaving needs it the same margin outside, so someone standing on the edge doesn't flicker. A track
that has never been detected as a confident person (texture in a grate, a coat on a hook) is not
`confirmed`: its zone evidence still accumulates, but it can't alert until it is. While the camera
is judged moved (view.py) every zone reads "can't tell".
"""
import json
from pathlib import Path

import cv2
import numpy as np

BOTTOM_EDGE_PX = 2      # a box ending this close to the bottom of the frame has its feet cut off
HIDDEN_FEET_CONF = 0.7  # an estimated ground point is weaker evidence than visible feet


class ZoneSet:
    def __init__(self, path, frame_size, cfg, fps):
        self.cfg = cfg
        self.decay = cfg["height_decay_per_s"] ** (1 / fps)  # per processed frame
        self.polygons = {}  # zone_id -> polygon in this video's pixels
        self.camera_id = None
        self.reference = None  # image of the view the polygons were drawn on
        self.suspended = False  # camera moved: zones no longer match the floor
        self.frame_h = frame_size[1]
        self.inside = {}   # (track_id, zone_id) -> debounced inside state
        self.full_h = {}   # track_id -> decaying max box height (unoccluded height estimate)
        self.best_conf = {}  # track_id -> highest person confidence seen
        if path is None:
            return
        with open(path) as f:
            data = json.load(f)
        self.camera_id = data.get("camera_id")
        if data.get("reference"):
            self.reference = Path(path).parent / data["reference"]
        ref_w, ref_h = data.get("frame_size", frame_size)
        scale = np.array([frame_size[0] / ref_w, frame_size[1] / ref_h], np.float32)
        for z in data["zones"]:
            self.polygons[z["zone_id"]] = (np.array(z["polygon"], np.float32) * scale).reshape(-1, 1, 2)

    def ground_point(self, track_id, box):
        x1, y1, x2, y2 = box
        h = y2 - y1
        full = max(h, self.full_h.get(track_id, 0.0) * self.decay)
        self.full_h[track_id] = full
        hidden = h < self.cfg["hidden_feet_ratio"] * full or y2 >= self.frame_h - BOTTOM_EDGE_PX
        foot_y = y1 + full if hidden else y2
        return ((x1 + x2) / 2, foot_y), hidden

    def read(self, track_id, box, conf):
        """Returns ({zone_id: (inside, conf)}, (ground_point, feet_hidden)) for this person; ({}, None)
        when the camera has no zones."""
        if not self.polygons:
            return {}, None
        pt, hidden = self.ground_point(track_id, box)
        self.best_conf[track_id] = max(self.best_conf.get(track_id, 0.0), float(conf))
        if self.suspended:
            return {zid: (None, 0.0) for zid in self.polygons}, (pt, hidden)
        margin = self.cfg["hysteresis"] * (box[3] - box[1])
        out = {}
        for zid, poly in self.polygons.items():
            d = cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), True)  # >0 inside
            was = self.inside.get((track_id, zid), False)
            now = d > -margin if was else d > margin
            self.inside[(track_id, zid)] = now
            out[zid] = (now, float(conf) * (HIDDEN_FEET_CONF if hidden else 1.0))
        return out, (pt, hidden)

    def confirmed(self, track_id):
        """Has this track ever been detected as a confident person?"""
        return self.best_conf.get(track_id, 0.0) >= self.cfg.get("confirm_conf", 0.0)
