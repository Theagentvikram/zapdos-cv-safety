"""Floor-contact zone tests with dwell time and a signed-distance dead band."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np


def load_zones(path, shape):
    if path is None:
        return []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    height, width = shape[:2]
    zones, seen = [], set()
    for item in data["zones"]:
        if not item.get("enabled", True):
            continue
        name = str(item["id"])
        points = np.asarray(item["polygon"], np.float32)
        if name in seen or points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
            raise ValueError(f"invalid or duplicate zone: {name}")
        if data.get("coordinates", "normalized") == "normalized":
            if (points < 0).any() or (points > 1).any():
                raise ValueError("normalized zone coordinates must be in [0, 1]")
            points *= [width, height]
        elif data["coordinates"] != "pixels":
            raise ValueError("zone coordinates must be normalized or pixels")
        if cv2.contourArea(points) <= 0:
            raise ValueError(f"zero-area zone: {name}")
        zones.append({**item, "id": name, "contour": points})
        seen.add(name)
    return zones


def floor_point(box, others, shape, cfg):
    """BBox bottom centre is a proxy, not proof of visible feet. Reject known occlusions."""
    x1, y1, x2, y2 = box
    point = (float((x1 + x2) / 2), float(y2))
    if x2<=x1 or y2<=y1:
        return None
    # A short/wide box often ends on equipment rather than the floor. This is
    # conservative abstention, not a claim to estimate hidden feet or seated pose.
    if (y2-y1)<cfg.get("min_floor_box_height_px",0) or (y2-y1)/(x2-x1)<cfg.get("min_full_body_aspect",0):
        return None
    if y2 >= shape[0] - cfg.get("border_margin_px", 3):
        return None
    for b in others:
        if b[0] <= point[0] <= b[2] and b[1] <= point[1] <= b[3]:
            return None
    for polygon in cfg.get("feet_occlusion_polygons", []):
        contour = np.asarray(polygon, np.float32) * [shape[1], shape[0]]
        if cv2.pointPolygonTest(contour.astype(np.float32), point, False) >= 0:
            return None
    return point


@dataclass
class ZoneState:
    cfg: dict
    active: bool = False
    candidate_t: float | None = None
    exit_t: float | None = None
    last_observed_t: float | None = None
    start_t: float | None = None
    episode_number: int = 0

    def step(self, t, distance):
        started = ended = False
        if distance is None:
            self.candidate_t = self.exit_t = None
            if self.last_observed_t is not None and t - self.last_observed_t > self.cfg.get("unknown_hold_seconds", 1):
                ended, self.active = self.active, False
            return started, ended
        if self.last_observed_t is not None and t - self.last_observed_t > self.cfg.get("unknown_hold_seconds", 1):
            self.candidate_t = self.exit_t = None
            ended, self.active = self.active, False
        self.last_observed_t = t
        margin = self.cfg.get("boundary_margin_px", 5)
        if distance >= margin:
            self.exit_t = None
            if not self.active:
                if self.candidate_t is None:
                    self.candidate_t = t
                if t - self.candidate_t + 1e-6 >= self.cfg.get("entry_seconds", 0.6):
                    self.active, started, self.start_t = True, True, self.candidate_t
                    self.episode_number += 1
        elif distance <= -margin:
            self.candidate_t = None
            if self.active:
                if self.exit_t is None:
                    self.exit_t = t
                if t - self.exit_t + 1e-6 >= self.cfg.get("exit_seconds", 0.6):
                    self.active, ended = False, True
                    self.exit_t = None
        else:
            # The boundary does not advance either dwell clock.
            self.candidate_t = self.exit_t = None
        return started, ended


def draw_zones(canvas, zones):
    for zone in zones:
        points = zone["contour"].astype(np.int32)
        cv2.polylines(canvas, [points], True, (255, 120, 0), 2)
        cv2.putText(canvas, zone["id"], tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 120, 0), 2)
