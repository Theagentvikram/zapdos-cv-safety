"""Cheap, configurable plausibility checks; never require worker motion.

Low-confidence persons are retained for ByteTrack's second-stage recovery. Invalid PPE
is kept as an uncertainty blocker for association rather than used as absence evidence.
Camera exclusions must be drawn explicitly; no background object is automatically masked.
"""
from __future__ import annotations

from collections import Counter
import cv2
import numpy as np


def filter_detections(rows, shape, cfg, camera=None):
    rows = np.asarray(rows, dtype=np.float32).reshape(-1, 6)
    kept, rejected = [], Counter()
    height, width = shape[:2]
    classes = cfg.get("classes", {})
    exclusions = (camera or {}).get("person_exclusion_polygons", [])
    for row in rows:
        if not np.isfinite(row).all():
            rejected["non_finite"] += 1
            continue
        x1, y1, x2, y2, conf, cls = map(float, row)
        if x2 <= x1 or y2 <= y1 or not 0 <= conf <= 1:
            rejected["invalid_box"] += 1
            continue
        rule = classes.get(int(cls), classes.get(str(int(cls)), {}))
        clipped = row.copy()
        clipped[[0, 2]] = np.clip(clipped[[0, 2]], 0, width)
        clipped[[1, 3]] = np.clip(clipped[[1, 3]], 0, height)
        w, h = clipped[2] - clipped[0], clipped[3] - clipped[1]
        reason = None
        if w < rule.get("min_width_px", 1) or h < rule.get("min_height_px", 1):
            reason = "too_small"
        elif conf < rule.get("min_conf", 0):
            reason = "low_confidence"
        elif not rule.get("min_aspect", 0) <= h / max(w, 1e-6) <= rule.get("max_aspect", float("inf")):
            reason = "aspect"
        elif w * h / (width * height) > rule.get("max_area_fraction", 1):
            reason = "area"
        if int(cls) == cfg.get("person_class", 0):
            point = (float((clipped[0] + clipped[2]) / 2), float(clipped[3]))
            for polygon in exclusions:
                contour = np.array(polygon, np.float32) * [width, height]
                if cv2.pointPolygonTest(contour.astype(np.float32), point, False) >= 0:
                    reason = "camera_exclusion"
        if reason:
            rejected[f"class_{int(cls)}_{reason}"] += 1
        else:
            kept.append(clipped)
    return np.asarray(kept, np.float32).reshape(-1, 6), dict(rejected)
