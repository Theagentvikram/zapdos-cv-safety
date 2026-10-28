"""Conservative visibility / image-quality gate for head and torso observations.

Passing the gate only means the region was large, inside the frame, not covered by
another worker, and not too dark, bright or blurred. It does NOT prove that a bare
head or uncovered torso is visible; machinery occlusion is not detected here.
"""

from __future__ import annotations

import cv2
import numpy as np

from .association import area, intersection, region_for


def assess(kind: str, person, person_conf: float, others: list, image: np.ndarray, obs_cfg, assoc_cfg):
    """Return (observable, reasons, metrics) for the head ('helmet') or torso ('vest') region.

    image is the BGR frame (only the region crop is converted to gray) or a gray frame.
    others: boxes of every other person detection, tracked or not.
    """
    reasons = []
    metrics = {}
    height, width = image.shape[:2]
    x1, y1, x2, y2 = person
    pw, ph = x2 - x1, y2 - y1
    region = region_for(kind, person, assoc_cfg)
    rw, rh = region[2] - region[0], region[3] - region[1]
    margin = obs_cfg["border_margin_px"]

    if person_conf < obs_cfg["min_person_conf"]:
        reasons.append("low person confidence")
    if ph < obs_cfg["min_person_height_px"]:
        reasons.append("person too small")
    if pw > 0 and ph / pw < obs_cfg["min_aspect_ratio"]:
        reasons.append("partial or truncated person box")
    if x1 <= margin or y1 <= margin or x2 >= width - margin or y2 >= height - margin:
        reasons.append("person box touches image border")
    min_side = obs_cfg["min_head_region_px"] if kind == "helmet" else obs_cfg["min_torso_region_px"]
    if min(rw, rh) < min_side:
        reasons.append("region too small")
    region_area = area(region)
    inside = intersection(region, (0, 0, width, height)) / region_area if region_area else 0.0
    if inside < 0.95:
        reasons.append("region outside image")
    covered = sum(intersection(region, other) for other in others) / region_area if region_area else 1.0
    metrics["other_person_overlap"] = round(min(1.0, covered), 3)
    if covered > obs_cfg["max_other_person_overlap"]:
        reasons.append("occluded by another worker")
    if reasons:
        return False, reasons, metrics

    crop = image[int(max(0, region[1])):int(min(height, region[3])), int(max(0, region[0])):int(min(width, region[2]))]
    if crop.size == 0:
        return False, ["empty region"], metrics
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mean = float(crop.mean())
    dark = float((crop <= obs_cfg["dark_pixel_level"]).mean())
    bright = float((crop >= obs_cfg["bright_pixel_level"]).mean())
    sharp = float(cv2.Laplacian(crop, cv2.CV_64F).var())
    metrics.update(mean=round(mean, 1), dark=round(dark, 3), bright=round(bright, 3), lap_var=round(sharp, 1))
    if mean < obs_cfg["min_mean_intensity"] or dark > obs_cfg["max_dark_fraction"]:
        reasons.append("too dark")
    if mean > obs_cfg["max_mean_intensity"] or bright > obs_cfg["max_bright_fraction"]:
        reasons.append("overexposed / glare")
    if sharp < obs_cfg["min_laplacian_var"]:
        reasons.append("blurred / low detail")
    return not reasons, reasons, metrics
