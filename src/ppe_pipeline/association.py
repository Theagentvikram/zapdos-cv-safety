"""Spatial association of SH17 helmet / vest boxes with tracked person boxes.

Boxes are (x1, y1, x2, y2) in pixels. Association is one-to-one per PPE kind:
a PPE box goes to at most one person and a person gets at most one box.
"""

from __future__ import annotations

from dataclasses import dataclass

ASSOCIATED = "associated"
AMBIGUOUS = "ambiguous"
NONE = "none"


@dataclass
class PPEMatch:
    status: str = NONE
    ppe_index: int | None = None
    conf: float | None = None
    score: float | None = None
    near_miss: bool = False


def area(box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection(a, b) -> float:
    return area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def head_region(person, cfg) -> tuple:
    x1, y1, x2, y2 = person
    return (x1, y1, x2, y1 + cfg["head_region_fraction"] * (y2 - y1))


def torso_region(person, cfg) -> tuple:
    x1, y1, x2, y2 = person
    h = y2 - y1
    return (x1, y1 + cfg["torso_start_fraction"] * h, x2, y1 + cfg["torso_end_fraction"] * h)


def region_for(kind: str, person, cfg) -> tuple:
    return head_region(person, cfg) if kind == "helmet" else torso_region(person, cfg)


def _pair_score(kind: str, person, ppe, cfg) -> tuple[float | None, bool]:
    """Return (score or None when invalid, near_miss)."""
    region = region_for(kind, person, cfg)
    ppe_area = area(ppe)
    if ppe_area <= 0:
        return None, False
    overlap = intersection(ppe, region) / ppe_area
    near_miss = overlap >= cfg["near_miss_overlap_fraction"]
    # Reject tiny/implausible PPE as positive evidence, but keep it as an
    # uncertainty blocker: rejecting a helmet must not manufacture a bare head.
    pw, ph = person[2] - person[0], person[3] - person[1]
    if (ppe[2] - ppe[0] < cfg.get(f"{kind}_min_width_fraction", 0) * pw or
            ppe[3] - ppe[1] < cfg.get(f"{kind}_min_height_fraction", 0) * ph):
        return None, near_miss
    cx, cy = (ppe[0] + ppe[2]) / 2, (ppe[1] + ppe[3]) / 2
    if not (region[0] <= cx <= region[2] and region[1] <= cy <= region[3]):
        return None, near_miss
    if overlap < cfg["min_ppe_overlap_fraction"]:
        return None, True
    pw, ph = person[2] - person[0], person[3] - person[1]
    if (ppe[2] - ppe[0]) > cfg[f"{kind}_max_width_fraction"] * pw:
        return None, True
    if (ppe[3] - ppe[1]) > cfg[f"{kind}_max_height_fraction"] * ph:
        return None, True
    # Closeness of the PPE centre to the region centre, normalised by the region half-diagonal.
    rx, ry = (region[0] + region[2]) / 2, (region[1] + region[3]) / 2
    half_diag = max(1e-6, (((region[2] - region[0]) ** 2 + (region[3] - region[1]) ** 2) ** 0.5) / 2)
    closeness = max(0.0, 1.0 - (((cx - rx) ** 2 + (cy - ry) ** 2) ** 0.5) / half_diag)
    return 0.5 * overlap + 0.5 * closeness, True


def associate(kind: str, persons: list, ppe_boxes: list, ppe_confs: list, cfg) -> list[PPEMatch]:
    """Assign PPE boxes of one kind to persons.

    persons: list of person boxes. ppe_boxes/ppe_confs: detections of `kind`.
    A PPE box whose best and second-best persons score within `ambiguity_margin`
    is not assigned; both persons get status AMBIGUOUS for that frame. A box goes only to its
    best-scoring person; if that person already holds a better box, it stays unassigned.
    `persons` should include untracked person detections so they can compete for PPE.
    A PPE box outside every person's region (e.g. a helmet on machinery) is never assigned.
    """
    matches = [PPEMatch() for _ in persons]
    candidates = {}  # ppe index -> list of (score, person index)
    for j, ppe in enumerate(ppe_boxes):
        for i, person in enumerate(persons):
            score, near_miss = _pair_score(kind, person, ppe, cfg)
            if near_miss:
                matches[i].near_miss = True
            if score is not None and ppe_confs[j] >= cfg.get(f"{kind}_min_conf", 0):
                candidates.setdefault(j, []).append((score, i))

    ambiguous_ppe = set()
    pairs = []
    for j, options in candidates.items():
        options.sort(reverse=True)
        if len(options) > 1 and options[0][0] - options[1][0] < cfg["ambiguity_margin"]:
            ambiguous_ppe.add(j)
            best = options[0][0]
            for score, i in options:
                if best - score < cfg["ambiguity_margin"]:
                    matches[i].status = AMBIGUOUS
            continue
        # Only the best person may take this box; never fall back to a worse-fitting neighbour.
        pairs.append((options[0][0], j, options[0][1]))

    taken_ppe, taken_person = set(), set()
    for score, j, i in sorted(pairs, key=lambda p: (-p[0], p[1], p[2])):
        if j in taken_ppe or i in taken_person:
            continue
        taken_ppe.add(j)
        taken_person.add(i)
        matches[i] = PPEMatch(ASSOCIATED, j, float(ppe_confs[j]), round(score, 4), matches[i].near_miss)
    return matches
