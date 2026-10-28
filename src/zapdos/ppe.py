"""Per-frame PPE reading for each tracked person, in two steps.

measure() collects the per-person evidence: the head box (detected or estimated), the classifier's
p(bare) for it, the assigned vest confidence, which frame edges cut the person, and the
detector-method helmet reading. Box assignment, the edge margin and the detector-method rule are
applied here. decide() turns that evidence into readings with the classifier threshold, the head
gates and the vest rule. The split lets evaluation cache measure() once per clip and re-run decide()
with other settings in seconds.

A reading is (value, evidence_conf): value True = violation, False = compliant, None = can't tell.
"Can't tell" never counts as a violation; that is what keeps tiny or occluded heads from alerting.

Helmet state comes from one of two methods (config ppe.helmet_method):
  classifier - head crop from the native frame -> HelmetClassifier p(bare). Default.
  detector   - SH17 helmet box over the SH17 head box. Kept as the measured baseline: on SteelBench
               it calls 47% of helmeted workers bare, because SH17 rarely fires on 17-42 px helmets.
"""
from .helmet import head_from_person

CANT_TELL = (None, 0.0)
HEAD_BAND_TOP = -0.1    # a head may poke above a tight person box by this share of its height
HELMET_BAND_TOP = -0.2  # a helmet box reaches higher than the head under it
ABSENT_VEST_CONF = 0.5  # no vest detected is weaker evidence than a detected vest


def _inter(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(w, 0) * max(h, 0)


def _area(a):
    return max(a[2] - a[0], 0) * max(a[3] - a[1], 0)


def _band_distance(cx, cy, person, y_band):
    """How far the point (cx, cy) is from the centre of the person's band, in person widths and
    heights; None when the point is outside the person box or the band."""
    x1, y1, x2, y2 = person
    h = y2 - y1
    lo, hi = y1 + y_band[0] * h, y1 + y_band[1] * h
    if not (x1 <= cx <= x2 and lo <= cy <= hi):
        return None
    return abs(cy - (lo + hi) / 2) / h + abs(cx - (x1 + x2) / 2) / (x2 - x1)


def _assign(parts, persons, y_band):
    """Give each part box to at most one person: the one whose box holds the part's centre inside
    the vertical band `y_band` (fractions of person height), preferring the nearest band centre."""
    owner = {}
    for p in parts:
        cx, cy = (p[0] + p[2]) / 2, (p[1] + p[3]) / 2
        dists = [(d, i) for i, person in enumerate(persons)
                 if (d := _band_distance(cx, cy, person, y_band)) is not None]
        if dists:
            owner.setdefault(min(dists, key=lambda di: di[0])[1], []).append(p)
    return owner


def head_candidates(heads, helmets, persons, head_region):
    """The detected head and helmet boxes assigned to each person: ({i: [head]}, {i: [helmet]})."""
    return (_assign(heads, persons, (HEAD_BAND_TOP, head_region)),
            _assign(helmets, persons, (HELMET_BAND_TOP, head_region)))


def cut_edges(box, frame_shape, margin):
    """Which frame edges the box touches: a subset of "ltrb". A person cut by the side of the frame is
    a sliver whose PPE can't be judged; one cut at the top has no visible head."""
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box[:4]
    return "".join(e for e, hit in zip("ltrb", (x1 <= margin, y1 <= margin, x2 >= w - margin, y2 >= h - margin))
                   if hit)


def _head_box(heads, helmets, person_box):
    """The most confident detected head or helmet box, else the head position estimated from the person
    box. Returns (box, source) with source "head", "helmet" or "est"."""
    cands = [(h, "head") for h in heads] + [(h, "helmet") for h in helmets]
    if not cands:
        return head_from_person(person_box), "est"
    hb, src = max(cands, key=lambda c: c[0][4])
    return tuple(float(v) for v in hb[:4]), src


def measure(person_boxes, det, cfg, frame=None, helmet_clf=None):
    """Evidence per person: {"head", "head_src" (head|helmet|est), "p_bare", "rule_helmet", "vest",
    "cut"}. p_bare is None without a classifier."""
    heads, helmets = head_candidates(det.heads, det.helmets, person_boxes, cfg["head_region"])
    vests = _assign(det.vests, person_boxes, cfg["vest"]["torso"]) if cfg["vest"]["enabled"] else {}

    out = []
    for i, box in enumerate(person_boxes):
        hb, src = _head_box(heads.get(i, []), helmets.get(i, []), box)
        out.append({"head": hb, "head_src": src, "p_bare": None,
                    "rule_helmet": _helmet(heads.get(i, []), helmets.get(i, []), cfg),
                    "vest": float(max((v[4] for v in vests.get(i, [])), default=0.0)),
                    "cut": cut_edges(box, frame.shape, cfg["edge_margin_px"]) if frame is not None else ""})
    if helmet_clf is not None and out:
        for m, p in zip(out, helmet_clf.p_bare(frame, [m["head"] for m in out])):
            m["p_bare"] = float(p)
    return out


def decide(meas, person_boxes, cfg, threshold=None):
    """Readings {"no_helmet": (v, c), "no_vest": (v, c)} per person from measure()'s output.
    threshold: p(bare) cutoff for the classifier (see helmet_threshold)."""
    readings = []
    for m, box in zip(meas, person_boxes):
        r = {"no_helmet": _helmet_reading(m, cfg, threshold)}
        if cfg["vest"]["enabled"]:
            side_cut = "l" in m.get("cut", "") or "r" in m.get("cut", "")
            r["no_vest"] = CANT_TELL if side_cut else _vest(box, m["vest"], cfg["vest"])
        readings.append(r)
    return readings


def helmet_threshold(cfg, helmet_clf):
    """The p(bare) cutoff: ppe.classifier.threshold if set, else the one shipped with the weights."""
    return cfg.get("classifier", {}).get("threshold") or (helmet_clf.threshold if helmet_clf else None)


def read_ppe(person_boxes, det, cfg, frame=None, helmet_clf=None):
    """measure + decide. Returns (readings, head_boxes)."""
    meas = measure(person_boxes, det, cfg, frame, helmet_clf)
    return decide(meas, person_boxes, cfg, helmet_threshold(cfg, helmet_clf)), [m["head"] for m in meas]


def _helmet_reading(m, cfg, threshold):
    if any(e in m.get("cut", "") for e in "lrt"):
        return CANT_TELL  # a sliver at the side, or the head is above the frame
    if cfg["helmet_method"] == "classifier" and m["p_bare"] is not None:
        return _classifier_helmet(m, cfg.get("classifier", {}), threshold)
    return m["rule_helmet"]


def _classifier_helmet(m, ccfg, threshold):
    hb = m["head"]
    # an estimated head position (no head/helmet box) is too often wrong to call "bare":
    # bent-over workers, and false person boxes on machinery
    estimated = ccfg["require_head_det"] and m["head_src"] == "est"
    if hb[3] - hb[1] < ccfg["min_head_px"] or estimated:
        return CANT_TELL
    p = m["p_bare"]
    bare = p >= threshold
    return bare, (p if bare else 1 - p)


def _helmet(heads, helmets, cfg):
    """The detector-method reading: a helmet box over the most confident head box."""
    if not heads:
        # helmet visible but no head box: compliant; nothing visible: can't tell
        return (False, float(max(h[4] for h in helmets))) if helmets else CANT_TELL
    head = max(heads, key=lambda h: h[4])
    hh = head[3] - head[1]
    if hh < cfg["min_head_px"]:
        return CANT_TELL
    grown = (head[0], head[1] - cfg["helmet_expand"] * hh, head[2], head[3])
    for hm in helmets:
        if _inter(hm, grown) >= cfg["helmet_overlap"] * _area(hm):
            return False, float(hm[4])
    return True, float(head[4])


def _vest(box, vest_conf, cfg):
    if vest_conf > 0:
        return False, vest_conf
    if box[3] - box[1] < cfg["min_person_px"]:
        return CANT_TELL
    return True, ABSENT_VEST_CONF
