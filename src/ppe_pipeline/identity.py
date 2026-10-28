"""Worker identity on top of ByteTrack track IDs.

* Every ByteTrack track starts as a candidate. It becomes a worker only after
  `min_confirmed_observations` genuine detections within `confirmation_window_seconds`
  whose median confidence and box height pass the configured minimums.
* Confirmed workers get sequential IDs (1, 2, 3, ...) unique within the video.
* Jump guard: if ByteTrack matches a track to a box that overlaps the identity's previous box by
  less than `jump_min_iou` within `jump_window_seconds` (or, after a longer loss, also differs in
  height by more than `jump_max_size_ratio`), the old identity ends and a new candidate starts,
  so PPE history never moves to another person.
* Optional stitching: a new candidate inherits a recently lost confirmed worker's identity only
  when exactly one such worker fits (time gap, box overlap, size ratio). The caller must reset
  that worker's PPE evidence; identity continues, PPE evidence does not.
"""

from __future__ import annotations

import statistics
import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass, field

EPS = 1e-6


def appearance_descriptor(image, box):
    """Small torso colour histogram; useful only to veto incompatible stitches."""
    x1,y1,x2,y2=map(float,box)
    w,h=x2-x1,y2-y1
    if w<24 or h<80:
        return None
    left,right=max(0,int(x1+.2*w)),min(image.shape[1],int(x2-.2*w))
    top,bottom=max(0,int(y1+.25*h)),min(image.shape[0],int(y1+.65*h))
    crop=image[top:bottom,left:right]
    if crop.size==0:return None
    crop=cv2.resize(crop,(24,32))
    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    hist=cv2.calcHist([hsv],[0,1,2],None,[12,4,4],[0,180,0,256,0,256]).reshape(-1)
    return hist/max(float(hist.sum()),1e-6)


def appearance_similarity(a,b):
    return float(np.sqrt(np.asarray(a)*np.asarray(b)).sum())


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Identity:
    key: int                       # stable internal key, independent of tracker ids
    tracker_ids: list
    first_t: float
    last_t: float
    last_box: tuple
    worker_id: int | None = None   # sequential display id, set on confirmation
    confirmed_t: float | None = None
    window: deque = field(default_factory=deque)  # (t, conf, height)
    stitched_from_t: list = field(default_factory=list)
    velocity: tuple = (0.0, 0.0)
    appearance: object = None

    @property
    def confirmed(self) -> bool:
        return self.worker_id is not None


@dataclass
class Assignment:
    key: int
    tracker_id: int
    worker_id: int | None
    newly_confirmed: bool
    stitched: bool


class WorkerRegistry:
    def __init__(self, tracking_cfg: dict):
        self.cfg = tracking_cfg
        self.by_tracker: dict[int, Identity] = {}
        self.identities: dict[int, Identity] = {}
        self.recent: dict[int, Identity] = {}
        self.next_worker_id = 1
        self.next_key = 1
        self.stitches: list[dict] = []
        self.splits: list[dict] = []

    def _jumped(self, ident: Identity, t: float, box) -> bool:
        cfg = self.cfg
        overlap = iou(box, ident.last_box)
        dt = t - ident.last_t
        # At sampled rates normal walking may move a narrow box > one third of
        # its width. Compare against the motion prediction as well as the old box.
        if 0 < dt <= cfg["jump_window_seconds"] + EPS:
            dx, dy = ident.velocity[0] * dt, ident.velocity[1] * dt
            predicted = (ident.last_box[0] + dx, ident.last_box[1] + dy,
                         ident.last_box[2] + dx, ident.last_box[3] + dy)
            overlap = max(overlap, iou(box, predicted))
        if overlap >= cfg["jump_min_iou"]:
            return False
        if t - ident.last_t <= cfg["jump_window_seconds"] + EPS:
            return True
        # After a longer loss the person may have moved; only a size change as well marks a different person.
        h, old_h = box[3] - box[1], ident.last_box[3] - ident.last_box[1]
        return max(h, old_h) / max(1e-6, min(h, old_h)) > cfg["jump_max_size_ratio"]

    def _new_identity(self, tid: int, t: float, box) -> Identity:
        ident = Identity(key=self.next_key, tracker_ids=[tid], first_t=t, last_t=t, last_box=tuple(box))
        self.next_key += 1
        self.identities[ident.key] = ident
        self.recent[ident.key] = ident
        self.by_tracker[tid] = ident
        return ident

    def _stitch_target(self, t: float, box, observed_keys: set, competing_boxes=(), appearance=None):
        cfg = self.cfg
        max_gap = cfg.get("stitch_max_gap_seconds") or 0
        if max_gap <= 0:
            return None
        h = box[3] - box[1]
        fits = []
        for ident in self.recent.values():
            if not ident.confirmed or ident.key in observed_keys:
                continue
            gap = t - ident.last_t
            if gap <= EPS or gap > max_gap + EPS:
                continue
            old_h = ident.last_box[3] - ident.last_box[1]
            ratio = max(h, old_h) / max(1e-6, min(h, old_h))
            if (appearance is not None and ident.appearance is not None and
                    appearance_similarity(appearance,ident.appearance)<cfg.get("stitch_min_appearance_similarity",.65)):
                continue
            if iou(box, ident.last_box) >= cfg["stitch_min_iou"] and ratio <= cfg["stitch_max_size_ratio"]:
                fits.append(ident)
        if len(fits) != 1:
            return None
        target = fits[0]
        # Reciprocal uniqueness: a lost worker cannot be claimed by whichever of
        # two plausible new detections happens to be processed first.
        compatible = 0
        old_h = target.last_box[3] - target.last_box[1]
        for candidate in competing_boxes:
            h = candidate[3] - candidate[1]
            ratio = max(h, old_h) / max(1e-6, min(h, old_h))
            compatible += iou(candidate, target.last_box) >= cfg["stitch_min_iou"] and ratio <= cfg["stitch_max_size_ratio"]
        return target if compatible <= 1 else None

    def update(self, t: float, matched: list[tuple[int, tuple, float]], confirmation_eligible=None, appearances=None) -> list[Assignment]:
        """matched: (tracker_id, detection box, detection conf) for tracks matched to a detection this frame.

        Returns one Assignment per input item, in input order. Known tracks are processed before new
        ones so that a stitch can never take an identity that is still observed this frame.
        """
        cfg = self.cfg
        gap = cfg.get("stitch_max_gap_seconds") or 0
        self.recent = {key: ident for key, ident in self.recent.items() if t - ident.last_t <= gap + EPS}
        out: list = [None] * len(matched)
        order = sorted(range(len(matched)), key=lambda n: matched[n][0] not in self.by_tracker)
        observed_keys = {self.by_tracker[tid].key for tid, _, _ in matched if tid in self.by_tracker}
        competing = [box for tid, box, _ in matched if tid not in self.by_tracker]
        processed = set()
        for n in order:
            tid, box, conf = matched[n]
            appearance=appearances[n] if appearances is not None else None
            stitched = False
            ident = self.by_tracker.get(tid)
            if ident is not None and ident.key in processed:
                # Two tracker ids claim one identity (old id recovered after a stitch): split it off.
                ident = self._new_identity(tid, t, box)
            elif ident is not None and self._jumped(ident, t, box):
                # ByteTrack moved this id onto a box that does not fit the previous one (likely another
                # person). End the old identity so no PPE history transfers; start a new candidate.
                self.splits.append({"t": round(t, 3), "tracker_id": tid, "old_worker_id": ident.worker_id,
                                    "old_box": [round(v) for v in ident.last_box], "new_box": [round(v) for v in box]})
                ident = self._new_identity(tid, t, box)
            if ident is None:
                target = self._stitch_target(t, box, observed_keys, competing, appearance)
                if target is not None:
                    ident, stitched = target, True
                    ident.tracker_ids.append(tid)
                    ident.stitched_from_t.append(round(t, 3))
                    self.stitches.append({"t": round(t, 3), "worker_id": ident.worker_id, "new_tracker_id": tid,
                                          "gap_s": round(t - ident.last_t, 3),
                                          "iou": round(iou(box, ident.last_box), 3)})
                    self.by_tracker[tid] = ident
                else:
                    ident = self._new_identity(tid, t, box)
                observed_keys.add(ident.key)
            processed.add(ident.key)
            dt = t - ident.last_t
            if dt > EPS:
                vx = ((box[0] + box[2]) - (ident.last_box[0] + ident.last_box[2])) / (2 * dt)
                vy = ((box[1] + box[3]) - (ident.last_box[1] + ident.last_box[3])) / (2 * dt)
                ident.velocity = ((ident.velocity[0] + vx) / 2, (ident.velocity[1] + vy) / 2)
            ident.last_t, ident.last_box = t, tuple(box)
            if appearance is not None:
                ident.appearance=appearance
            self.recent[ident.key] = ident
            if confirmation_eligible is None or confirmation_eligible[n]:
                ident.window.append((t, float(conf), float(box[3] - box[1])))
            while ident.window and ident.window[0][0] < t - cfg["confirmation_window_seconds"] - EPS:
                ident.window.popleft()
            newly = False
            if not ident.confirmed and len(ident.window) >= cfg["min_confirmed_observations"]:
                confs = [c for _, c, _ in ident.window]
                heights = [h for _, _, h in ident.window]
                if (statistics.median(confs) >= cfg["min_confirm_median_conf"]
                        and statistics.median(heights) >= cfg["min_confirm_height_px"]):
                    ident.worker_id, ident.confirmed_t, newly = self.next_worker_id, t, True
                    self.next_worker_id += 1
            out[n] = Assignment(ident.key, tid, ident.worker_id, newly, stitched)
        return out

    def forget_tracker(self, tracker_id: int) -> None:
        """ByteTrack removed this track; a later reuse of the identity can only come via stitching."""
        self.by_tracker.pop(tracker_id, None)
