"""Turn per-frame readings into deduplicated alert events.

Each (track, type, zone) key keeps a sliding window of readings. An alert fires once when, within
the window, enough frames are readable, the violating share reaches the threshold, and the current
violating run has lasted min_duration_s. The episode stays open (end_ts keeps moving) until the
violation has been unseen for end_after_s. A new episode for the same person and type inside the
cooldown re-attaches to the old event instead of alerting again.

A new track inherits a recent event of the same type, instead of alerting, when it is plausibly the
same person:
  (a) its box lies inside the box now feeding the event (a second detection of part of the same
      body); or
  (b) the event's track is lost, the two tracks were never seen side by side in one frame (that is
      two people), and the new box either overlaps where the lost one was (any gap within the
      cooldown: someone who stood still behind an occluder) or appears within merge_radius person
      heights of it within merge_window_s (an ID switch while walking).
"""
import math
from collections import deque
from dataclasses import asdict, dataclass, field

NESTED_MAX_GAP_S = 1.0   # (a) applies only to an event fed within this long: its person is still there
REACQUIRE_OVERLAP = 0.5  # (b) a new box overlapping the lost box by this much (of the smaller) = same spot


@dataclass
class Event:
    event_id: str
    camera_id: str
    type: str
    track_id: int
    zone_id: str | None
    start_ts: float
    first_frame: int
    alert_ts: float
    alert_frame: int
    end_ts: float
    confidence: float        # vote x mean confidence of the violating readings in the window
    vote: float              # share of readable readings in the window that were violating
    bbox: list
    snapshot: str = ""
    clip: str = ""
    merged_track_ids: list = field(default_factory=list)
    note: str = None         # camera-level events: what was measured
    owner: int = -1          # track currently feeding this event (internal)
    last_bbox: list = None   # (internal)

    @property
    def delay_s(self):
        return round(self.alert_ts - self.start_ts, 3)

    def to_json(self):
        d = asdict(self)
        d.pop("owner"), d.pop("last_bbox")
        if d["note"] is None:
            d.pop("note")
        d["delay_s"] = self.delay_s
        for k in ("start_ts", "alert_ts", "end_ts"):
            d[k] = round(d[k], 3)
        return d


@dataclass
class Observation:
    track_id: int
    box: list
    readings: dict  # (type, zone_id) -> (value, conf)
    hold: frozenset = frozenset()  # reading keys that gather evidence but may not alert yet


@dataclass
class _Frame:
    idx: int
    ts: float
    visible: set  # track ids detected in this frame


class _KeyState:
    def __init__(self):
        self.window = deque()   # (ts, frame_idx, value, conf)
        self.run_start = None   # (ts, frame_idx) of the first violating frame of the current run
        self.last_true = None
        self.event = None

    def add(self, now, value, conf, window_s):
        """Slide the window to `now` and extend or end the current violating run."""
        self.window.append((now.ts, now.idx, value, conf))
        while self.window[0][0] <= now.ts - window_s:
            self.window.popleft()
        if value:
            if self.run_start is None:
                self.run_start = (now.ts, now.idx)
            self.last_true = now.ts
        elif not any(w[2] for w in self.window):
            self.run_start = None


class EventEngine:
    def __init__(self, cfg, camera_id, fps):
        self.cfg = cfg
        self.camera_id = camera_id
        self.fps = fps
        self.keys = {}
        self.events = []
        self.apart = set()  # pairs of track ids seen in one frame with neither box inside the other
        self._params = {}

    def params(self, etype):
        """The rule settings for one event type: its own section over `default`."""
        if etype not in self._params:
            p = {**self.cfg["default"], **self.cfg.get(etype, {})}
            p["min_observed"] = math.ceil(p["min_observed_s"] * self.fps - 1e-6)  # frames at this fps
            self._params[etype] = p
        return self._params[etype]

    def step(self, frame_idx, ts, observations):
        """Feed one processed frame; returns the events that alerted on this frame."""
        now = _Frame(frame_idx, ts, {o.track_id for o in observations})
        self._note_apart(observations)
        new = []
        for o in observations:
            for key, (value, conf) in o.readings.items():
                ev = self._update(o, key, value, conf, now)
                if ev is not None:
                    new.append(ev)
        self._end_episodes(ts)
        return new

    def camera_event(self, etype, start_ts, first_frame, ts, frame_idx, frame_size, note):
        """An event about the camera rather than a person (camera_moved): no track, whole-frame box."""
        ev = Event(event_id=self._next_id(), camera_id=self.camera_id,
                   type=etype, track_id=-1, zone_id=None, start_ts=start_ts, first_frame=first_frame,
                   alert_ts=ts, alert_frame=frame_idx, end_ts=ts, confidence=1.0, vote=1.0,
                   bbox=[0.0, 0.0, float(frame_size[0]), float(frame_size[1])], note=note)
        self.events.append(ev)
        return ev

    def _next_id(self):
        return f"{self.camera_id}-{len(self.events) + 1:03d}"

    def _note_apart(self, observations):
        nested = self.cfg["default"]["nested_overlap"]
        for i, a in enumerate(observations):
            for b in observations[i + 1:]:
                if _nested(a.box, b.box) < nested:
                    self.apart.add(frozenset((a.track_id, b.track_id)))

    def _update(self, o, key, value, conf, now):
        """Add one reading to its (track, type, zone) state; returns the event if it alerts now."""
        st = self.keys.setdefault((o.track_id, *key), _KeyState())
        st.add(now, value, conf, self.params(key[0])["window_s"])
        if value and st.event:
            st.event.end_ts = now.ts
            st.event.last_bbox = list(o.box)
        if st.event is None and st.run_start is not None and key not in o.hold:
            return self._maybe_alert(st, o, key, now)
        return None

    def _end_episodes(self, ts):
        for (_, etype, _), st in self.keys.items():
            if st.event and ts - st.last_true >= self.params(etype)["end_after_s"]:
                st.event, st.run_start = None, None

    def _maybe_alert(self, st, o, key, now):
        p = self.params(key[0])
        readable = [(w[2], w[3]) for w in st.window if w[2] is not None]
        if len(readable) < p["min_observed"] or now.ts - st.run_start[0] < p["min_duration_s"]:
            return None
        hits = [c for v, c in readable if v]
        vote = len(hits) / len(readable)
        if vote < p["threshold"]:
            return None

        old = self._reusable(o, key, now)
        if old is not None:
            old.owner, old.end_ts, old.last_bbox = o.track_id, now.ts, list(o.box)
            st.event = old
            return None

        ev = Event(
            event_id=self._next_id(),
            camera_id=self.camera_id, type=key[0], track_id=o.track_id, zone_id=key[1],
            start_ts=st.run_start[0], first_frame=st.run_start[1],
            alert_ts=now.ts, alert_frame=now.idx, end_ts=now.ts,
            confidence=round(vote * sum(hits) / len(hits), 3), vote=round(vote, 3),
            bbox=[round(float(v), 1) for v in o.box], owner=o.track_id, last_bbox=list(o.box),
        )
        self.events.append(ev)
        st.event = ev
        return ev

    def _reusable(self, o, key, now):
        """The recent event of this type and zone that `o` continues, or None to alert."""
        cooldown_s = self.params(key[0])["cooldown_s"]
        for ev in reversed(self.events):
            if (ev.type, ev.zone_id) != key or now.ts - ev.end_ts > cooldown_s:
                continue
            if o.track_id == ev.track_id or o.track_id in ev.merged_track_ids:
                return ev  # same person inside cooldown: extend, don't re-alert
            if self._same_person(o, ev, now):
                ev.merged_track_ids.append(o.track_id)
                return ev
        return None

    def _same_person(self, o, ev, now):
        """Is new track `o` the person who fed `ev` (cases (a) and (b) in the module docstring)?"""
        p = self.params(ev.type)
        overlap = _nested(o.box, ev.last_bbox)
        gap = now.ts - ev.end_ts
        if ev.owner in now.visible:
            # the event's person is on screen: only a box nested in theirs (a second detection of
            # part of the same body) is them; someone standing next to them is not
            return overlap >= p["nested_overlap"] and gap <= NESTED_MAX_GAP_S
        if frozenset((o.track_id, ev.owner)) in self.apart:
            return False  # seen side by side earlier (the owner just dropped a frame): two people
        if overlap >= REACQUIRE_OVERLAP:
            return True  # re-acquired where they were lost (stood still behind an occluder)
        height = o.box[3] - o.box[1]
        return gap <= p["merge_window_s"] and _centre_distance(o.box, ev.last_bbox) <= p["merge_radius"] * height


def _nested(a, b):
    """Intersection over the smaller box's area: 1.0 when one box lies inside the other."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return w * h / small if small > 0 else 0.0


def _centre_distance(a, b):
    dx = (a[0] + a[2] - b[0] - b[2]) / 2
    dy = (a[1] + a[3] - b[1] - b[3]) / 2
    return (dx * dx + dy * dy) ** 0.5
