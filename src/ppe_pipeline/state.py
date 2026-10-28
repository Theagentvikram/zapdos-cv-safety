"""Per-worker temporal PPE state machines and alert cooldowns.

All times are video timestamps in seconds. Nothing here reads images, so the
rules can be unit-tested with synthetic observation sequences.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

EPS = 1e-6
UNKNOWN = "UNKNOWN"

EMITTED = "emitted"
SUPPRESSED_COOLDOWN = "suppressed_cooldown"
SUPPRESSED_UNVALIDATED = "suppressed_camera_unvalidated"
SUPPRESSED_UNCONFIRMED = "suppressed_unconfirmed_track"


@dataclass
class Observation:
    """One genuine detector observation of a worker's head or torso region.

    eligible_negative must only be True when the region passed the observability gate,
    no PPE box of this kind was associated, ambiguous, or near the region.
    """

    positive: bool = False
    conf: float | None = None
    eligible_negative: bool = False


def _prune(times: deque, oldest: float) -> None:
    while times and times[0] < oldest - EPS:
        times.popleft()


class PPEStateMachine:
    """UNKNOWN / <KIND>_ON / <KIND>_ON_HELD / SUSPECTED_NO_<KIND> for one worker."""

    def __init__(self, kind: str, ppe_cfg: dict, absence_allowed: bool):
        tag = kind.upper()
        self.kind = kind
        self.ON = f"{tag}_ON"
        self.HELD = f"{tag}_ON_HELD"
        self.SUSPECTED = f"SUSPECTED_NO_{tag}"
        self.cfg = ppe_cfg
        self.absence_allowed = absence_allowed
        self.state = UNKNOWN
        self.reason = "new worker: no evidence yet"
        self.positive_times: deque = deque()
        self.negative_times: deque = deque()
        self.observation_times: deque = deque()
        self.last_positive_t: float | None = None
        self.last_positive_conf: float | None = None
        self.last_confirmed_t: float | None = None
        self.last_negative_t: float | None = None
        self.absence_evidence = False
        self.episode_active = False
        self.episode_start_t: float | None = None
        self.episodes: list[list] = []  # [start, end or None, kind of episode]
        self.evidence_start_t: float | None = None  # set when evidence is reset (identity stitched)
        self.negative_start_t: float | None = None

    @property
    def eligible_negative_count(self) -> int:
        return len(self.negative_times)

    def reset_evidence(self, t: float, reason: str) -> None:
        """Drop all PPE evidence; the absence clock restarts at t. Episode bookkeeping is closed."""
        self.positive_times.clear()
        self.negative_times.clear()
        self.observation_times.clear()
        self.last_positive_t = None
        self.last_positive_conf = None
        self.last_confirmed_t = None
        self.last_negative_t = None
        self.negative_start_t = None
        self.absence_evidence = False
        self.state, self.reason = UNKNOWN, reason
        self.episode_start_t = None
        self.evidence_start_t = t
        if self.episode_active:
            self.episodes[-1][1] = round(t, 3)
            self.episode_active = False

    def _absence_check(self, t: float, track_start: float) -> tuple[bool, str]:
        cfg = self.cfg
        need = cfg["suspected_absence_seconds"]
        start = max(track_start, self.evidence_start_t) if self.evidence_start_t is not None else track_start
        age = t - start
        since = t - (self.last_positive_t if self.last_positive_t is not None else start)
        n = len(self.negative_times)
        total = len(self.observation_times)
        frac = n / total if total else 0.0
        if age + EPS < need:
            return False, f"track age {age:.2f}s < {need}s"
        if since + EPS < need:
            return False, f"{self.kind} seen {since:.2f}s ago"
        duration = t - self.negative_start_t if self.negative_start_t is not None else 0.0
        if duration + EPS < need:
            return False, f"eligible negative evidence spans {duration:.2f}s < {need}s"
        if n < cfg["min_eligible_negative_observations"]:
            return False, f"only {n}/{cfg['min_eligible_negative_observations']} eligible observations"
        if frac + EPS < cfg["min_eligible_fraction"]:
            return False, f"eligible fraction {frac:.0%} < {cfg['min_eligible_fraction']:.0%}"
        return True, f"no {self.kind} for {since:.1f}s with {n} eligible observations ({frac:.0%})"

    def step(self, t: float, track_start: float, obs: Observation | None) -> dict:
        """Advance to time t. obs is None when the worker was not genuinely detected this frame."""
        cfg = self.cfg
        ended_early = False
        # Unknown/lost gaps cannot count as sustained negative evidence.
        if self.last_negative_t is not None and t - self.last_negative_t > cfg["stale_negative_seconds"] + EPS:
            self.negative_start_t = None
            self.negative_times.clear()
            if self.episode_active:
                self.episodes[-1][1] = round(self.last_negative_t + cfg["stale_negative_seconds"], 3)
                self.episode_active = False
                ended_early = True
                if self.state == self.SUSPECTED:
                    self.state, self.reason = UNKNOWN, "absence episode expired: visibility gap"
        if obs is not None:
            self.observation_times.append(t)
            if obs.positive:
                self.positive_times.append(t)
                self.last_positive_t = t
                self.last_positive_conf = obs.conf
                self.negative_times.clear()
                self.negative_start_t = None
            elif obs.eligible_negative:
                if self.negative_start_t is None:
                    self.negative_start_t = t
                self.negative_times.append(t)
                self.last_negative_t = t
        _prune(self.positive_times, t - cfg["positive_window_seconds"])
        _prune(self.negative_times, t - cfg["suspected_absence_seconds"])
        _prune(self.observation_times, t - cfg["suspected_absence_seconds"])

        if obs is not None and obs.positive:
            self.absence_evidence = False
            count = len(self.positive_times)
            if self.state in (self.ON, self.HELD):
                self.state, self.reason = self.ON, "fresh detection"
            elif count >= cfg["positive_confirmation_count"]:
                cancelled = self.state == self.SUSPECTED
                self.state = self.ON
                self.reason = f"confirmed: {count} detections within {cfg['positive_window_seconds']}s"
                if cancelled:
                    self.reason += "; suspected absence cancelled"
            else:
                self.reason = f"detected {count}/{cfg['positive_confirmation_count']}, awaiting confirmation"
            if self.state == self.ON:
                self.last_confirmed_t = t
        else:
            if self.state in (self.ON, self.HELD):
                gap = t - self.last_positive_t
                if gap <= cfg["positive_hold_seconds"] + EPS:
                    self.state, self.reason = self.HELD, f"held: last detection {gap:.2f}s ago"
                else:
                    self.state, self.reason = UNKNOWN, f"no fresh detection for {gap:.2f}s"
            stale = self.last_negative_t is None or t - self.last_negative_t > cfg["stale_negative_seconds"] + EPS
            if obs is not None and obs.eligible_negative:
                self.absence_evidence, why = self._absence_check(t, track_start)
            else:
                # An ambiguous/hidden/lost region cannot start an absence episode.
                # Episode continuity is handled independently below.
                self.absence_evidence = False
                why = "worker not detected this frame" if obs is None else "region not eligible for negative evidence"
            if self.absence_evidence:
                if self.absence_allowed:
                    self.state, self.reason = self.SUSPECTED, f"suspected absence: {why}"
                else:
                    self.state = UNKNOWN
                    self.reason = f"absence evidence met ({why}) but camera not validated: stays UNKNOWN"
            elif self.state == self.SUSPECTED:
                if stale:
                    self.state, self.reason = UNKNOWN, "suspected absence dropped: region no longer observable"
                elif self.positive_times:
                    self.reason = "suspected absence persists: single detection awaiting confirmation"
                else:
                    self.reason = "suspected absence persists"
            elif self.state == UNKNOWN:
                self.reason = f"insufficient evidence: {why}" if obs is not None else "not observed this frame"

        result = {"state": self.state, "reason": self.reason, "absence_evidence": self.absence_evidence,
                  "episode_started": False, "episode_ended": ended_early}
        stale = self.last_negative_t is None or t - self.last_negative_t > cfg["stale_negative_seconds"] + EPS
        recovered = self.state == self.ON
        # Crossing the evidence threshold STARTS an episode; a count/fraction dip
        # does not end it. Validated and unvalidated cameras share this lifecycle.
        # Only confirmed presence, expired visibility, or reset_evidence closes it.
        active = self.absence_evidence or (self.episode_active and not recovered and not stale)
        if active and not self.episode_active:
            self.episode_start_t = self.negative_start_t if self.negative_start_t is not None else t
            kind = "suspected" if self.state == self.SUSPECTED else "evidence_only_unvalidated"
            self.episodes.append([round(self.episode_start_t, 3), None, kind])
            result["episode_started"] = True
        elif not active and self.episode_active:
            self.episodes[-1][1] = round(t, 3)
            result["episode_ended"] = True
        self.episode_active = active
        return result


class AlertGate:
    """Per-track, per-violation alert decisions with a cooldown."""

    def __init__(self, cooldown_seconds: float):
        self.cooldown = cooldown_seconds
        self.last_emitted_t: float | None = None
        self.history: list[tuple[float, str]] = []
        self.pending = False
        self.episode_emitted = False

    def update(self, t: float, *, active: bool, episode_started: bool = False,
               track_confirmed: bool, camera_validated: bool, allow_unvalidated: bool,
               can_emit: bool = True) -> str:
        """One emitted alert per episode; retry a pending episode when cooldown expires."""
        if not active:
            self.pending = self.episode_emitted = False
            return ""
        if episode_started:
            self.pending, self.episode_emitted = True, False
        if self.episode_emitted or not self.pending:
            return ""
        if not can_emit:
            return ""
        if not episode_started:
            if not track_confirmed or (not camera_validated and not allow_unvalidated):
                return ""
            if self.last_emitted_t is not None and t - self.last_emitted_t < self.cooldown - EPS:
                return ""
        status = self.request(t, track_confirmed=track_confirmed, camera_validated=camera_validated,
                              allow_unvalidated=allow_unvalidated)
        if status == EMITTED:
            self.pending, self.episode_emitted = False, True
        return status

    def request(self, t: float, *, track_confirmed: bool, camera_validated: bool, allow_unvalidated: bool) -> str:
        if not track_confirmed:
            status = SUPPRESSED_UNCONFIRMED
        elif not camera_validated and not allow_unvalidated:
            status = SUPPRESSED_UNVALIDATED
        elif self.last_emitted_t is not None and t - self.last_emitted_t < self.cooldown - EPS:
            status = SUPPRESSED_COOLDOWN
        else:
            status = EMITTED
            self.last_emitted_t = t
        self.history.append((round(t, 3), status))
        return status


@dataclass
class Worker:
    """PPE history of one identity. worker_id is None until the identity is confirmed."""

    key: int
    start_t: float
    helmet: PPEStateMachine
    vest: PPEStateMachine
    helmet_alerts: AlertGate
    vest_alerts: AlertGate
    worker_id: int | None = None
    confirmed_t: float | None = None
    tracker_ids: list = field(default_factory=list)
    last_seen_t: float = 0.0
    person_conf: float = 0.0
    box: tuple = ()
    observed_frames: int = 0
    lost_frames: int = 0
    person_confs: list = field(default_factory=list)  # (t, conf)
    helmet_detections: list = field(default_factory=list)  # (t, conf)
    vest_detections: list = field(default_factory=list)
    head_observable_frames: int = 0
    torso_observable_frames: int = 0
    state_seconds: dict = field(default_factory=dict)  # observed frames only
    stitches: list = field(default_factory=list)

    @property
    def confirmed(self) -> bool:
        return self.worker_id is not None

    def observe_person(self, t: float, box, conf: float) -> None:
        self.last_seen_t = t
        self.box = tuple(round(float(v), 1) for v in box)
        self.person_conf = float(conf)
        self.person_confs.append((round(t, 3), float(conf)))
        self.observed_frames += 1

    def stitch(self, t: float, tracker_id: int) -> None:
        """A new tracker id continues this identity: keep the ID, drop all PPE evidence."""
        self.tracker_ids.append(tracker_id)
        self.stitches.append(round(t, 3))
        for machine in (self.helmet, self.vest):
            machine.reset_evidence(t, "identity stitched after track loss: PPE evidence reset")


def new_worker(key: int, t: float, ppe_cfg: dict, alert_cfg: dict, helmet_allowed: bool, vest_allowed: bool):
    return Worker(
        key=key,
        start_t=t,
        helmet=PPEStateMachine("helmet", ppe_cfg, helmet_allowed),
        vest=PPEStateMachine("vest", ppe_cfg, vest_allowed),
        helmet_alerts=AlertGate(alert_cfg["cooldown_seconds"]),
        vest_alerts=AlertGate(alert_cfg["cooldown_seconds"]),
        last_seen_t=t,
    )
