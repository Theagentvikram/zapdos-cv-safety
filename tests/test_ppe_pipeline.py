"""Deterministic tests for PPE association, the observability gate and the temporal state machines.

Run: python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ppe_pipeline.association import AMBIGUOUS, ASSOCIATED, NONE, associate  # noqa: E402
from src.ppe_pipeline.observability import assess  # noqa: E402
from src.ppe_pipeline.identity import WorkerRegistry  # noqa: E402
from src.ppe_pipeline.pipeline import load_config, untracked_persons  # noqa: E402
from src.ppe_pipeline.state import (  # noqa: E402
    EMITTED,
    SUPPRESSED_COOLDOWN,
    SUPPRESSED_UNVALIDATED,
    UNKNOWN,
    AlertGate,
    Observation,
    PPEStateMachine,
    new_worker,
)
from src.ppe_pipeline.tracking import PersonTracker  # noqa: E402

CFG = load_config(ROOT / "configs" / "ppe_pipeline.yaml")
FPS = 25.0
DT = 1 / FPS
SEEN = Observation(positive=True, conf=0.6)
MISS_ELIGIBLE = Observation(positive=False, eligible_negative=True)
MISS_INELIGIBLE = Observation(positive=False, eligible_negative=False)


def run(machine, observations, start_frame=0, track_start=0.0):
    """Feed one observation per frame; return list of (t, state after step, step result)."""
    out = []
    for k, obs in enumerate(observations):
        t = (start_frame + k) * DT
        result = machine.step(t, track_start, obs)
        out.append((t, machine.state, result))
    return out


def helmet_machine(validated=True):
    return PPEStateMachine("helmet", CFG["ppe"], absence_allowed=validated)


def textured_gray(h=1080, w=1920, low=60, high=200):
    return np.random.default_rng(0).integers(low, high, size=(h, w), dtype=np.uint8)


PERSON = (800.0, 300.0, 900.0, 600.0)  # 100 x 300 px standing worker


class HelmetStateMachineTests(unittest.TestCase):
    def test_01_helmet_continuously_visible(self):
        m = helmet_machine()
        trace = run(m, [SEEN] * 75)
        self.assertEqual(trace[0][1], UNKNOWN)  # one detection is not confirmation
        self.assertEqual(trace[1][1], "HELMET_ON")
        self.assertTrue(all(state == "HELMET_ON" for _, state, _ in trace[1:]))
        self.assertEqual(m.episodes, [])

    def test_02_helmet_disappears_for_one_frame(self):
        m = helmet_machine()
        trace = run(m, [SEEN] * 10 + [MISS_ELIGIBLE] + [SEEN] * 10)
        self.assertEqual(trace[10][1], "HELMET_ON_HELD")
        self.assertEqual(trace[11][1], "HELMET_ON")
        self.assertNotIn("SUSPECTED_NO_HELMET", [s for _, s, _ in trace])

    def test_03_helmet_disappears_for_one_second(self):
        m = helmet_machine()
        trace = run(m, [SEEN] * 10 + [MISS_ELIGIBLE] * 25 + [SEEN] * 5)
        states = [s for _, s, _ in trace]
        self.assertTrue(all(s == "HELMET_ON_HELD" for s in states[10:35]))
        self.assertEqual(states[35], "HELMET_ON")
        self.assertNotIn("SUSPECTED_NO_HELMET", states)

        # Longer than the 1 s hold: falls back to UNKNOWN, still not suspected before 2.5 s.
        m = helmet_machine()
        trace = run(m, [SEEN] * 10 + [MISS_ELIGIBLE] * 40)
        states = [s for _, s, _ in trace]
        self.assertEqual(states[-1], UNKNOWN)
        self.assertNotIn("SUSPECTED_NO_HELMET", states)

    def test_04_no_helmet_for_2_5_seconds_with_eligible_observations(self):
        m = helmet_machine(validated=True)
        trace = run(m, [MISS_ELIGIBLE] * 100)
        first = next(t for t, s, _ in trace if s == "SUSPECTED_NO_HELMET")
        self.assertGreaterEqual(first, 2.5 - 1e-9)
        self.assertLess(first, 2.5 + DT)  # first frame at or after 2.5 s
        started = [t for t, _, r in trace if r["episode_started"]]
        self.assertEqual(len(started), 1)  # one episode while the violation stays active

        # Unvalidated camera: same evidence, but the state stays UNKNOWN and is flagged.
        m = helmet_machine(validated=False)
        trace = run(m, [MISS_ELIGIBLE] * 100)
        self.assertNotIn("SUSPECTED_NO_HELMET", [s for _, s, _ in trace])
        self.assertTrue(trace[-1][2]["absence_evidence"])
        self.assertIn("camera not validated", trace[-1][2]["reason"])

    def test_05_head_occluded(self):
        blocker = (780.0, 250.0, 930.0, 420.0)  # second worker covering the head region
        ok, reasons, _ = assess("helmet", PERSON, 0.8, [blocker], textured_gray(), CFG["observability"], CFG["association"])
        self.assertFalse(ok)
        self.assertIn("occluded by another worker", reasons)
        m = helmet_machine(validated=True)
        trace = run(m, [MISS_INELIGIBLE] * 150)
        self.assertTrue(all(s == UNKNOWN for _, s, _ in trace))
        self.assertFalse(m.absence_evidence)

    def test_06_torso_occluded(self):
        blocker = (780.0, 380.0, 940.0, 560.0)
        ok, reasons, _ = assess("vest", PERSON, 0.8, [blocker], textured_gray(), CFG["observability"], CFG["association"])
        self.assertFalse(ok)
        self.assertIn("occluded by another worker", reasons)
        head_ok, _, _ = assess("helmet", PERSON, 0.8, [blocker], textured_gray(), CFG["observability"], CFG["association"])
        self.assertTrue(head_ok)  # occluding the torso does not block the head
        m = PPEStateMachine("vest", CFG["ppe"], absence_allowed=True)
        trace = run(m, [MISS_INELIGIBLE] * 150)
        self.assertTrue(all(s == UNKNOWN for _, s, _ in trace))

    def test_07_worker_track_disappears_temporarily(self):
        m = helmet_machine(validated=True)
        trace = run(m, [SEEN] * 10 + [None] * 15 + [SEEN] * 5)
        states = [s for _, s, _ in trace]
        self.assertTrue(all(s == "HELMET_ON_HELD" for s in states[10:25]))
        self.assertEqual(states[25], "HELMET_ON")

        # Lost for 3 s: no fresh evidence either way, so UNKNOWN, never suspected.
        m = helmet_machine(validated=True)
        trace = run(m, [SEEN] * 10 + [None] * 75)
        states = [s for _, s, _ in trace]
        self.assertEqual(states[-1], UNKNOWN)
        self.assertNotIn("SUSPECTED_NO_HELMET", states)
        self.assertEqual(m.eligible_negative_count, 0)

    def test_10_helmeted_worker_missed_repeatedly_in_low_light(self):
        dark = np.full((1080, 1920), 12, dtype=np.uint8)
        ok, reasons, _ = assess("helmet", PERSON, 0.8, [], dark, CFG["observability"], CFG["association"])
        self.assertFalse(ok)
        self.assertIn("too dark", reasons)
        m = helmet_machine(validated=True)
        trace = run(m, [MISS_INELIGIBLE] * 200)  # misses that the gate marks ineligible
        self.assertNotIn("SUSPECTED_NO_HELMET", [s for _, s, _ in trace])
        self.assertEqual(m.state, UNKNOWN)

    def test_11_worker_starts_with_no_helmet_detection(self):
        m = helmet_machine(validated=True)
        trace = run(m, [MISS_ELIGIBLE] * 62)  # up to t = 2.44 s
        self.assertTrue(all(s == UNKNOWN for _, s, _ in trace))
        self.assertIn("track age", trace[0][2]["reason"])

    def test_suspected_cancelled_only_after_positive_confirmation(self):
        m = helmet_machine(validated=True)
        run(m, [MISS_ELIGIBLE] * 70)
        self.assertEqual(m.state, "SUSPECTED_NO_HELMET")
        run(m, [SEEN], start_frame=70)
        self.assertEqual(m.state, "SUSPECTED_NO_HELMET")  # a single detection is not enough
        run(m, [SEEN], start_frame=71)
        self.assertEqual(m.state, "HELMET_ON")
        self.assertIsNotNone(m.episodes[-1][1])


class AssociationTests(unittest.TestCase):
    A = CFG["association"]

    def test_08_two_workers_side_by_side(self):
        left, right = (500.0, 300.0, 600.0, 600.0), (610.0, 300.0, 710.0, 600.0)
        helmets = [(630.0, 302.0, 690.0, 340.0), (520.0, 305.0, 580.0, 345.0)]
        matches = associate("helmet", [left, right], helmets, [0.7, 0.8], self.A)
        self.assertEqual((matches[0].status, matches[0].ppe_index), (ASSOCIATED, 1))
        self.assertEqual((matches[1].status, matches[1].ppe_index), (ASSOCIATED, 0))

        # Heavily overlapping workers and one helmet that fits both equally: ambiguous, not assigned.
        a, b = (500.0, 300.0, 600.0, 600.0), (505.0, 302.0, 605.0, 602.0)
        matches = associate("helmet", [a, b], [(525.0, 305.0, 580.0, 340.0)], [0.9], self.A)
        self.assertEqual([m.status for m in matches], [AMBIGUOUS, AMBIGUOUS])
        self.assertTrue(all(m.ppe_index is None for m in matches))

    def test_one_to_one_matching(self):
        person = (500.0, 300.0, 600.0, 600.0)
        helmets = [(520.0, 305.0, 580.0, 345.0), (525.0, 310.0, 575.0, 350.0)]
        matches = associate("helmet", [person], helmets, [0.5, 0.9], self.A)
        self.assertEqual(matches[0].status, ASSOCIATED)
        self.assertEqual(sum(1 for m in matches if m.status == ASSOCIATED), 1)

    def test_09_helmet_on_machinery(self):
        person = (500.0, 300.0, 600.0, 600.0)
        on_machine = (900.0, 400.0, 960.0, 440.0)  # far from any worker
        at_waist = (530.0, 450.0, 580.0, 490.0)  # carried / resting near the body, not on the head
        oversized = (440.0, 280.0, 660.0, 420.0)  # implausibly large for this person
        for box in (on_machine, at_waist, oversized):
            matches = associate("helmet", [person], [box], [0.9], self.A)
            self.assertEqual(matches[0].status, NONE, box)
        self.assertEqual(associate("helmet", [], [on_machine], [0.9], self.A), [])


class TrackAndAlertTests(unittest.TestCase):
    def test_12_second_alert_request_during_cooldown(self):
        gate = AlertGate(CFG["alerts"]["cooldown_seconds"])
        self.assertEqual(gate.request(3.0, track_confirmed=True, camera_validated=True, allow_unvalidated=False), EMITTED)
        self.assertEqual(gate.request(10.0, track_confirmed=True, camera_validated=True, allow_unvalidated=False),
                         SUPPRESSED_COOLDOWN)
        self.assertEqual(gate.request(33.5, track_confirmed=True, camera_validated=True, allow_unvalidated=False), EMITTED)

    def test_unvalidated_camera_never_emits(self):
        gate = AlertGate(CFG["alerts"]["cooldown_seconds"])
        status = gate.request(5.0, track_confirmed=True, camera_validated=False, allow_unvalidated=False)
        self.assertEqual(status, SUPPRESSED_UNVALIDATED)

    def test_ladle_bay_known_helmets_missed_do_not_create_violations(self):
        # Ladle-bay pattern: workers labelled as wearing helmets, detector finds a helmet in ~2% of frames.
        m = helmet_machine(validated=False)
        gate = AlertGate(CFG["alerts"]["cooldown_seconds"])
        observations = [SEEN if k in (40, 200) else MISS_ELIGIBLE for k in range(375)]
        statuses = []
        for t, state, result in run(m, observations):
            self.assertNotEqual(state, "SUSPECTED_NO_HELMET")
            if result["episode_started"]:
                statuses.append(gate.request(t, track_confirmed=True, camera_validated=False, allow_unvalidated=False))
        self.assertTrue(statuses)
        self.assertTrue(all(s == SUPPRESSED_UNVALIDATED for s in statuses))

    def test_stitch_resets_ppe_evidence_but_keeps_worker(self):
        w = new_worker(1, 0.0, CFG["ppe"], CFG["alerts"], True, True)
        w.worker_id = 4
        run(w.helmet, [MISS_ELIGIBLE] * 70)
        self.assertEqual(w.helmet.state, "SUSPECTED_NO_HELMET")
        w.stitch(3.0, 9)
        self.assertEqual((w.worker_id, w.helmet.state, w.helmet.eligible_negative_count), (4, UNKNOWN, 0))
        self.assertIsNotNone(w.helmet.episodes[-1][1])  # episode closed at the stitch
        # The absence clock restarts at the stitch: no evidence again before 2.5 s of new observations.
        trace = run(w.helmet, [MISS_ELIGIBLE] * 60, start_frame=75, track_start=0.0)
        self.assertNotIn("SUSPECTED_NO_HELMET", [s for _, s, _ in trace])


def tracking_cfg(**overrides):
    return {**CFG["tracking"], **overrides}


def feed(registry, t, *items):
    """items: (tracker_id, box, conf)."""
    return registry.update(t, [(tid, tuple(map(float, box)), conf) for tid, box, conf in items])


class WorkerIdentityTests(unittest.TestCase):
    BOX = (500, 300, 600, 600)

    def test_confirmation_needs_three_detections_in_window(self):
        reg = WorkerRegistry(tracking_cfg())
        for t in (0.0, 0.5, 1.2):  # never three within 0.6 s
            self.assertIsNone(feed(reg, t, (1, self.BOX, 0.6))[0].worker_id)
        feed(reg, 1.24, (1, self.BOX, 0.6))
        a = feed(reg, 1.28, (1, self.BOX, 0.6))[0]
        self.assertEqual((a.worker_id, a.newly_confirmed), (1, True))  # stationary worker: no motion required
        self.assertFalse(feed(reg, 1.32, (1, self.BOX, 0.6))[0].newly_confirmed)

    def test_low_confidence_or_tiny_box_never_confirmed(self):
        reg = WorkerRegistry(tracking_cfg())
        for k in range(50):
            a, b = feed(reg, k * DT, (1, self.BOX, 0.15), (2, (100, 100, 110, 130), 0.9))
            self.assertIsNone(a.worker_id)  # median conf below min_confirm_median_conf
            self.assertIsNone(b.worker_id)  # 30 px tall static box: below min_confirm_height_px

    def test_worker_ids_are_sequential_and_skip_candidates(self):
        reg = WorkerRegistry(tracking_cfg())
        for k in range(5):
            out = feed(reg, k * DT, (3, self.BOX, 0.7), (7, (100, 100, 110, 130), 0.9), (12, (900, 300, 1000, 600), 0.7))
        self.assertEqual([a.worker_id for a in out], [1, None, 2])  # ByteTrack ids 3/7/12 do not leak into worker ids
        out = feed(reg, 5 * DT, (20, (1200, 300, 1300, 600), 0.7))
        for k in range(6, 9):
            out = feed(reg, k * DT, (20, (1200, 300, 1300, 600), 0.7))
        self.assertEqual(out[0].worker_id, 3)

    def test_assignments_follow_input_order(self):
        reg = WorkerRegistry(tracking_cfg())
        for k in range(5):
            feed(reg, k * DT, (8, self.BOX, 0.7))
        # A new tracker id listed before a known one must not receive the known worker's identity.
        new, known = feed(reg, 5 * DT, (5, (69, 985, 81, 1005), 0.3), (8, self.BOX, 0.7))
        self.assertEqual((new.tracker_id, new.worker_id), (5, None))
        self.assertEqual((known.tracker_id, known.worker_id), (8, 1))

    def test_jump_guard_splits_swapped_track(self):
        reg = WorkerRegistry(tracking_cfg())
        for k in range(10):
            first = feed(reg, k * DT, (2, (588, 462, 675, 661), 0.7))[0]
        self.assertEqual(first.worker_id, 1)
        # Workshop 8.2 s pattern: ByteTrack moves id 2 from a walking worker onto a seated one (IoU 0.18).
        jumped = feed(reg, 10 * DT, (2, (637, 564, 700, 672), 0.7))[0]
        self.assertNotEqual(jumped.key, first.key)
        self.assertIsNone(jumped.worker_id)
        self.assertEqual(len(reg.splits), 1)

    def test_jump_guard_ignores_box_height_flicker(self):
        reg = WorkerRegistry(tracking_cfg())
        for k in range(10):
            first = feed(reg, k * DT, (4, (385, 672, 487, 776), 0.7))[0]
        # Seated worker whose box sometimes includes the head: 1.6x taller, IoU 0.61.
        same = feed(reg, 10 * DT, (4, (384, 614, 490, 777), 0.7))[0]
        self.assertEqual((same.key, same.worker_id), (first.key, 1))
        self.assertEqual(reg.splits, [])

    def test_stitching_rules(self):
        def confirmed_then_lost(cfg):
            reg = WorkerRegistry(cfg)
            for k in range(10):
                feed(reg, k * DT, (1, self.BOX, 0.7))
            reg.forget_tracker(1)
            return reg

        reg = confirmed_then_lost(tracking_cfg(stitch_max_gap_seconds=3.0))
        a = feed(reg, 2.0, (8, (505, 305, 605, 605), 0.7))[0]
        self.assertEqual((a.worker_id, a.stitched), (1, True))

        reg = confirmed_then_lost(tracking_cfg(stitch_max_gap_seconds=3.0))
        self.assertFalse(feed(reg, 4.0, (8, (505, 305, 605, 605), 0.7))[0].stitched)  # gap too long

        reg = confirmed_then_lost(tracking_cfg(stitch_max_gap_seconds=3.0))
        self.assertFalse(feed(reg, 2.0, (8, (505, 200, 605, 650), 0.7))[0].stitched)  # 1.5x taller

        reg = confirmed_then_lost(tracking_cfg(stitch_max_gap_seconds=0))
        self.assertFalse(feed(reg, 2.0, (8, (505, 305, 605, 605), 0.7))[0].stitched)  # disabled

        # Two lost workers both fit the new box: refuse rather than guess.
        reg = WorkerRegistry(tracking_cfg(stitch_max_gap_seconds=3.0))
        for k in range(10):
            feed(reg, k * DT, (1, (500, 300, 600, 600), 0.7), (2, (520, 300, 620, 600), 0.7))
        reg.forget_tracker(1), reg.forget_tracker(2)
        self.assertFalse(feed(reg, 2.0, (8, (510, 300, 610, 600), 0.7))[0].stitched)


class ByteTrackIntegrationTests(unittest.TestCase):
    """Real Ultralytics BYTETracker (configured from the YAML) feeding the worker registry."""

    @staticmethod
    def run_frames(frames):
        """frames: list of lists of (x1, y1, x2, y2, conf). Returns per-frame {det index: worker_id}."""
        tracker = PersonTracker(CFG["tracking"], FPS, (1080, 1920))
        registry = WorkerRegistry(CFG["tracking"])
        out = []
        for k, dets in enumerate(frames):
            rows = np.array([[*d, 0.0] for d in dets], np.float32).reshape(-1, 6)
            matched = tracker.update(rows)
            alive = tracker.alive_ids()
            assigns = registry.update(k * DT, [(tid, tuple(map(float, rows[i, :4])), float(rows[i, 4]))
                                               for tid, i in matched])
            for tid in list(registry.by_tracker):
                if tid not in alive:
                    registry.forget_tracker(tid)
            out.append({i: a.worker_id for (_, i), a in zip(matched, assigns)})
        return out, registry

    def test_person_missing_two_seconds_keeps_worker_id(self):
        person = (800, 300, 900, 600, 0.8)
        frames = [[person]] * 25 + [[]] * 50 + [[person]] * 25
        out, registry = self.run_frames(frames)
        self.assertEqual(out[24][0], 1)
        self.assertEqual({o[0] for o in out[75:] if 0 in o}, {1})
        self.assertEqual(registry.next_worker_id, 2)  # no second worker was created

    def test_side_by_side_workers_stay_distinct(self):
        frames = []
        for k in range(100):
            dx = 2 * k  # both walk right together, 10 px apart
            frames.append([(500 + dx, 300, 600 + dx, 600, 0.8), (610 + dx, 302, 710 + dx, 602, 0.75)])
        out, registry = self.run_frames(frames)
        ids = {(o.get(0), o.get(1)) for o in out[5:]}
        self.assertEqual(ids, {(1, 2)})
        self.assertEqual(registry.splits, [])

    def test_crossing_workers_do_not_swap_identity(self):
        frames = []
        for k in range(80):
            frames.append([(400 + 5 * k, 300, 500 + 5 * k, 600, 0.8), (800 - 5 * k, 320, 900 - 5 * k, 640, 0.8)])
        out, _ = self.run_frames(frames)
        # Detection 0 always moves right; it must keep one worker id for the whole crossing (or lose it, never swap).
        left_ids = {o[0] for o in out if 0 in o and o[0] is not None}
        right_ids = {o[1] for o in out if 1 in o and o[1] is not None}
        self.assertTrue(left_ids.isdisjoint(right_ids), (left_ids, right_ids))

    def test_tiny_static_low_confidence_box_never_confirmed(self):
        frames = [[(800, 300, 900, 600, 0.8), (1500, 200, 1512, 230, 0.2)]] * 75
        out, registry = self.run_frames(frames)
        self.assertEqual({o.get(1) for o in out}, {None})
        self.assertEqual(registry.next_worker_id, 2)


class UntrackedPersonTests(unittest.TestCase):
    A = CFG["association"]
    TRACKED = (500.0, 300.0, 600.0, 600.0)
    UNTRACKED = (560.0, 300.0, 660.0, 600.0)  # IoU 0.25 with TRACKED: a real neighbour, not a duplicate

    def test_untracked_neighbour_takes_its_own_helmet(self):
        helmet = [(565.0, 305.0, 615.0, 345.0)]  # on the neighbour's head, overlapping the tracked head region
        alone = associate("helmet", [self.TRACKED], helmet, [0.8], self.A)
        self.assertEqual(alone[0].status, ASSOCIATED)  # what happened before untracked persons were included
        both = associate("helmet", [self.TRACKED, self.UNTRACKED], helmet, [0.8], self.A)
        self.assertEqual((both[0].status, both[1].status), (NONE, ASSOCIATED))

    def test_untracked_neighbour_makes_shared_helmet_ambiguous(self):
        helmet = [(555.0, 305.0, 605.0, 345.0)]  # equally plausible for both heads
        both = associate("helmet", [self.TRACKED, self.UNTRACKED], helmet, [0.8], self.A)
        self.assertEqual((both[0].status, both[1].status), (AMBIGUOUS, AMBIGUOUS))

    def test_duplicate_untracked_boxes_are_dropped(self):
        persons = np.array([[500, 300, 600, 600, 0.8, 0],   # tracked
                            [505, 302, 602, 604, 0.3, 0],   # duplicate of the tracked box
                            [560, 300, 660, 600, 0.6, 0],   # real neighbour
                            [562, 301, 661, 601, 0.4, 0]],  # duplicate of the neighbour
                           np.float32)
        kept = untracked_persons(persons, {0}, CFG["tracking"]["duplicate_person_iou"])
        self.assertEqual([round(c, 2) for _, c in kept], [0.6])


if __name__ == "__main__":
    unittest.main()
