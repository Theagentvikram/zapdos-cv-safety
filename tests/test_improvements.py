"""Regression tests for timing, quality filtering, zones, evidence and event evaluation."""
import copy
import json
import tempfile
import threading
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.ppe_pipeline.association import associate, ASSOCIATED
from src.ppe_pipeline.evaluation import evaluate
from src.ppe_pipeline.identity import WorkerRegistry, appearance_descriptor
from src.ppe_pipeline.pipeline import load_config, process_video, output_component
from src.ppe_pipeline.quality import filter_detections
from src.ppe_pipeline.sampling import FrameSampler
from src.ppe_pipeline.state import PPEStateMachine, Observation, AlertGate, EMITTED
from src.ppe_pipeline.tracking import PersonTracker
from src.ppe_pipeline.zones import ZoneState, load_zones, floor_point

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "configs/ppe_pipeline.yaml")


class TemporalRegressionTests(unittest.TestCase):
    def test_gap_between_callbacks_expires_old_episode_before_fresh_evidence(self):
        for allowed in (False, True):
            m = PPEStateMachine("helmet", CFG["ppe"], allowed)
            for k in range(15):
                m.step(k * .2, 0, Observation(eligible_negative=True))
            result = m.step(5, 0, Observation(eligible_negative=True))
            self.assertTrue(result["episode_ended"])
            self.assertFalse(m.episode_active)
            self.assertEqual(m.state, "UNKNOWN")
            self.assertEqual(m.negative_start_t, 5)

    def test_cooldown_retry_waits_for_current_observable_evidence(self):
        gate = AlertGate(30)
        kw = dict(track_confirmed=True, camera_validated=True, allow_unvalidated=False)
        gate.update(0, active=True, episode_started=True, **kw)
        gate.update(1, active=False, **kw)
        gate.update(10, active=True, episode_started=True, **kw)
        self.assertEqual(gate.update(31, active=True, can_emit=False, **kw), "")
        self.assertEqual(gate.update(31.2, active=True, can_emit=True, **kw), EMITTED)

    def test_threshold_dip_does_not_restart_episode_on_either_camera_mode(self):
        for allowed in (False, True):
            with self.subTest(absence_allowed=allowed):
                m = PPEStateMachine("helmet", CFG["ppe"], allowed)
                starts = []
                for k in range(31):
                    result = m.step(k * .2, 0, Observation(eligible_negative=not 14 <= k <= 17))
                    if result["episode_started"]:
                        starts.append(k)
                self.assertEqual(starts, [13])
                self.assertTrue(m.episode_active)
                self.assertEqual(m.episode_start_t, 0)
                self.assertEqual(len(m.episodes), 1)

    def test_unvalidated_episode_requires_confirmed_recovery(self):
        m = PPEStateMachine("helmet", CFG["ppe"], False)
        for k in range(15):
            m.step(k * .2, 0, Observation(eligible_negative=True))
        m.step(3, 0, Observation(positive=True, conf=.8))
        self.assertTrue(m.episode_active)
        result = m.step(3.2, 0, Observation(positive=True, conf=.8))
        self.assertTrue(result["episode_ended"])
        self.assertFalse(m.episode_active)

    def test_stale_visibility_closes_episode_and_new_evidence_gets_new_onset(self):
        m = PPEStateMachine("helmet", CFG["ppe"], False)
        for k in range(15):
            m.step(k * .2, 0, Observation(eligible_negative=True))
        self.assertTrue(m.step(4, 0, None)["episode_ended"])
        for k in range(15):
            result = m.step(4.2 + k * .2, 0, Observation(eligible_negative=True))
        self.assertEqual(len(m.episodes), 2)
        self.assertAlmostEqual(m.episode_start_t, 4.2)

    def test_ineligible_current_frame_cannot_start_episode(self):
        m = PPEStateMachine("helmet", CFG["ppe"], True)
        for k in range(13):
            m.step(k * .2, 0, Observation(eligible_negative=True))
        result = m.step(2.6, 0, Observation())
        self.assertFalse(result["episode_started"])
        self.assertFalse(m.episode_active)

    def test_old_track_does_not_substitute_for_negative_duration(self):
        m = PPEStateMachine("helmet", CFG["ppe"], True)
        m.step(0, 0, Observation())
        for i in range(11):
            m.step(2.1 + i * .04, 0, Observation(eligible_negative=True))
        self.assertFalse(m.episode_active)
        for i in range(11, 75):
            m.step(2.1 + i * .04, 0, Observation(eligible_negative=True))
        self.assertTrue(m.episode_active)
        self.assertAlmostEqual(m.episode_start_t, 2.1)

    def test_stale_gap_restarts_absence_clock(self):
        m = PPEStateMachine("helmet", CFG["ppe"], True)
        for i in range(10):
            m.step(i * .2, 0, Observation(eligible_negative=True))
        m.step(4, 0, Observation(eligible_negative=True))
        self.assertFalse(m.episode_active)
        self.assertEqual(m.negative_start_t, 4)

    def test_cooldown_pending_episode_retries_once(self):
        gate = AlertGate(30)
        kw = dict(track_confirmed=True, camera_validated=True, allow_unvalidated=False)
        self.assertEqual(gate.update(0, active=True, episode_started=True, **kw), EMITTED)
        gate.update(5, active=False, **kw)
        self.assertEqual(gate.update(10, active=True, episode_started=True, **kw), "suppressed_cooldown")
        self.assertEqual(gate.update(29, active=True, **kw), "")
        self.assertEqual(gate.update(30, active=True, **kw), EMITTED)
        self.assertEqual(gate.update(60, active=True, **kw), "")

    def test_resolved_pending_episode_never_alerts(self):
        gate = AlertGate(30)
        kw = dict(track_confirmed=True, camera_validated=True, allow_unvalidated=False)
        gate.update(0, active=True, episode_started=True, **kw)
        gate.update(1, active=False, **kw)
        gate.update(10, active=True, episode_started=True, **kw)
        gate.update(12, active=False, **kw)
        self.assertEqual(gate.update(31, active=False, **kw), "")


class SamplingQualityTests(unittest.TestCase):
    def test_incompatible_appearance_vetoes_a_geometric_stitch(self):
        registry=WorkerRegistry(CFG['tracking'])
        a=np.zeros(192,np.float32);a[0]=1
        b=np.zeros(192,np.float32);b[-1]=1
        for i in range(3):
            registry.update(i*.1,[(1,(50,20,150,320),.9)],appearances=[a])
        registry.forget_tracker(1)
        result=registry.update(.6,[(2,(50,20,150,320),.9)],appearances=[b])[0]
        self.assertFalse(result.stitched)

    def test_compatible_appearance_allows_unique_geometric_stitch(self):
        registry=WorkerRegistry(CFG['tracking'])
        a=np.zeros(192,np.float32);a[0]=1
        for i in range(3):
            registry.update(i*.1,[(1,(50,20,150,320),.9)],appearances=[a])
        registry.forget_tracker(1)
        result=registry.update(.6,[(2,(50,20,150,320),.9)],appearances=[a])[0]
        self.assertTrue(result.stitched)
        self.assertEqual(result.worker_id,1)
    def test_output_names_are_short_stable_and_do_not_collide_on_prefix(self):
        a = output_component("same_long_camera_name_" * 4 + "A")
        b = output_component("same_long_camera_name_" * 4 + "B")
        self.assertLessEqual(len(a), 40)
        self.assertNotEqual(a, b)
        self.assertEqual(a, output_component("same_long_camera_name_" * 4 + "A"))
        self.assertNotIn("/", output_component("../camera/name"))
    def test_smooth_sampled_motion_does_not_split_identity(self):
        registry = WorkerRegistry(CFG["tracking"])
        for i in range(25):
            x = 100 + 45 * i
            registry.update(i * .2, [(1, (x, 20, x + 90, 320), .9)])
        self.assertEqual(registry.next_worker_id, 2)
        self.assertEqual(registry.splits, [])

    def test_partial_candidates_cannot_confirm(self):
        registry = WorkerRegistry(CFG["tracking"])
        for i in range(20):
            registry.update(i * .2, [(1, (10, 10, 80, 200), .9)], [False])
        self.assertEqual(registry.next_worker_id, 1)

    def test_two_new_tracks_cannot_race_for_one_lost_identity(self):
        registry = WorkerRegistry(CFG["tracking"])
        for i in range(3):
            registry.update(i * .1, [(1, (50, 20, 150, 320), .9)])
        registry.forget_tracker(1)
        assigned = registry.update(.6, [(2, (50, 20, 150, 320), .9), (3, (55, 20, 155, 320), .9)])
        self.assertFalse(any(a.stitched for a in assigned))

    def test_sampling_preserves_time_and_exact_budget(self):
        sampler = FrameSampler(25, 5)
        self.assertEqual([i for i in range(250) if sampler.keep(i)], list(range(0, 250, 5)))

    def test_non_integer_sampling_rate(self):
        sampler = FrameSampler(29.97, 5)
        selected = [i for i in range(300) if sampler.keep(i)]
        self.assertEqual(len(selected), 50)
        self.assertLess(max(abs(i / 29.97 - k / 5) for k, i in enumerate(selected)), 1 / 29.97)

    def test_invalid_rate_fails(self):
        for fps in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                FrameSampler(25, fps)

    def test_sampled_tracker_buffer_uses_update_rate(self):
        t = PersonTracker(CFG["tracking"], 5, (1080, 1920))
        self.assertEqual(t.tracker.max_frames_lost, round(2.5 * 5))

    def test_quality_keeps_low_confidence_people_but_rejects_text(self):
        rows = [[50, 50, 90, 200, .12, 0], [10, 10, 22, 32, .8, 0], [0, 0, 10, 10, float("nan"), 0]]
        kept, counts = filter_detections(rows, (300, 300), CFG["detection_quality"])
        self.assertEqual(len(kept), 1)
        self.assertEqual(sum(counts.values()), 2)

    def test_camera_panel_exclusion_is_local(self):
        rows = [[890, 160, 940, 265, .7, 0], [1200, 550, 1290, 870, .7, 0]]
        kept, _ = filter_detections(rows, (1080, 1920), CFG["detection_quality"], CFG["cameras"][0])
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0, 0], 1200)

    def test_weak_ppe_blocks_negative_without_becoming_positive(self):
        result = associate("helmet", [(0, 0, 100, 300)], [(25, 5, 70, 50)], [.15], CFG["association"])[0]
        self.assertNotEqual(result.status, ASSOCIATED)
        self.assertTrue(result.near_miss)


class ZoneTests(unittest.TestCase):
    def test_partial_equipment_level_box_does_not_establish_floor_contact(self):
        self.assertIsNone(floor_point((700,410,880,490),[],(1080,1920),CFG['zones']))
        self.assertIsNotNone(floor_point((700,300,800,650),[],(1080,1920),CFG['zones']))
    def test_entry_exit_hysteresis_and_unknown(self):
        state = ZoneState(CFG["zones"])
        self.assertFalse(state.step(0, 10)[0])
        self.assertFalse(state.step(.4, 10)[0])
        self.assertTrue(state.step(.6, 10)[0])
        self.assertFalse(state.step(.8, -2)[1])
        self.assertTrue(state.active)
        state.step(1, -10)
        self.assertTrue(state.step(1.6, -10)[1])
        self.assertFalse(state.active)

    def test_unknown_does_not_confirm_entry(self):
        state = ZoneState(CFG["zones"])
        state.step(0, 10)
        state.step(.2, None)
        self.assertFalse(state.step(.8, 10)[0])

    def test_hidden_feet_are_unknown(self):
        box = (30, 10, 70, 100)
        self.assertIsNone(floor_point(box, [(20, 90, 90, 120)], (128, 160), {}))
        self.assertEqual(floor_point(box, [], (128, 160), {}), (50, 100))

    def test_invalid_polygon_fails(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "zone.json"
            p.write_text(json.dumps({"zones": [{"id": "x", "polygon": [[0, 0], [1, 1]]}]}))
            with self.assertRaises(ValueError):
                load_zones(p, (100, 100))


class EvaluationTests(unittest.TestCase):
    def make_data(self):
        label = dict(video="x.mp4", camera_id="cam", type="no_helmet", person_id="P1", start_ts=0, end_ts=10)
        pred = dict(video="x.mp4", camera_id="cam", type="no_helmet", track_id=1, start_ts=0, end_ts=10,
                    alert_ts=2.6, event_id="a")
        return {"reviewed": True, "clips": ["x.mp4"], "events": [label]}, pred, {"reviewed": True, "tracks": {"x.mp4": {"1": "P1"}}}

    def test_duplicates_and_misses_are_counted(self):
        labels, pred, mapping = self.make_data()
        result = evaluate(labels, [pred, {**pred, "event_id": "b"}], mapping)
        self.assertEqual((result["true"], result["duplicates"], result["precision"]), (1, 1, .5))
        self.assertEqual(evaluate(labels, [], mapping)["missed"], 1)

    def test_missing_identity_mapping_is_not_a_match(self):
        labels, pred, mapping = self.make_data()
        mapping["tracks"] = {}
        result = evaluate(labels, [pred], mapping)
        self.assertEqual((result["false"], result["missed"]), (1, 1))

    def test_unreviewed_labels_fail(self):
        labels, pred, mapping = self.make_data()
        labels["reviewed"] = False
        with self.assertRaises(ValueError):
            evaluate(labels, [pred], mapping)

    def test_maximum_matching_avoids_greedy_miss(self):
        labels, pred, mapping = self.make_data()
        labels["events"] += [{**labels["events"][0], "start_ts": 9, "end_ts": 20}]
        result = evaluate(labels, [{**pred, "alert_ts": 9}, {**pred, "event_id": "b", "alert_ts": 2}], mapping, 0)
        self.assertEqual(result["true"], 2)


class PipelineIntegrationTests(unittest.TestCase):
    def run_pipeline(self, directory, broken=False):
        clip = Path(directory) / "synthetic.mp4"
        video = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 128))
        rng = np.random.default_rng(42)
        for _ in range(80):
            video.write(rng.integers(60, 200, (128, 160, 3), dtype=np.uint8))
        video.release()
        zones = Path(directory) / "zones.json"
        zones.write_text(json.dumps({"zones": [{"id": "test_floor", "polygon": [[.2, .6], [.8, .6], [.8, .95], [.2, .95]]}]}))
        cfg = copy.deepcopy(CFG)
        cfg["cameras"] = [dict(id="synthetic", scene="synthetic", match="synthetic", zones=str(zones),
                               helmet_absence_validated=True, vest_absence_validated=True)]

        class FakeDetector:
            backend = "fake"
            calls = 0

            def __call__(self, frame):
                self.calls += 1
                if broken and self.calls == 3:
                    raise RuntimeError("injected inference error")
                return np.array([[50, 20, 90, 115, .9, 0]], np.float32)

        return process_video(FakeDetector(), clip, cfg, Path(directory) / "out")

    def test_full_schema_clips_and_sampled_timestamps(self):
        with tempfile.TemporaryDirectory() as d:
            result = self.run_pipeline(d)
            self.assertEqual(result["detector_calls"], 40)
            self.assertEqual(result["source_frames"], 80)
            output = Path(result["annotated_video"]).parent
            events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
            self.assertEqual({e["type"] for e in events}, {"no_helmet", "no_vest", "zone_entry"})
            for event in events:
                for key in ("event_id", "camera_id", "type", "track_id", "start_ts", "end_ts", "first_frame", "confidence", "bbox", "snapshot_path", "clip_path"):
                    self.assertIn(key, event)
                self.assertLessEqual(event["start_ts"], event["end_ts"])
                self.assertTrue(Path(event["snapshot_path"]).is_file())
                cap = cv2.VideoCapture(event["clip_path"])
                self.assertTrue(cap.isOpened())
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.assertGreater(frames, 15)
                cap.release()
            self.assertEqual(len({e["event_id"] for e in events}), 3)

    def test_inference_error_cleans_up_background_threads(self):
        before = set(threading.enumerate())
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError, "injected"):
                self.run_pipeline(d, broken=True)
        self.assertEqual(set(threading.enumerate()), before)


if __name__ == "__main__":
    unittest.main()
