"""Video loop: one SH17 forward pass per frame -> ByteTrack on persons -> worker identities
-> PPE association against every person detection -> observability gate -> temporal state machines
-> rendering, logs and events.

Decoding and all file writing run on background threads with bounded queues; detection, tracking,
association, rules and rendering run in order on the main thread.
"""

from __future__ import annotations

import csv
import json
import queue
import statistics
import threading
import time
import platform
import hashlib
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

from . import render
from .association import ASSOCIATED, AMBIGUOUS, associate
from .detector import Detector
from .identity import WorkerRegistry, iou, appearance_descriptor
from .observability import assess
from .state import EMITTED, Observation, new_worker
from .tracking import PersonTracker
from .sampling import FrameSampler
from .quality import filter_detections
from .evidence import ClipRecorder
from .zones import load_zones, floor_point, ZoneState, draw_zones
from .state import AlertGate
import uuid

STAGES = ("decode", "inference", "tracking", "association", "rules", "render", "write")

LOG_FIELDS = [
    "frame", "t_sec", "identity", "worker_id", "tracker_id", "confirmed", "observed", "person_conf",
    "x1", "y1", "x2", "y2",
    "helmet_assoc", "helmet_conf", "helmet_near_miss", "head_observable", "head_gate",
    "vest_assoc", "vest_conf", "vest_near_miss", "torso_observable", "torso_gate",
    "helmet_state", "helmet_absence_evidence", "helmet_eligible_negatives", "helmet_reason",
    "vest_state", "vest_absence_evidence", "vest_eligible_negatives", "vest_reason",
    "last_helmet_t", "last_vest_t", "helmet_alert", "vest_alert",
    "zone_point_x", "zone_point_y", "zone_visibility", "zone_states",
]

CANDIDATE_FIELDS = ["frame", "t_sec", "kind", "identity", "tracker_id", "person_conf", "x1", "y1", "x2", "y2",
                    "helmet_assoc", "vest_assoc"]


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg["_config_dir"] = str(Path(path).resolve().parent)
    if cfg.get("runtime", {}).get("target_fps") is not None and cfg["runtime"]["target_fps"] <= 0:
        raise ValueError("runtime.target_fps must be positive")
    if cfg["output"].get("queue_frames", 8) <= 0:
        raise ValueError("output.queue_frames must be positive")
    return cfg


def camera_for(clip: Path, cfg: dict, camera_id: str | None = None) -> dict:
    if camera_id is not None:
        for camera in cfg.get("cameras", []):
            if camera["id"] == camera_id:
                return dict(camera)
        return {"id": camera_id, "scene": camera_id, "helmet_absence_validated": False, "vest_absence_validated": False}
    for camera in cfg.get("cameras", []):
        if clip.name.startswith(camera["match"]):
            return camera
    return {"id": clip.stem, "scene": clip.stem, "helmet_absence_validated": False, "vest_absence_validated": False}


def warmup(detector: Detector, clip: Path, passes: int = 5) -> None:
    """Untimed forward passes so first-call CPU setup is not counted as pipeline latency."""
    cap = cv2.VideoCapture(str(clip))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"cannot read {clip}")
    for _ in range(passes):
        detector(frame)


def _round(v, n=3):
    return None if v is None else round(float(v), n)


def _median(values):
    return round(statistics.median(values), 4) if values else None


def _box_cells(box) -> dict:
    return {"x1": round(box[0], 1), "y1": round(box[1], 1), "x2": round(box[2], 1), "y2": round(box[3], 1)}


def output_component(value: str) -> str:
    """Short, stable directory names avoid Windows codec/image path limits."""
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", str(value)).strip(".") or "unnamed"
    if len(safe) > 40 or safe != value:
        return safe[:24] + "-" + hashlib.sha256(str(value).encode()).hexdigest()[:10]
    return safe


def untracked_persons(persons: np.ndarray, tracked_idx: set, dup_iou: float) -> list[tuple[tuple, float]]:
    """Person detections ByteTrack did not match, minus near-duplicates of tracked boxes or of each other.

    They still take part in PPE association and occlusion checks, but never get PPE state.
    """
    kept: list[tuple[tuple, float]] = []
    tracked = [tuple(map(float, persons[i, :4])) for i in tracked_idx]
    for i in sorted(set(range(len(persons))) - tracked_idx, key=lambda i: -persons[i, 4]):
        box = tuple(map(float, persons[i, :4]))
        if any(iou(box, other) > dup_iou for other in tracked + [b for b, _ in kept]):
            continue
        kept.append((box, float(persons[i, 4])))
    return kept


class _Reader(threading.Thread):
    """Decodes frames ahead of the main loop into a bounded queue; None marks the end."""

    def __init__(self, cap, max_frames, size, sampler, realtime=False):
        super().__init__(daemon=True)
        self.cap, self.max_frames, self.q = cap, max_frames, queue.Queue(maxsize=size)
        self.sampler, self.stop = sampler, threading.Event()
        self.error = None
        self.source_frames = 0
        self.realtime, self.max_depth = realtime, 0

    def put(self, item):
        while not self.stop.is_set():
            try:
                self.q.put(item, timeout=0.1)
                self.max_depth=max(self.max_depth,self.q.qsize())
                return
            except queue.Full:
                pass

    def run(self):
        index, decode_ms = 0, 0.0
        origin=time.perf_counter()
        try:
            while not self.stop.is_set() and (self.max_frames is None or index < self.max_frames):
                scheduled=origin+index/self.sampler.source_fps
                if self.realtime and self.stop.wait(max(0,scheduled-time.perf_counter())):
                    break
                mark = time.perf_counter()
                if not self.cap.grab():
                    break
                keep = self.sampler.keep(index)
                frame = None
                if keep:
                    ok, frame = self.cap.retrieve()
                    if not ok:
                        raise OSError(f"cannot retrieve frame {index}")
                decode_ms += (time.perf_counter() - mark) * 1000
                self.source_frames = index + 1
                if keep:
                    self.put((index, frame, decode_ms, scheduled if self.realtime else time.perf_counter()))
                    decode_ms = 0.0
                index += 1
        except Exception as exc:
            self.error = exc
        finally:
            self.put(None)


class _Writer(threading.Thread):
    """Writes annotated frames, CSV rows and evidence images off the main thread."""

    def __init__(self, video, log, candidates, size, clips):
        super().__init__(daemon=True)
        self.video, self.log, self.candidates = video, log, candidates
        self.q = queue.Queue(maxsize=size)
        self.ms: list[float] = []
        self.error: Exception | None = None
        self.clips = clips

    def run(self):
        while True:
            job = self.q.get()
            if job is None:
                self.clips.close()
                return
            mark = time.perf_counter()
            try:
                t, canvas, rows, cand_rows, images, clips = job
                self.video.write(canvas)
                self.log.writerows(rows)
                self.candidates.writerows(cand_rows)
                for path, image in images:
                    if not cv2.imwrite(str(path), image):
                        raise OSError(f"cannot write snapshot: {path}")
                self.clips.push(t, canvas, clips)
            except Exception as exc:  # surfaced by the main thread after join
                self.error = exc
            self.ms.append((time.perf_counter() - mark) * 1000)


def process_video(detector: Detector, clip: Path, cfg: dict, out_root: Path, max_frames: int | None = None) -> dict:
    tc, ac, oc, pc, alc = (cfg[k] for k in ("tracking", "association", "observability", "ppe", "alerts"))
    mc = cfg["model"]
    camera = camera_for(clip, cfg)
    allow_unvalidated = bool(alc["allow_unvalidated_camera_absence_alerts"])
    helmet_validated = bool(camera["helmet_absence_validated"])
    vest_validated = bool(camera["vest_absence_validated"])
    helmet_allowed = helmet_validated or allow_unvalidated
    vest_allowed = vest_validated or allow_unvalidated
    ids = {name: int(k) for k, name in mc["class_names"].items()}
    dup_iou = tc["duplicate_person_iou"]
    stitch_gap = tc.get("stitch_max_gap_seconds") or 0

    out_dir = out_root / output_component(camera["scene"]) / output_component(clip.stem)
    (out_dir / "evidence").mkdir(parents=True, exist_ok=True)
    (out_dir / "snapshots").mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise OSError(f"cannot open input video: {clip}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sampler = FrameSampler(fps, cfg.get("runtime", {}).get("target_fps"))
    processed_fps = sampler.rate
    zone_path = camera.get("zones")
    zones = load_zones(Path(cfg.get("_config_dir", ".")) / zone_path if zone_path else None, (height, width))
    zone_cfg = {**cfg.get("zones", {}), "feet_occlusion_polygons": camera.get("feet_occlusion_polygons", [])}
    zone_states, zone_gates, zone_events = {}, {}, []
    snapshot_frames = {int(n_frames * f) for f in cfg["output"]["snapshot_fractions"]}
    video_path = out_dir / "annotated.mp4"
    video = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*cfg["output"]["video_codec"]), processed_fps, (width, height))
    if not video.isOpened():
        cap.release()
        raise OSError(f"cannot write annotated video: {video_path}")

    tracker = PersonTracker(tc, processed_fps, (height, width))
    registry = WorkerRegistry(tc)
    workers: dict = {}  # identity key -> Worker
    finished: list = []
    events = []
    event_episodes = []
    rejected_counts = Counter()
    detector_calls = 0
    frame_capture_times = {}
    diagnostics_handle = open(out_dir / "diagnostics.jsonl", "w", encoding="utf-8")
    timings = {stage: [] for stage in STAGES + ("total", "frame_age")}
    untracked_person_dets = untracked_kept = 0
    log_handle = open(out_dir / "worker_log.csv", "w", newline="", encoding="utf-8")
    log = csv.DictWriter(log_handle, fieldnames=LOG_FIELDS)
    log.writeheader()
    cand_handle = open(out_dir / "candidates.csv", "w", newline="", encoding="utf-8")
    cand_log = csv.DictWriter(cand_handle, fieldnames=CANDIDATE_FIELDS)
    cand_log.writeheader()
    events_handle = open(out_dir / "events.jsonl", "w", encoding="utf-8")

    qsize = int(cfg["output"].get("queue_frames", 8))
    realtime=bool(cfg.get("runtime",{}).get("realtime",False))
    reader = _Reader(cap, max_frames, qsize, sampler, realtime)
    writer = _Writer(video, log, cand_log, qsize,
                     ClipRecorder(processed_fps, (height, width), cfg["output"].get("evidence_seconds", 3)))
    reader.start()
    writer.start()

    frames = 0
    wall_start = time.perf_counter()
    try:
        while True:
            start = time.perf_counter()
            item = reader.q.get()
            if item is None:
                break
            index, frame, decode_ms, captured_at = item
            frame_capture_times[index] = captured_at
            t = index / fps
            stage = {}
            mark = time.perf_counter()

            det = detector(frame)
            detector_calls += 1
            # Only person filtering is destructive. Weak/odd PPE stays available as an
            # uncertainty blocker and is prevented from becoming positive by association.
            det, rejected = filter_detections(det, frame.shape, cfg.get("detection_quality", {}), camera)
            rejected_counts.update(rejected)
            now = time.perf_counter(); stage["inference"] = now - mark; mark = now

            cls = det[:, 5].astype(int)
            persons, helmets, vests = det[cls == ids["person"]], det[cls == ids["helmet"]], det[cls == ids["safety-vest"]]
            matched = tracker.update(persons)
            alive = tracker.alive_ids()
            lost_boxes = tracker.lost()
            triples = [(tid, tuple(map(float, persons[i, :4])), float(persons[i, 4])) for tid, i in matched]
            border = tc.get("confirmation_border_margin_px", 0)
            eligible_confirmation = [border <= b[0] and border <= b[1] and b[2] < width - border and b[3] < height - border
                                     for _, b, _ in triples] if border else None
            appearances = [appearance_descriptor(frame,b) for _,b,_ in triples] if tc.get("appearance_stitch_guard",False) else None
            assignments = registry.update(t, triples, eligible_confirmation, appearances)
            for tid in list(registry.by_tracker):
                if tid not in alive:
                    registry.forget_tracker(tid)
            now = time.perf_counter(); stage["tracking"] = now - mark; mark = now

            tracked_boxes = [box for _, box, _ in triples]
            extra = untracked_persons(persons, {i for _, i in matched}, dup_iou)
            untracked_person_dets += len(persons) - len(matched)
            untracked_kept += len(extra)
            all_boxes = tracked_boxes + [b for b, _ in extra]
            helmet_boxes = [tuple(map(float, b)) for b in helmets[:, :4]]
            vest_boxes = [tuple(map(float, b)) for b in vests[:, :4]]
            helmet_matches = associate("helmet", all_boxes, helmet_boxes, list(helmets[:, 4]), ac)
            vest_matches = associate("vest", all_boxes, vest_boxes, list(vests[:, 4]), ac)
            now = time.perf_counter(); stage["association"] = now - mark; mark = now

            rows, cand_rows, draw_items, evidence_requests = [], [], [], []
            observed_keys = set()
            for k, ((tid, box, conf), a) in enumerate(zip(triples, assignments)):
                worker = workers.get(a.key)
                if worker is None:
                    worker = workers[a.key] = new_worker(a.key, t, pc, alc, helmet_allowed, vest_allowed)
                if a.stitched:
                    worker.stitch(t, tid)
                elif tid not in worker.tracker_ids:
                    worker.tracker_ids.append(tid)
                worker.worker_id = a.worker_id
                if a.newly_confirmed:
                    worker.confirmed_t = t
                worker.observe_person(t, box, conf)
                observed_keys.add(a.key)
                others = [b for j, b in enumerate(all_boxes) if j != k and iou(b, box) <= dup_iou]
                row = {"frame": index, "t_sec": round(t, 3), "identity": a.key, "worker_id": a.worker_id or "",
                       "tracker_id": tid, "confirmed": worker.confirmed, "observed": True,
                       "person_conf": round(conf, 4), **_box_cells(box)}
                decisions = {}
                for kind, match, machine, region_key, gate_key in (
                    ("helmet", helmet_matches[k], worker.helmet, "head_observable", "head_gate"),
                    ("vest", vest_matches[k], worker.vest, "torso_observable", "torso_gate"),
                ):
                    observable, reasons, _ = assess(kind, box, conf, others, frame, oc, ac)
                    positive = match.status == ASSOCIATED and match.conf >= pc["min_positive_conf"]
                    eligible = observable and match.status not in (ASSOCIATED, AMBIGUOUS) and not match.near_miss
                    if positive:
                        (worker.helmet_detections if kind == "helmet" else worker.vest_detections).append(
                            (round(t, 3), round(match.conf, 4)))
                    if observable:
                        if kind == "helmet":
                            worker.head_observable_frames += 1
                        else:
                            worker.torso_observable_frames += 1
                    decision = machine.step(t, worker.start_t, Observation(positive, match.conf, eligible))
                    decisions[kind] = decision
                    seconds = worker.state_seconds.setdefault(kind, {})
                    seconds[machine.state] = seconds.get(machine.state, 0) + 1 / processed_fps
                    row.update({
                        f"{kind}_assoc": match.status, f"{kind}_conf": _round(match.conf, 4),
                        f"{kind}_near_miss": match.near_miss, region_key: observable,
                        gate_key: "ok" if observable else "; ".join(reasons),
                    })
                for kind, machine, gate, validated in (("helmet", worker.helmet, worker.helmet_alerts, helmet_validated),
                                                       ("vest", worker.vest, worker.vest_alerts, vest_validated)):
                    status = gate.update(t, active=machine.episode_active,
                                         episode_started=decisions[kind]["episode_started"],
                                         track_confirmed=worker.confirmed, camera_validated=validated,
                                         allow_unvalidated=allow_unvalidated, can_emit=machine.absence_evidence)
                    if status:
                        evidence_requests.append((worker, kind, machine, status, validated))
                    row[f"{kind}_alert"] = status
                if worker.confirmed:
                    point = floor_point(box, others, frame.shape, zone_cfg)
                    if zones:
                        row.update(zone_point_x=point[0] if point is not None else "",
                                   zone_point_y=point[1] if point is not None else "",
                                   zone_visibility="bbox_floor_proxy" if point is not None else "unknown")
                    zone_status={}
                    for zone in zones:
                        key = (worker.key, zone["id"])
                        state = zone_states.setdefault(key, ZoneState(zone_cfg))
                        distance = cv2.pointPolygonTest(zone["contour"], point, True) if point is not None else None
                        started, ended = state.step(t, distance)
                        zone_status[zone["id"]]={"active":state.active,"distance_px":_round(distance),
                                                 "observation":"unknown" if point is None else "known"}
                        if ended:
                            for event in zone_events:
                                if event["identity"] == worker.key and event["zone_id"] == zone["id"] and event["end_ts"] is None:
                                    event["end_ts"] = round(t, 3)
                        gate = zone_gates.setdefault(key, AlertGate(alc["cooldown_seconds"]))
                        status = gate.update(t, active=state.active, episode_started=started,
                                             track_confirmed=True, camera_validated=True, allow_unvalidated=False,
                                             can_emit=distance is not None and distance >= zone_cfg.get("boundary_margin_px", 5))
                        if status:
                            evidence_requests.append((worker, "zone_entry", state, status, True))
                            evidence_requests[-1] += (zone["id"],)
                    if zones:
                        row["zone_states"]=json.dumps(zone_status)
                _fill_state(row, worker)
                rows.append(row)
                if worker.confirmed:
                    hm, vm = helmet_matches[k], vest_matches[k]
                    draw_items.append((worker.worker_id, box, (worker.helmet.state, worker.helmet.absence_evidence),
                                       (worker.vest.state, worker.vest.absence_evidence),
                                       helmet_boxes[hm.ppe_index] if hm.status == ASSOCIATED else None,
                                       vest_boxes[vm.ppe_index] if vm.status == ASSOCIATED else None))
                else:
                    draw_items.append((f"cand {a.key}", box, None, None, None, None))
                    cand_rows.append({"frame": index, "t_sec": round(t, 3), "kind": "unconfirmed_track", "identity": a.key,
                                      "tracker_id": tid, "person_conf": round(conf, 4), **_box_cells(box),
                                      "helmet_assoc": helmet_matches[k].status, "vest_assoc": vest_matches[k].status})
            for j, (box, conf) in enumerate(extra):
                k = len(triples) + j
                draw_items.append(("det", box, None, None, None, None))
                cand_rows.append({"frame": index, "t_sec": round(t, 3), "kind": "untracked_detection", "identity": "",
                                  "tracker_id": "", "person_conf": round(conf, 4), **_box_cells(box),
                                  "helmet_assoc": helmet_matches[k].status, "vest_assoc": vest_matches[k].status})

            live_keys = {ident.key for ident in registry.by_tracker.values()}
            for key, state in zone_states.items():
                if key[0] not in observed_keys:
                    _, ended = state.step(t, None)
                    if ended:
                        for event in zone_events:
                            if event["identity"] == key[0] and event["zone_id"] == key[1] and event["end_ts"] is None:
                                event["end_ts"] = round(t, 3)
                    zone_gates[key].update(t, active=state.active, track_confirmed=False,
                                           camera_validated=True, allow_unvalidated=False)
            tracker_of = {ident.key: tid for tid, ident in registry.by_tracker.items()}
            for key, worker in list(workers.items()):
                if key in observed_keys:
                    continue
                stitchable = worker.confirmed and stitch_gap > 0 and t - worker.last_seen_t <= stitch_gap
                if key not in live_keys and not stitchable:
                    finished.append(worker)
                    del workers[key]
                    continue
                worker.lost_frames += 1
                worker.helmet.step(t, worker.start_t, None)
                worker.vest.step(t, worker.start_t, None)
                for kind, machine, gate, validated in (("helmet", worker.helmet, worker.helmet_alerts, helmet_validated),
                                                      ("vest", worker.vest, worker.vest_alerts, vest_validated)):
                    # Expire pending episodes while lost; never emit on a predicted box.
                    if not machine.episode_active:
                        gate.update(t, active=False, track_confirmed=worker.confirmed,
                                    camera_validated=validated, allow_unvalidated=allow_unvalidated)
                tid = tracker_of.get(key)
                box = lost_boxes.get(tid, worker.box)
                row = {"frame": index, "t_sec": round(t, 3), "identity": key, "worker_id": worker.worker_id or "",
                       "tracker_id": tid if tid is not None else "", "confirmed": worker.confirmed, "observed": False,
                       "person_conf": "", **_box_cells(box),
                       "helmet_assoc": "not_observed", "vest_assoc": "not_observed",
                       "head_gate": "interpolated (lost track)", "torso_gate": "interpolated (lost track)",
                       "head_observable": False, "torso_observable": False, "helmet_alert": "", "vest_alert": ""}
                _fill_state(row, worker)
                rows.append(row)
            now = time.perf_counter(); stage["rules"] = now - mark; mark = now

            canvas = frame  # the reader hands over a fresh array per frame, so draw in place
            draw_zones(canvas, zones)
            render.draw_header(canvas, camera["id"], t, helmet_validated, vest_validated)
            for label, box, helmet_state, vest_state, hbox, vbox in draw_items:
                if helmet_state is None:
                    render.draw_candidate(canvas, label, box)
                else:
                    render.draw_worker(canvas, label, box, helmet_state, vest_state, hbox, vbox)
            now = time.perf_counter(); stage["render"] = now - mark; mark = now

            images, clip_requests = [], []
            for request in evidence_requests:
                worker, kind, machine, status, validated = request[:5]
                zone_id = request[5] if len(request) > 5 else None
                event, path, image = _event(worker, kind, machine, status, validated, camera, clip, index, t, fps,
                                            canvas, out_dir, zone_id)
                event["processing_elapsed_sec"] = round(time.perf_counter() - wall_start, 4)
                event["wall_alert_delay_sec"] = round(time.perf_counter() - frame_capture_times[event["first_frame"]], 4)
                event["wall_delay_basis"] = ("paced source-frame schedule to alert; includes local decode and queueing" if realtime else
                                             "first supporting frame retrieved by local reader to alert; includes queueing, excludes camera/network capture")
                diagnostics_handle.write(json.dumps(event) + "\n")
                diagnostics_handle.flush()
                if status == EMITTED:
                    events_handle.write(json.dumps(event) + "\n")
                    events_handle.flush()
                    events.append(event)
                    clip_requests.append((event["clip_path"], t))
                    if kind == "zone_entry":
                        zone_events.append(event)
                    else:
                        event_episodes.append((event, machine.episodes[-1], worker))
                images.append((path, image))
            due_snapshots = {f for f in snapshot_frames if f <= index}
            if due_snapshots:
                images.append((out_dir / "snapshots" / f"t{t:05.2f}s.jpg", canvas))
                snapshot_frames -= due_snapshots
            writer.q.put((t, canvas, rows, cand_rows, images, clip_requests))
            now = time.perf_counter(); stage["write"] = now - mark
            stage["total"] = now - start
            stage["frame_age"] = now - captured_at
            stage["decode"] = decode_ms / 1000
            for key, value in stage.items():
                timings[key].append(value * 1000)
            frames += 1
            if frames % 75 == 0:
                print(f"  {camera['scene']} frame {frames}/{n_frames}", flush=True)

    finally:
        reader.stop.set()
        reader.join()
        writer.q.put(None)
        writer.join()
        cap.release()
        video.release()
        log_handle.close()
        cand_handle.close()
        events_handle.close()
        diagnostics_handle.close()
    wall_s = time.perf_counter() - wall_start
    if reader.error is not None:
        raise reader.error
    if writer.error is not None:
        raise writer.error
    finished.extend(workers.values())
    if not frames:
        raise ValueError(f"video contains no decodable frames: {clip}")
    last_t = (reader.source_frames - 1) / fps
    for event, episode, worker in event_episodes:
        event["end_ts"] = round(episode[1] if episode[1] is not None else worker.last_seen_t, 3)
        event["end_ts"] = max(event["start_ts"], event["end_ts"])
        event["open_at_end"] = episode[1] is None
    for event in zone_events:
        if event["end_ts"] is None:
            event["end_ts"], event["open_at_end"] = round(last_t, 3), True
    (out_dir / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")

    ordered = sorted(finished, key=lambda w: (w.worker_id is None, w.worker_id or 0, w.key))
    summaries = [_worker_summary(w) for w in ordered]
    (out_dir / "workers.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    latency = {
        key: {"median_ms": round(statistics.median(v), 2), "p95_ms": round(float(np.percentile(v, 95)), 2),
              "mean_ms": round(statistics.fmean(v), 2)}
        for key, v in timings.items() if v
    }
    if writer.ms:
        latency["writer_thread"] = {"median_ms": round(statistics.median(writer.ms), 2),
                                    "p95_ms": round(float(np.percentile(writer.ms, 95)), 2),
                                    "mean_ms": round(statistics.fmean(writer.ms), 2)}
    record = {
        "clip": clip.name,
        "camera_id": camera["id"],
        "scene": camera["scene"],
        "backend": detector.backend,
        "fps_video": fps,
        "processed_fps_video": processed_fps,
        "input_mode": "paced_file" if realtime else "offline_file",
        "reader_queue_max_depth": reader.max_depth,
        "frame_age_ms": latency["frame_age"],
        "frame_age_basis": "paced source schedule to main-thread completion" if realtime else "local retrieval to main-thread completion",
        "source_frames": reader.source_frames,
        "detector_calls": detector_calls,
        "detection_rejections": dict(rejected_counts),
        "zones_loaded": len(zones),
        "hardware": {"platform": platform.platform(), "processor": platform.processor(), "device": mc.get("device")},
        "input_resolution": [width, height],
        "detector_input": mc.get("onnx_imgsz") if detector.backend == "onnx" else mc["imgsz"],
        "frames": frames,
        "helmet_absence_validated": helmet_validated,
        "vest_absence_validated": vest_validated,
        "allow_unvalidated_camera_absence_alerts": allow_unvalidated,
        "tracker_args": tracker.args,
        "identities": len(summaries),
        "confirmed_workers": sum(1 for s in summaries if s["confirmed"]),
        "unconfirmed_candidates": sum(1 for s in summaries if not s["confirmed"]),
        "tracker_ids_used": len({tid for s in summaries for tid in s["tracker_ids"]}),
        "stitches": registry.stitches,
        "jump_guard_splits": registry.splits,
        "untracked_person_detections": untracked_person_dets,
        "untracked_person_detections_non_duplicate": untracked_kept,
        "events": len(events),
        "events_emitted": sum(1 for e in events if e["alert_status"] == EMITTED),
        "helmet_association_rate": _association_rate(out_dir / "worker_log.csv", "helmet", "head_observable"),
        "vest_association_rate": _association_rate(out_dir / "worker_log.csv", "vest", "torso_observable"),
        "latency_ms": latency,
        "wall_seconds": round(wall_s, 2),
        "pipeline_fps_end_to_end": round(frames / wall_s, 2) if wall_s else None,
        "video_seconds": round(reader.source_frames / fps, 3),
        "realtime_factor": round((reader.source_frames / fps) / wall_s, 3),
        "median_alert_delay_sec": _median([e["alert_delay_sec"] for e in events]),
        "inference_only_fps": round(1000 / latency["inference"]["median_ms"], 2) if latency else None,
        "annotated_video": str(video_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    (out_dir / "timeline.md").write_text(timeline_markdown(out_dir / "worker_log.csv", camera["id"]), encoding="utf-8")
    return record


def _fill_state(row: dict, worker) -> None:
    for kind, machine in (("helmet", worker.helmet), ("vest", worker.vest)):
        row[f"{kind}_state"] = machine.state
        row[f"{kind}_absence_evidence"] = machine.absence_evidence
        row[f"{kind}_eligible_negatives"] = machine.eligible_negative_count
        row[f"{kind}_reason"] = machine.reason
        row[f"last_{kind}_t"] = _round(machine.last_positive_t)


def _event(worker, kind, machine, status, validated, camera, clip, index, t, fps, canvas, out_dir, zone_id=None):
    label = f"id{worker.worker_id:02d}" if worker.confirmed else f"cand{worker.key}"
    onset = machine.start_t if kind == "zone_entry" else machine.episode_start_t
    onset = t if onset is None else onset
    episode_number = machine.episode_number if kind == "zone_entry" else len(machine.episodes)
    event_id = uuid.uuid5(uuid.NAMESPACE_URL, f"{camera['id']}/{clip.name}/{worker.key}/{kind}/{zone_id}/{onset:.6f}/{episode_number}").hex
    # Multiple decisions (candidate -> confirmed, or cooldown retry) may belong
    # to one episode. Keep a unique snapshot for each decision, without overwrite.
    decision_id = uuid.uuid5(uuid.NAMESPACE_URL, f"{event_id}/{status}/{index}").hex
    path = out_dir / "evidence" / f"{decision_id}.jpg"
    image = canvas.copy()
    x1, y1, x2, y2 = (int(v) for v in worker.box)
    cv2.rectangle(image, (x1 - 6, y1 - 6), (x2 + 6, y2 + 6), (255, 0, 255), 3)
    detections = worker.helmet_detections if kind == "helmet" else worker.vest_detections
    window = machine.cfg.get("suspected_absence_seconds", machine.cfg.get("entry_seconds", .6))
    recent_person = [c for ts, c in worker.person_confs if ts >= t - window - 0.5 / fps]
    event = {
        "schema_version": 1,
        "event_id": event_id,
        "decision_id": decision_id,
        "episode_number": episode_number,
        "type": "zone_entry" if kind == "zone_entry" else f"no_{kind}",
        "track_id": worker.worker_id,
        "start_ts": round(onset, 3),
        "end_ts": None,
        "first_frame": round(onset * fps),
        "alert_ts": round(t, 3),
        "alert_delay_sec": round(t - onset, 3),
        "confidence": _median(recent_person),
        "confidence_basis": "person detection confidence; not a calibrated violation probability",
        "bbox": list(worker.box),
        "zone_id": zone_id,
        "snapshot_path": str(path.resolve()),
        "clip_path": str((out_dir / "clips" / f"{event_id}.mp4").resolve()) if status == EMITTED else None,
        "camera_id": camera["id"],
        "video": clip.name,
        "frame": index,
        "video_timestamp_sec": round(t, 3),
        "worker_id": worker.worker_id,
        "identity": worker.key,
        "tracker_ids": list(worker.tracker_ids),
        "violation_type": "zone_entry" if kind == "zone_entry" else f"suspected_missing_{kind}",
        "ppe_state": "INSIDE" if kind == "zone_entry" else machine.state,
        "absence_evidence_met": None if kind == "zone_entry" else machine.absence_evidence,
        "supporting_observation_count": None if kind == "zone_entry" else machine.eligible_negative_count,
        "observation_window_sec": [round(max(worker.start_t, t - window), 3), round(t, 3)],
        "detection_confidence_summary": {
            "person_conf_median": _median(recent_person),
            "person_conf_min": round(min(recent_person), 4) if recent_person else None,
            "person_detections_in_window": len(recent_person),
            f"{kind}_detections_in_window": sum(1 for dt, _ in detections if dt >= t - window),
            f"last_{kind}_detection_sec": _round(getattr(machine, "last_positive_t", None)),
            f"last_{kind}_conf": _round(getattr(machine, "last_positive_conf", None), 4),
        },
        "evidence_frame": str(path),
        "camera_validated_for_absence": validated,
        "alert_status": status,
        "note": "floor point is bbox bottom-centre; machinery may obscure feet" if kind == "zone_entry" else
                "SH17 has no no-helmet/no-vest class; this is suspected absence, not visually confirmed absence",
        "reason": "floor contact proxy inside polygon for entry dwell time" if kind == "zone_entry" else machine.reason,
    }
    return event, path, image


def _worker_summary(worker) -> dict:
    confs = [c for _, c in worker.person_confs]
    out = {
        "worker_id": worker.worker_id,
        "identity": worker.key,
        "confirmed": worker.confirmed,
        "tracker_ids": worker.tracker_ids,
        "stitched_at_sec": worker.stitches,
        "track_start_sec": round(worker.start_t, 3),
        "confirmed_at_sec": _round(worker.confirmed_t),
        "last_observed_sec": round(worker.last_seen_t, 3),
        "observed_frames": worker.observed_frames,
        "lost_frames": worker.lost_frames,
        "person_conf_median": _median(confs),
        "person_conf_last": round(worker.person_conf, 4),
        "last_bbox": worker.box,
        "head_observable_frames": worker.head_observable_frames,
        "torso_observable_frames": worker.torso_observable_frames,
    }
    for kind, machine, gate, dets in (("helmet", worker.helmet, worker.helmet_alerts, worker.helmet_detections),
                                      ("vest", worker.vest, worker.vest_alerts, worker.vest_detections)):
        kconfs = [c for _, c in dets]
        out[kind] = {
            "detections": len(dets),
            "conf_median": _median(kconfs),
            "conf_max": max(kconfs) if kconfs else None,
            "final_state": machine.state,
            "final_reason": machine.reason,
            "last_detection_sec": _round(machine.last_positive_t),
            "last_confirmed_sec": _round(machine.last_confirmed_t),
            "state_seconds_observed": {k: round(v, 2) for k, v in worker.state_seconds.get(kind, {}).items()},
            "violation_episodes": machine.episodes,
            "alerts": gate.history,
        }
    return out


def _association_rate(log_path: Path, kind: str, observable_key: str) -> dict:
    """Share of confirmed, genuinely observed, gate-passing frames where this PPE kind was associated."""
    total = hit = 0
    with open(log_path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["observed"] != "True" or row["confirmed"] != "True" or row[observable_key] != "True":
                continue
            total += 1
            hit += row[f"{kind}_assoc"] == ASSOCIATED
    return {"observable_frames": total, "associated_frames": hit, "rate": round(hit / total, 3) if total else None}


def timeline_markdown(log_path: Path, camera_id: str) -> str:
    """Compress the per-frame worker log into constant-state segments per identity."""
    by_identity: dict = {}
    with open(log_path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            by_identity.setdefault(int(row["identity"]), []).append(row)
    lines = [f"# Worker timelines: {camera_id}", "",
             "Segments group consecutive frames with the same observed flag and helmet / vest state;",
             "ON and ON_HELD are grouped, with the number of held frames in brackets.",
             "`H det` / `V det` = frames in the segment with an associated helmet / vest box.",
             "`(evidence)` = absence evidence met; on an unvalidated camera the state stays UNKNOWN.",
             "Rows before a worker's confirmation belong to the same identity while it was a candidate.", ""]
    confirmed = {key: rows for key, rows in by_identity.items() if any(r["worker_id"] for r in rows)}
    order = sorted(confirmed, key=lambda key: int(next(r["worker_id"] for r in confirmed[key] if r["worker_id"])))
    for key in order:
        rows = confirmed[key]
        wid = int(next(r["worker_id"] for r in rows if r["worker_id"]))
        tids = sorted({int(r["tracker_id"]) for r in rows if r["tracker_id"]})
        lines += [f"## Worker {wid:02d} (identity {key}, ByteTrack ids {tids})", "",
                  "| time (s) | frames | observed | H det | V det | helmet state | vest state | alert | reason (end of segment) |",
                  "|---|---|---|---|---|---|---|---|---|"]

        def bucket(state):
            return state.replace("_ON_HELD", "_ON")

        def shown(rs, kind):
            state = bucket(rs[-1][f"{kind}_state"])
            held = sum(r[f"{kind}_state"].endswith("_HELD") for r in rs)
            if state.endswith("_ON"):
                state += f" ({held} held)" if held else ""
            return state + (" (evidence)" if rs[-1][f"{kind}_absence_evidence"] == "True" else "")

        segments = []
        for r in rows:
            seg_key = (r["observed"], r["tracker_id"], bucket(r["helmet_state"]), bucket(r["vest_state"]),
                       r["helmet_absence_evidence"], r["vest_absence_evidence"])
            if segments and segments[-1]["key"] == seg_key and not (r["helmet_alert"] or r["vest_alert"]):
                segments[-1]["rows"].append(r)
            else:
                segments.append({"key": seg_key, "rows": [r]})
        for seg in segments:
            rs = seg["rows"]
            alerts = "; ".join(f"{k}: {r[f'{k}_alert']}" for r in rs for k in ("helmet", "vest") if r[f"{k}_alert"])
            hstate, vstate = shown(rs, "helmet"), shown(rs, "vest")
            reason = f"H: {rs[-1]['helmet_reason']} / V: {rs[-1]['vest_reason']}"
            if rs[-1]["observed"] == "True" and rs[-1]["head_gate"] != "ok":
                reason += f" / head gate: {rs[-1]['head_gate']}"
            lines.append(
                f"| {rs[0]['t_sec']}-{rs[-1]['t_sec']} | {len(rs)} | {rs[0]['observed']} | "
                f"{sum(r['helmet_assoc'] == ASSOCIATED for r in rs)} | {sum(r['vest_assoc'] == ASSOCIATED for r in rs)} | "
                f"{hstate} | {vstate} | {alerts or '-'} | {reason} |"
            )
        lines.append("")
    candidates = {key: rows for key, rows in by_identity.items() if key not in confirmed}
    if candidates:
        lines += ["## Unconfirmed candidates", "", "| identity | ByteTrack ids | first (s) | last (s) | observed frames | median conf |",
                  "|---|---|---|---|---|---|"]
        for key, rows in sorted(candidates.items()):
            confs = [float(r["person_conf"]) for r in rows if r["person_conf"]]
            tids = sorted({int(r["tracker_id"]) for r in rows if r["tracker_id"]})
            lines.append(f"| {key} | {tids} | {rows[0]['t_sec']} | {rows[-1]['t_sec']} | {len(confs)} | "
                         f"{_median(confs)} |")
        lines.append("")
    return "\n".join(lines)
