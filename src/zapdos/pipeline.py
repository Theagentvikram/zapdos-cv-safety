"""decode -> detect -> track -> helmet -> rules (PPE + zones + events) -> media, with per-stage timing."""
import json
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np

from .detect import PPEDetector, available_device
from .events import EventEngine, Observation
from .helmet import HelmetClassifier
from .media import MediaRecorder, draw
from .ppe import decide, helmet_threshold, measure
from .track import PersonTracker
from .view import ViewCheck
from .zones import ZoneSet

STAGES = ("decode", "detect", "track", "helmet", "rules", "media")


def open_video(video):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    return cap, src_fps, size


def sample_frames(cap, src_fps, fps):
    """Yields (frame_idx, ts, frame, decode_ms) at `fps`. Every source frame is grabbed (H.264 needs
    them to decode the next one) but only the sampled ones are converted to images; decode_ms covers
    both."""
    frame_idx, next_ts, acc = -1, 0.0, 0.0
    while True:
        t0 = time.perf_counter()
        if not cap.grab():
            return
        frame_idx += 1
        ts = frame_idx / src_fps
        if ts + 1e-6 < next_ts:
            acc += time.perf_counter() - t0
            continue
        next_ts += 1.0 / fps
        _, frame = cap.retrieve()
        ms = (acc + time.perf_counter() - t0) * 1000
        acc = 0.0
        yield frame_idx, ts, frame, ms


def build_models(cfg):
    pcfg = cfg["ppe"]
    cfg["detector"]["device"] = available_device(cfg["detector"]["device"])
    detector = PPEDetector(cfg["detector"])
    helmet_clf = (HelmetClassifier(pcfg["classifier"]["weights"], cfg["detector"]["device"])
                  if pcfg["helmet_method"] == "classifier" else None)
    return detector, helmet_clf


def observe(tracks, meas, readings, zones):
    """Per-person observations for the event engine, plus the overlay records for drawing."""
    observations, people = [], []
    for (x1, y1, x2, y2, tid, conf), m, reading in zip(tracks, meas, readings):
        tid, box = int(tid), [float(x1), float(y1), float(x2), float(y2)]
        zone_read, foot = zones.read(tid, box, conf)
        r = {(k, None): v for k, v in reading.items()}
        r.update({("zone_entry", z): v for z, v in zone_read.items()})
        hold = frozenset() if zones.confirmed(tid) else frozenset(("zone_entry", z) for z in zone_read)
        observations.append(Observation(tid, box, r, hold))
        flags = {k: v[0] for k, v in reading.items()}
        inside = [v[0] for v in zone_read.values() if v[0] is not None]
        flags["zone_entry"] = any(inside) if inside and not hold else None
        people.append({"tid": tid, "box": box, "flags": flags, "foot": foot, "head": m["head"]})
    return observations, people


class Camera:
    """One camera's state: tracker, zones, event engine, media and stage timings. The models are passed
    in so that several cameras can share one copy (multi.py)."""

    def __init__(self, video, zones_path, out_root, cfg, models, camera_id=None, save_video=False):
        self.cap, self.src_fps, self.size = open_video(video)
        self.fps = min(cfg["sampling"]["process_fps"], self.src_fps)
        self.zones = ZoneSet(zones_path, self.size, cfg["zones"], self.fps)
        self.camera_id = camera_id or self.zones.camera_id or Path(video).stem
        self.out = Path(out_root) / self.camera_id
        (self.out / "snapshots").mkdir(parents=True, exist_ok=True)
        (self.out / "clips").mkdir(exist_ok=True)
        for old in [*self.out.glob("snapshots/*.jpg"), *self.out.glob("clips/*.mp4")]:
            old.unlink()  # a previous run's alerts; events.jsonl and perf.json are rewritten too

        self.cfg, self.pcfg = cfg, cfg["ppe"]
        self.detector, self.helmet_clf = models
        self.tracker = PersonTracker(cfg["tracker"], self.fps)
        self.thr = helmet_threshold(self.pcfg, self.helmet_clf)
        self.engine = EventEngine(cfg["events"], self.camera_id, self.fps)
        self.media = MediaRecorder(self.out, cfg["output"], self.fps, self.zones.polygons, self.size, save_video)
        corners = [p for poly in self.zones.polygons.values() for p in poly.reshape(-1, 2)]
        self.view = ViewCheck(cfg["view"], corners, self.size, self.zones.reference) if corners else None
        self.moved_ev = None
        self.ms = {s: [] for s in STAGES}
        self.n_src = 0
        self.tag = ""  # alert-line prefix; the multi-camera runner sets the camera id
        self.live = open(self.out / "events.jsonl", "w")  # one line the moment each alert fires (end_ts = alert time)

    def frames(self):
        return sample_frames(self.cap, self.src_fps, self.fps)

    def step(self, frame_idx, ts, frame, decode_ms, det=None, detect_ms=None, quiet=False):
        """Process one sampled frame; `det` may come from a shared detector thread. Returns the overlay
        and this frame's time in ms from decode to media (perf.json's pipeline_ms_per_frame leaves
        media out)."""
        t1 = time.perf_counter()
        if det is None:
            det = self.detector(frame)
            detect_ms = (time.perf_counter() - t1) * 1000
        t2 = time.perf_counter()
        tracks = self.tracker(det.persons)
        t3 = time.perf_counter()
        boxes = tracks[:, :4]
        meas = measure(boxes, det, self.pcfg, frame, self.helmet_clf)
        t3b = time.perf_counter()
        alerts = self._check_view(frame_idx, ts, frame)
        observations, people = observe(tracks, meas, decide(meas, boxes, self.pcfg, self.thr), self.zones)
        alerts += self.engine.step(frame_idx, ts, observations)
        t4 = time.perf_counter()

        hidden = self.tracker.hidden(int(round(self.cfg["output"]["hidden_s"] * self.fps)))
        overlay = {"ts": ts, "people": people + [{"tid": int(h[4]), "box": list(h[:4]), "flags": {}, "hidden": True}
                                                 for h in hidden]}
        if self.zones.suspended:
            overlay["banner"] = "CAMERA MOVED: zones paused"
        for ev in alerts:
            self.media.on_alert(ev, frame, overlay)
            self.live.write(json.dumps(ev.to_json()) + "\n")
            self.live.flush()
            if not quiet:
                who = f"track #{ev.track_id}" if ev.track_id >= 0 else ev.note
                zone = f" zone={ev.zone_id}" if ev.zone_id else ""
                print(f"[ALERT] {self.tag}t={ev.alert_ts:6.2f}s {ev.type:<10} "
                      f"{who}{zone} (started {ev.start_ts:.2f}s, delay {ev.delay_s:.2f}s, "
                      f"conf {ev.confidence:.2f})")
        self.media.push(frame, overlay)
        t5 = time.perf_counter()
        self.n_src = frame_idx + 1
        self.ms["decode"].append(decode_ms)
        self.ms["detect"].append(detect_ms)
        for s, a, b in (("track", t2, t3), ("helmet", t3, t3b), ("rules", t3b, t4), ("media", t4, t5)):
            self.ms[s].append((b - a) * 1000)
        return overlay, decode_ms + detect_ms + (t5 - t2) * 1000

    def _check_view(self, frame_idx, ts, frame):
        """One camera_moved event per stretch of time the view doesn't match the zones' reference; zones
        read "can't tell" meanwhile."""
        moved = self.view.update(ts, frame) if self.view else None
        if self.moved_ev is not None:
            self.moved_ev.end_ts = ts
        if moved is None:
            return []
        self.zones.suspended = moved
        if not moved:
            self.moved_ev = None
            return []
        shift = self.view.shift
        note = ("view no longer matches the zone reference" if math.isinf(shift) else
                f"zone corners moved {shift * self.size[0]:.0f} px") + "; zone alerts paused"
        self.moved_ev = self.engine.camera_event("camera_moved", self.view.since, round(self.view.since * self.src_fps),
                                                 ts, frame_idx, self.size, note)
        return [self.moved_ev]

    def close(self, wall0):
        """Finish clips, write the final events.jsonl and perf.json; returns (events, perf)."""
        self.media.close()
        if self.view:
            self.view.pool.shutdown(wait=True)
        wall = time.perf_counter() - wall0
        self.cap.release()
        self.live.close()
        with open(self.out / "events.jsonl.tmp", "w") as f:  # same lines, with each episode's final end_ts
            for ev in self.engine.events:
                f.write(json.dumps(ev.to_json()) + "\n")
        os.replace(self.out / "events.jsonl.tmp", self.out / "events.jsonl")
        perf = _perf(self.ms, wall, self.n_src, self.src_fps, self.size, self.fps, self.cfg["detector"],
                     self.detector, self.helmet_clf)
        with open(self.out / "perf.json", "w") as f:
            json.dump(perf, f, indent=2)
        return self.engine.events, perf


def run(video, zones_path, out_root, cfg, camera_id=None, save_video=False, models=None, show=False):
    cam = Camera(video, zones_path, out_root, cfg, models or build_models(cfg), camera_id, save_video)
    wall0 = time.perf_counter()
    for frame_idx, ts, frame, decode_ms in cam.frames():
        overlay, frame_ms = cam.step(frame_idx, ts, frame, decode_ms)
        if show and not _show(frame, overlay, cam.zones.polygons, cam.media.log, cfg["output"]["video_width"],
                              frame_ms, ts, wall0, f"{cam.fps:g} of {cam.src_fps:g} fps"):
            break
    if show:
        cv2.destroyAllWindows()
    events, perf = cam.close(wall0)
    _print_summary(cam.camera_id, events, perf, cam.out)
    return events, perf


def _show(frame, overlay, zones, log, width, frame_ms, ts, wall0, rate):
    """Live view at the video's own pace (not included in the stage timings). False = quit."""
    scale = width / frame.shape[1]
    img = draw(cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), overlay, zones,
               scale, log=log)
    hud = f"frame {frame_ms:.0f} ms incl. media = {1000 / frame_ms:.0f} fps capacity | processing {rate} | q quits"
    cv2.putText(img, hud, (10, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
    cv2.putText(img, hud, (10, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    ahead_s = ts - (time.perf_counter() - wall0)
    if ahead_s > 0:
        time.sleep(ahead_s)
    cv2.imshow("zapdos: live", img)
    return cv2.waitKey(1) & 0xFF != ord("q")


def _perf(ms, wall, n_src, src_fps, size, fps, dcfg, detector, helmet_clf):
    n = len(ms["detect"])
    stages = {s: {"mean_ms": round(float(np.mean(v)), 1), "p95_ms": round(float(np.percentile(v, 95)), 1)}
              for s, v in ms.items() if v}
    per_frame = sum(stages[s]["mean_ms"] for s in ("decode", "detect", "track", "helmet", "rules"))
    det_name = Path(dcfg["coreml"]).name if detector.backend.startswith("coreml") else Path(dcfg["weights"]).name
    return {
        "input_res": f"{size[0]}x{size[1]}", "source_fps": src_fps, "process_fps": fps,
        "detector": f"{det_name} on {detector.backend}",
        "helmet_classifier": helmet_clf.backend if helmet_clf else None,
        "frames_decoded": n_src, "frames_processed": n, "wall_s": round(wall, 2),
        "video_s": round(n_src / src_fps, 2),
        "throughput_fps": round(n / wall, 1),            # processed frames per wall-second, all stages
        "pipeline_ms_per_frame": round(per_frame, 1),     # decode to rules; media (snapshots, clips) is in stages
        "streams_at_process_fps": round(1000 / per_frame / fps, 2),
        "stages": stages,
    }


def _print_summary(camera_id, events, perf, out):
    s = perf["stages"]
    print(f"\n{camera_id}: {len(events)} alerts -> {out / 'events.jsonl'}")
    print(f"  {perf['frames_processed']} frames at {perf['process_fps']} fps from {perf['video_s']} s of video "
          f"in {perf['wall_s']} s  ({perf['throughput_fps']} fps end to end)")
    print("  ms/frame  " + "  ".join(f"{k} {v['mean_ms']}" for k, v in s.items()))
    print(f"  pipeline {perf['pipeline_ms_per_frame']} ms/frame -> "
          f"{perf['streams_at_process_fps']} streams at {perf['process_fps']} fps")
