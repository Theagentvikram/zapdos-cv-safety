"""Drawing, alert snapshots and short alert clips.

Clips come from a ring buffer of recent downscaled frames, as they would on a live stream (there is
no file to seek back into): pre-roll from the buffer, post-roll from the frames that follow. Drawing
and encoding a finished clip (~60 frames) runs on a background thread so it never stalls the frame
loop; OpenCV releases the GIL while it works.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

RED, GREEN, GREY, AMBER, WHITE = (40, 40, 230), (60, 200, 60), (160, 160, 160), (0, 190, 255), (255, 255, 255)
SHORT = {"no_helmet": "NO HELMET", "no_vest": "NO VEST", "zone_entry": "IN ZONE", "camera_moved": "CAMERA MOVED"}


def draw(frame, overlay, zones, scale=1.0, highlight=None, log=None):
    """overlay: {"ts": float, "people": [person], "banner": optional text}. A person is
    {"tid", "box", "flags": {type: value}, "foot": (point, feet_hidden), "head": box}, or
    {"tid", "box", "flags": {}, "hidden": True} for a track that is lost for a moment."""
    img = frame.copy()
    t = max(1, round(2 * scale))  # line/font weight: 2 at full 1080p, 1 on downscaled clips
    for zid, poly in zones.items():
        pts = (poly * scale).astype(np.int32)
        cv2.polylines(img, [pts], True, AMBER, 2 * t)
        x, y = pts[0, 0]
        cv2.putText(img, zid, (int(x), int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * t, AMBER, t)
    for p in overlay["people"]:
        x1, y1, x2, y2 = (int(v * scale) for v in p["box"])
        if p.get("hidden"):  # lost for a moment; the tracker still holds the ID at its predicted box
            _dashed_rect(img, (x1, y1), (x2, y2), WHITE, t + 1)
            cv2.putText(img, f"#{p['tid']} hidden", (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * t,
                        WHITE, t + 1)
            continue
        flags = p["flags"]
        bad = [SHORT.get(k, k) for k, v in flags.items() if v]
        color = RED if bad else (GREEN if any(v is False for v in flags.values()) else GREY)
        thick = 3 * t if p["tid"] == highlight else t
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        label = f"#{p['tid']} " + " ".join(bad)
        cv2.putText(img, label, (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * t, color, t)
        if p.get("head") is not None:  # the region the helmet decision looked at
            hx1, hy1, hx2, hy2 = (int(v * scale) for v in p["head"])
            hc = RED if flags.get("no_helmet") else (GREEN if flags.get("no_helmet") is False else GREY)
            cv2.rectangle(img, (hx1, hy1), (hx2, hy2), hc, 1)
        if p.get("foot"):
            (fx, fy), hidden = p["foot"]
            cv2.circle(img, (int(fx * scale), int(fy * scale)), 3 * t, AMBER if hidden else WHITE, -1)
    cv2.putText(img, f"t={overlay['ts']:.1f}s", (10, 24 * t), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * t, WHITE, 2 * t)
    if overlay.get("banner"):
        cv2.putText(img, overlay["banner"], (10, 50 * t), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * t, RED, 2 * t)
    if log:
        _draw_log(img, log, t)
    return img


def _dashed_rect(img, p1, p2, color, t, dash=8):
    (x1, y1), (x2, y2) = p1, p2
    for a, b in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
        n = max(1, int(np.hypot(b[0] - a[0], b[1] - a[1]) // dash))
        for i in range(0, n, 2):
            s, e = i / n, min(i + 1, n) / n
            cv2.line(img, (int(a[0] + (b[0] - a[0]) * s), int(a[1] + (b[1] - a[1]) * s)),
                     (int(a[0] + (b[0] - a[0]) * e), int(a[1] + (b[1] - a[1]) * e)), color, t)


def _draw_log(img, log, t):
    lines = log[-6:]
    h = (len(lines) + 1) * 20 * t
    x0 = img.shape[1] - 420 * t
    cv2.rectangle(img, (x0, 0), (img.shape[1], h + 10), (20, 20, 20), -1)
    cv2.putText(img, "ALERTS", (x0 + 8, 18 * t), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * t, AMBER, t)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (x0 + 8, (i + 2) * 20 * t), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * t, WHITE, t)


def _writer(path, fps, size):
    for codec in ("avc1", "mp4v"):
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if w.isOpened():
            return w
    raise RuntimeError(f"no usable mp4 codec for {path}")


class MediaRecorder:
    def __init__(self, out_dir, cfg, fps, zones, frame_size, save_video=False):
        self.out = out_dir
        self.cfg = cfg
        self.fps = fps
        self.zones = zones
        self.clip_scale = cfg["clip_width"] / frame_size[0]
        keep = cfg["clip_pre_s"] + cfg["buffer_extra_s"]
        self.buf = deque(maxlen=int(keep * fps) + 1)   # (ts, small_frame, overlay)
        self.pending = []                               # [event, frames, until_ts]
        self.pool = ThreadPoolExecutor(1)  # draws and encodes finished clips
        self.jobs = []
        self.log = []
        self.video = None
        if save_video:
            self.video_scale = cfg["video_width"] / frame_size[0]
            size = (cfg["video_width"], int(round(frame_size[1] * self.video_scale)))
            self.video = _writer(out_dir / "annotated.mp4", fps, size)

    def push(self, frame, overlay):
        small = cv2.resize(frame, None, fx=self.clip_scale, fy=self.clip_scale, interpolation=cv2.INTER_AREA)
        self.buf.append((overlay["ts"], small, overlay))
        for item in self.pending:
            item[1].append((overlay["ts"], small, overlay))
        self._flush_ready(overlay["ts"])
        if self.video is not None:
            self.video.write(draw(frame if self.video_scale == 1 else
                                  cv2.resize(frame, None, fx=self.video_scale, fy=self.video_scale),
                                  overlay, self.zones, self.video_scale, log=self.log))

    def on_alert(self, ev, frame, overlay):
        snap = self.out / "snapshots" / f"{ev.event_id}.jpg"
        cv2.imwrite(str(snap), draw(frame, overlay, self.zones, 1.0, highlight=ev.track_id),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        ev.snapshot = f"snapshots/{snap.name}"
        ev.clip = f"clips/{ev.event_id}.mp4"
        pre = [f for f in self.buf if f[0] >= ev.start_ts - self.cfg["clip_pre_s"]]
        # the clip runs from clip_pre_s before the onset to clip_post_s after the alert
        self.pending.append([ev, pre, ev.alert_ts + self.cfg["clip_post_s"]])
        zone = f" {ev.zone_id}" if ev.zone_id else ""
        who = f"#{ev.track_id} " if ev.track_id >= 0 else ""
        self.log.append(f"{ev.alert_ts:5.1f}s {who}{SHORT[ev.type]}{zone}")

    def _flush_ready(self, ts, force=False):
        for item in [i for i in self.pending if force or ts >= i[2]]:
            self.pending.remove(item)
            self.jobs.append(self.pool.submit(self._write_clip, item[0], item[1]))

    def _write_clip(self, ev, frames):
        h, w = frames[0][1].shape[:2]
        vw = _writer(self.out / ev.clip, self.fps, (w, h))
        for _, small, ov in frames:
            vw.write(draw(small, ov, self.zones, self.clip_scale, highlight=ev.track_id))
        vw.release()

    def close(self):
        self._flush_ready(0, force=True)
        self.pool.shutdown(wait=True)
        for job in self.jobs:
            job.result()  # surface any write error
        if self.video is not None:
            self.video.release()
