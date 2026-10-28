"""Check completed alert schemas and that snapshots/clips can actually be opened."""
import argparse
import json
from pathlib import Path
import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="completed run directory")
    args = parser.parse_args()
    root = Path(args.root)
    run = json.loads((root / "run_summary.json").read_text())
    count = 0
    required = {"event_id", "camera_id", "type", "track_id", "start_ts", "end_ts", "first_frame",
                "confidence", "bbox", "snapshot_path", "clip_path"}
    for summary in run["videos"]:
        out = Path(summary["annotated_video"]).parent
        events = [json.loads(line) for line in (out / "events.jsonl").read_text().splitlines() if line]
        if len(events) != summary["events_emitted"]:
            raise ValueError(f"event count mismatch: {out}")
        if len({e["event_id"] for e in events}) != len(events):
            raise ValueError(f"duplicate event ids: {out}")
        for event in events:
            if required - event.keys() or event["alert_status"] != "emitted":
                raise ValueError(f"invalid alert schema: {event}")
            if not event["start_ts"] <= event["alert_ts"] <= event["end_ts"] + 1e-6:
                raise ValueError(f"invalid episode times: {event['event_id']}")
            image = cv2.imread(event["snapshot_path"])
            if image is None:
                raise ValueError(f"unreadable snapshot: {event['snapshot_path']}")
            cap = cv2.VideoCapture(event["clip_path"])
            try:
                fps = cap.get(cv2.CAP_PROP_FPS)
                frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                ok, _ = cap.read()
                if not ok or fps <= 0 or frames < 1:
                    raise ValueError(f"unplayable evidence clip: {event['clip_path']}")
                duration = frames / fps
                seconds = run["config"]["output"].get("evidence_seconds", 3)
                expected = min(summary["video_seconds"], event["alert_ts"] + seconds + 1 / fps) - max(0, event["alert_ts"] - seconds)
                if abs(duration - expected) > 2 / fps:
                    raise ValueError(f"evidence duration {duration} vs {expected}: {event['event_id']}")
            finally:
                cap.release()
            count += 1
    print(f"Verified {len(run['videos'])} video outputs and {count} emitted alerts with readable snapshots and playable clips.")


if __name__ == "__main__":
    main()
