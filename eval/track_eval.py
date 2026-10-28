"""Does one person keep one ID? ByteTrack against the labelled people in labels/.

Every processed frame, the labelled people present are matched one-to-one to tracks by box overlap
(keyframes interpolated; Hungarian on IoU, pairs under 0.3 dropped; label boxes are hand-drawn at 1 s
keyframes, so the bar is loose). Reported per person: share of frames with a track ("tracked"),
distinct track IDs, and ID switches (the matched ID changes from one matched frame to the next; a
gap in between doesn't count, so an ID held through an occlusion is not a switch).

python eval/track_eval.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))
from events_eval import box_at, iou  # noqa: E402
from steelbench_clips import extract  # noqa: E402
from zapdos.pipeline import build_models  # noqa: E402

MIN_IOU = 0.3


def score(cache, people):
    spans = {n: (min(map(float, p["boxes"])), max(map(float, p["boxes"]))) for n, p in people.items()}
    ids = {n: [] for n in people}  # (ts, track id) per matched frame
    frames = dict.fromkeys(people, 0)
    for _, ts, tracks, _ in cache["frames"]:
        here = [n for n, (a, b) in spans.items() if a <= ts <= b]
        for n in here:
            frames[n] += 1
        if not here or not len(tracks):
            continue
        m = np.array([[iou(box_at(people[n], ts), t[:4]) for t in tracks] for n in here])
        for i, j in zip(*linear_sum_assignment(-m)):  # one track per person, as in MOT scoring
            if m[i, j] >= MIN_IOU:
                ids[here[i]].append((ts, int(tracks[j, 4])))
    rows = {}
    for n, seq in ids.items():
        gaps = [round(b[0] - a[0], 1) for a, b in zip(seq, seq[1:]) if a[1] != b[1]]
        rows[n] = {"frames": frames[n], "tracked": round(len(seq) / max(frames[n], 1), 2),
                   "ids": sorted({i for _, i in seq}), "switches": len(gaps), "switch_gaps_s": gaps}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="labels")
    ap.add_argument("--config", default="configs/rules.yaml")
    ap.add_argument("--results", default="eval/results/tracking.json")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text())
    models = build_models(cfg)

    results = {}
    print(f"{'clip':26} {'person':7} {'tracked':>7} {'IDs':>4} {'switches':>8}  track ids, unseen before each switch")
    for f in sorted((ROOT / args.labels).glob("*.json")):
        lab = json.loads(f.read_text())
        rows = score(extract(ROOT / lab["video"], cfg, models), lab["people"])
        results[f.stem] = rows
        for name, r in rows.items():
            print(f"{f.stem:26} {name:7} {r['tracked']:7.0%} {len(r['ids']):4} {r['switches']:8}  {r['ids']} "
                  f"{' '.join(f'{g:g}s' for g in r['switch_gaps_s'])}")
    allr = [r for rows in results.values() for r in rows.values()]
    seen = [r for r in allr if r["ids"]]
    total = {"people": len(allr), "never_tracked": len(allr) - len(seen),
             "one_id": sum(len(r["ids"]) == 1 for r in seen),
             "switches": sum(r["switches"] for r in allr),
             "switches_after_buffer": sum(g > cfg["tracker"]["track_buffer_s"]
                                          for r in allr for g in r["switch_gaps_s"]),
             "tracked": round(sum(r["tracked"] * r["frames"] for r in allr) / sum(r["frames"] for r in allr), 2)}
    print(f"\n{total['people']} people: {total['one_id']} of {len(seen)} tracked people kept one ID, "
          f"{total['switches']} ID switches ({total['switches_after_buffer']} after the person was unseen longer than "
          f"the {cfg['tracker']['track_buffer_s']:g} s buffer), {total['never_tracked']} never tracked; "
          f"{total['tracked']:.0%} of person-frames tracked")
    Path(ROOT / args.results).write_text(json.dumps({"total": total, "clips": results}, indent=1))


if __name__ == "__main__":
    main()
