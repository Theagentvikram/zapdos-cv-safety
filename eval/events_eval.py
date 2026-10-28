"""Event-level precision / recall of the pipeline's alerts against the hand labels in labels/.

Each alert is first given to one labelled person: among the people whose labelled box at the alert
time (keyframes interpolated, grown by 30% each side for box jitter) contains the alert box's centre,
the one whose box overlaps the alert box most (IoU). It then matches that person's labelled event of
the same type (and zone) if the alert time falls inside the labelled interval (+- 1 s).
Alerts are taken in time order. The first alert on a labelled event is a true alert; later alerts on
the same event are duplicates; an alert whose person has an "ignore" entry for that type (state not
judgeable by eye either) is dropped from the counts; anything else, including an alert on nobody, is
false. Labelled events with no true alert are missed.
Precision counts duplicates as wrong: TP / (TP + FP + duplicates).

python eval/events_eval.py --run          # run the pipeline on each labelled clip, then score
python eval/events_eval.py                # score existing out/<camera>/events.jsonl
"""
import argparse
import json
import sys
from collections import Counter, namedtuple
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
TIME_TOL, GROW = 1.0, 0.3


def box_at(person, t):
    keys = sorted((float(k), v) for k, v in person["boxes"].items())
    ts = [k for k, _ in keys]
    boxes = np.array([v for _, v in keys], float)
    return np.array([np.interp(t, ts, boxes[:, i]) for i in range(4)])


def same_person(pred_box, person, t):
    x1, y1, x2, y2 = box_at(person, t)
    gw, gh = GROW * (x2 - x1), GROW * (y2 - y1)
    cx, cy = (pred_box[0] + pred_box[2]) / 2, (pred_box[1] + pred_box[3]) / 2
    return x1 - gw <= cx <= x2 + gw and y1 - gh <= cy <= y2 + gh


def iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
    return w * h / union if union > 0 else 0.0


def whose(pred_box, people, t):
    """The labelled person an alert box belongs to, or None."""
    near = [k for k, person in people.items() if same_person(pred_box, person, t)]
    return max(near, key=lambda k: iou(pred_box, box_at(people[k], t)), default=None)


def hits(entries, p, who, timed=True):
    """Indexes of the labelled entries alert `p`, on person `who`, can stand for."""
    t = p["alert_ts"]
    return [i for i, g in enumerate(entries)
            if g["type"] == p["type"] and g.get("zone") == p.get("zone_id") and g["person"] == who
            and (not timed or g["start"] - TIME_TOL <= t <= g["end"] + TIME_TOL)]


Row = namedtuple("Row", "event_id type zone_id alert_ts track_id verdict person")  # saved as a JSON list
COUNTS = ("labelled", "alerts", "true", "false", "duplicates", "ignored", "missed")


def _verdict(p, who, labels, matched):
    """true (first alert on a labelled event), duplicate, ignored or false; marks the matched event."""
    cand = hits(labels["events"], p, who)
    free = [i for i in cand if matched[i] is None]
    if free:
        matched[free[0]] = p
        return "true"
    if cand:
        return "duplicate"
    if hits(labels.get("ignore", []), p, who, timed=False):
        return "ignored"
    return "false"


def _by_type(gts, preds, rows, matched):
    out = {}
    for etype in sorted({g["type"] for g in gts} | {p["type"] for p in preds}):
        gi = [k for k, g in enumerate(gts) if g["type"] == etype]
        verdicts = Counter(r.verdict for r in rows if r.type == etype)
        out[etype] = {"labelled": len(gi), "true": verdicts["true"], "false": verdicts["false"],
                      "duplicates": verdicts["duplicate"], "ignored": verdicts["ignored"],
                      "missed": sum(matched[k] is None for k in gi)}
    return out


def score_clip(labels, preds):
    gts = labels["events"]
    matched = [None] * len(gts)
    rows = []
    for p in sorted(preds, key=lambda e: e["alert_ts"]):
        who = whose(p["bbox"], labels["people"], p["alert_ts"])
        verdict = _verdict(p, who, labels, matched)
        rows.append(Row(p["event_id"], p["type"], p.get("zone_id"), round(p["alert_ts"], 2), p["track_id"],
                        verdict, who))
    out = Counter(r.verdict for r in rows)
    delays = [m["alert_ts"] - g["start"] for g, m in zip(gts, matched) if m is not None]
    missed = [(g["type"], g.get("zone"), g["person"]) for g, m in zip(gts, matched) if m is None]
    return {"labelled": len(gts), "alerts": len(preds), "true": out["true"], "false": out["false"],
            "duplicates": out["duplicate"], "ignored": out["ignored"], "missed": len(missed),
            "median_delay_s": round(float(np.median(delays)), 2) if delays else None,
            "by_type": _by_type(gts, preds, rows, matched), "alerts_detail": rows, "missed_detail": missed}


def pr(s):
    wrong = s["false"] + s["duplicates"]
    return (round(s["true"] / (s["true"] + wrong), 3) if s["true"] + wrong else None,
            round(s["true"] / s["labelled"], 3) if s["labelled"] else None)


def run_pipeline(label_files, config, out):
    sys.path.insert(0, str(ROOT / "src"))
    from zapdos.pipeline import build_models, run
    cfg = yaml.safe_load((ROOT / config).read_text())
    models = build_models(cfg)
    for f in label_files:
        lab = json.loads(f.read_text())
        run(ROOT / lab["video"], ROOT / lab["zones"], ROOT / out, cfg, f.stem, models=models)


def _print_table(results, total):
    print(f"\n{'clip':<26} {'labelled':>8} {'alerts':>6} {'true':>5} {'false':>5} {'dup':>4} {'ignored':>7} "
          f"{'missed':>6} {'median delay':>12}")
    for clip, s in results.items():
        print(f"{clip:<26} {s['labelled']:>8} {s['alerts']:>6} {s['true']:>5} {s['false']:>5} "
              f"{s['duplicates']:>4} {s['ignored']:>7} {s['missed']:>6} {s['median_delay_s'] or '-':>12}")
    print(f"{'total':<26} {total['labelled']:>8} {total['alerts']:>6} {total['true']:>5} {total['false']:>5} "
          f"{total['duplicates']:>4} {total['ignored']:>7} {total['missed']:>6}")


def _print_by_type(results):
    types = Counter()
    for s in results.values():
        for etype, b in s["by_type"].items():
            for k, v in b.items():
                types[(etype, k)] += v
    for etype in sorted({t for t, _ in types}):
        b = {k: types[(etype, k)] for k in COUNTS if k != "alerts"}
        p_, r_ = pr(b)
        print(f"  {etype:<10} labelled {b['labelled']}  true {b['true']}  false {b['false']}  dup {b['duplicates']}  "
              f"ignored {b['ignored']}  missed {b['missed']}  -> precision {p_}  recall {r_}")


def _print_errors(results):
    for clip, s in results.items():
        for r in s["alerts_detail"]:
            if r.verdict in ("false", "duplicate"):
                print(f"  {clip}: {r.event_id} {r.type} {r.zone_id or ''} t={r.alert_ts} track {r.track_id} "
                      f"on {r.person or 'nobody'} -> {r.verdict}")
        for m in s["missed_detail"]:
            print(f"  {clip}: MISSED {m[0]} {m[1] or ''} person {m[2]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default="labels")
    ap.add_argument("--out", default="out", help="pipeline output root (out/<camera_id>/events.jsonl)")
    ap.add_argument("--run", action="store_true", help="run the pipeline on each labelled clip first")
    ap.add_argument("--config", default="configs/rules.yaml")
    ap.add_argument("--results", default="eval/results/events.json")
    args = ap.parse_args()

    label_files = sorted((ROOT / args.labels).glob("*.json"))
    if args.run:
        run_pipeline(label_files, args.config, args.out)

    results, total = {}, Counter()
    for f in label_files:
        ev_file = ROOT / args.out / f.stem / "events.jsonl"
        preds = [json.loads(line) for line in ev_file.read_text().splitlines()] if ev_file.exists() else []
        results[f.stem] = s = score_clip(json.loads(f.read_text()), preds)
        for k in COUNTS:
            total[k] += s[k]
    p, r = pr(total)
    _print_table(results, total)
    print(f"\nprecision {p}  recall {r}  (duplicates count against precision)")
    _print_by_type(results)
    _print_errors(results)

    out = ROOT / args.results
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"total": {**total, "precision": p, "recall": r}, "clips": results}, indent=2))


if __name__ == "__main__":
    main()
