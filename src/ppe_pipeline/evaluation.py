"""Event-level evaluation with explicit, reviewed person-ID mapping.

Predicted tracker IDs are not ground-truth person IDs. Mapping belongs to each run;
missing mappings count as false alerts, never as matches. Matching maximizes the
number of one-to-one matches, rather than greedily consuming the closest label.
"""
from __future__ import annotations
import statistics
import math

TYPES = {"no_helmet", "no_vest", "zone_entry"}


def evaluate(labels, predictions, identity_map, tolerance=1.0):
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("time tolerance cannot be negative")
    if labels.get("reviewed") is not True or identity_map.get("reviewed") is not True:
        raise ValueError("event labels and per-run person mapping must be reviewed before evaluation")
    truth = labels["events"]
    clips = set(labels["clips"])
    for item in truth:
        if (item["video"] not in clips or item["type"] not in TYPES or
                not all(math.isfinite(item[k]) for k in ("start_ts", "end_ts")) or
                item["end_ts"] < item["start_ts"] or not item.get("person_id")):
            raise ValueError(f"invalid ground-truth event: {item}")
    # Scope results to explicitly reviewed clips, including zero-event clips.
    pred = [e for e in predictions if e.get("alert_status", "emitted") == "emitted" and e["video"] in clips]
    if len({e["event_id"] for e in pred}) != len(pred):
        raise ValueError("duplicate event IDs in predictions; pass exactly one run")
    for event in pred:
        if event["type"] not in TYPES or not all(math.isfinite(event[k]) for k in ("start_ts", "alert_ts")):
            raise ValueError(f"invalid prediction: {event.get('event_id')}")
    mapping = identity_map.get("tracks", {})
    edges = []
    for event in pred:
        person = mapping.get(event["video"], {}).get(str(event["track_id"]))
        options = []
        for j, label in enumerate(truth):
            if (event["video"] == label["video"] and event["camera_id"] == label["camera_id"] and
                    event["type"] == label["type"] and person == label["person_id"] and
                    event.get("zone_id") == label.get("zone_id") and
                    label["start_ts"] - tolerance <= event["alert_ts"] <= label["end_ts"] + tolerance):
                options.append(j)
        options.sort(key=lambda j: abs(event["start_ts"] - truth[j]["start_ts"]))
        edges.append(options)
    matched = {}

    def assign(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in matched or assign(matched[j], seen):
                matched[j] = i
                return True
        return False

    for i in range(len(pred)):
        assign(i, set())
    used = set(matched.values())
    duplicates = [i for i in range(len(pred)) if i not in used and edges[i]]
    false = [i for i in range(len(pred)) if i not in used and not edges[i]]
    missed = [j for j in range(len(truth)) if j not in matched]
    delays = [pred[i]["alert_ts"] - truth[j]["start_ts"] for j, i in matched.items()]
    tp, fp, fn = len(matched), len(pred) - len(matched), len(missed)
    return {
        "labelled_events": len(truth), "alerts": len(pred), "true": tp,
        "false": len(false), "duplicates": len(duplicates), "missed": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "median_delay_sec": statistics.median(delays) if delays else None,
        "matches": [{"label_index": j, "event_id": pred[i]["event_id"]} for j, i in matched.items()],
        "false_event_ids": [pred[i]["event_id"] for i in false],
        "duplicate_event_ids": [pred[i]["event_id"] for i in duplicates],
        "missed_label_indices": missed,
        "note": "Duplicates are included in the precision denominator; null means undefined.",
    }
