"""Frame-level diagnostic evaluation, independent of tracker output.

Unknown PPE is excluded from PPE scoring. Missing people count as misses; unknown
predictions count as abstentions, never as correct negatives. Agent-reviewed references
may be used for diagnostics but cannot qualify a camera for production alerts.
"""
import math
from .identity import iou


def validate_reference(data):
    if data.get("reviewed") is not True or not data.get("reviewer", {}).get("kind"):
        raise ValueError("reference requires explicit review and reviewer provenance")
    cameras = {}
    ids = set()
    for f in data["frames"]:
        if not f.get("reviewed") or f["id"] in ids:
            raise ValueError("every frame must be reviewed and have a unique ID")
        ids.add(f["id"])
        split = f["split"]
        if split not in {"development", "test"}:
            raise ValueError("frame split must be development or test")
        if f["camera_id"] in cameras and cameras[f["camera_id"]] != split:
            raise ValueError("camera leakage between development and test splits")
        cameras[f["camera_id"]] = split
        for p in f["persons"]:
            box = p["bbox"]
            if len(box) != 4 or not all(math.isfinite(x) for x in box) or box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError("invalid reference box")
            if p["helmet"] not in {"on", "off", "unknown"} or p["vest"] not in {"on", "off", "unknown"}:
                raise ValueError("PPE reference state must be on/off/unknown")


def match_boxes(truth, predicted, threshold=.5):
    edges = [[j for j, p in sorted(enumerate(predicted), key=lambda item: -iou(box, item[1]))
              if iou(box, p) >= threshold] for box in truth]
    assigned = {}

    def assign(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in assigned or assign(assigned[j], seen):
                assigned[j] = i
                return True
        return False

    for i in range(len(truth)):
        assign(i, set())
    return {i: j for j, i in assigned.items()}


def score_reference(data, predictions, threshold=.5):
    validate_reference(data)
    result = {}
    for split in ("development", "test"):
        tp = fp = fn = 0
        ppe = {kind: {"known_labels": 0, "correct": 0, "wrong": 0, "abstained": 0,
                      "false_on": 0, "false_off": 0} for kind in ("helmet", "vest")}
        for f in data["frames"]:
            if f["split"] != split:
                continue
            pred = predictions[f["id"]]
            matches = match_boxes([p["bbox"] for p in f["persons"]], [p["bbox"] for p in pred], threshold)
            tp += len(matches)
            fn += len(f["persons"]) - len(matches)
            fp += len(pred) - len(matches)
            for i, person in enumerate(f["persons"]):
                for kind in ppe:
                    if person[kind] == "unknown":
                        continue
                    stats = ppe[kind]
                    stats["known_labels"] += 1
                    state = pred[matches[i]].get(kind, "unknown") if i in matches else "unknown"
                    if state == "unknown":
                        stats["abstained"] += 1
                    elif state == person[kind]:
                        stats["correct"] += 1
                    else:
                        stats["wrong"] += 1
                        stats["false_on" if state == "on" else "false_off"] += 1
        for stats in ppe.values():
            answered = stats["correct"] + stats["wrong"]
            stats["coverage"] = answered / stats["known_labels"] if stats["known_labels"] else None
            stats["accuracy_when_answered"] = stats["correct"] / answered if answered else None
        result[split] = {"persons": {"true": tp, "false": fp, "missed": fn,
                                    "precision": tp / (tp + fp) if tp + fp else None,
                                    "recall": tp / (tp + fn) if tp + fn else None}, "ppe": ppe}
    result["reference_provenance"] = data["reviewer"]
    result["human_reviewed"] = data.get("human_reviewed", False)
    result["limitations"] = "Sparse frame diagnostic; not event recall, tracking accuracy, or camera calibration."
    return result
