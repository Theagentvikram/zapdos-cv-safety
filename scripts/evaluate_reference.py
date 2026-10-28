"""Measure person detections and PPE states against reviewed source-frame labels."""
import argparse
import json
import sys
import time
import hashlib
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ppe_pipeline.detector import Detector
from src.ppe_pipeline.pipeline import load_config, camera_for
from src.ppe_pipeline.quality import filter_detections
from src.ppe_pipeline.association import associate, ASSOCIATED
from src.ppe_pipeline.reference import validate_reference, score_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--images", help="source image directory, defaults to reference parent")
    parser.add_argument("--config", default=str(ROOT / "configs/ppe_pipeline.yaml"))
    parser.add_argument("--person-model", help="optional local independent person detector weights")
    parser.add_argument("--sh17-persons", action="store_true", help="measure the original person branch")
    parser.add_argument("--head-model", help="optional local hardhat/no-hardhat detector weights")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    data = json.loads(Path(args.reference).read_text())
    validate_reference(data)
    cfg = load_config(args.config)
    if args.person_model:
        cfg["model"]["person_model"] = {"path": str(Path(args.person_model).resolve()), "imgsz": 640, "conf": .15}
    if args.sh17_persons:
        if args.person_model:parser.error("conflicting person detector options")
        cfg["model"].pop("person_model",None)
    detector = Detector(cfg["model"], ROOT)
    head = None
    if args.head_model:
        from src.ppe_pipeline.verification import HeadVerifier
        head = HeadVerifier(Path(args.head_model))
    root = Path(args.images) if args.images else Path(args.reference).parent
    predictions, latency = {}, []
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for frame in data["frames"]:
        image = cv2.imread(str(root / frame["image"]))
        if image is None:
            raise OSError(f"cannot read {frame['image']}")
        start = time.perf_counter()
        rows = detector(image)
        camera = camera_for(Path(frame["video"]), cfg)
        rows, _ = filter_detections(rows, image.shape, cfg.get("detection_quality", {}), camera)
        persons = rows[rows[:, 5] == 0]
        boxes = [tuple(map(float, p[:4])) for p in persons]
        record = [{"bbox": list(box), "confidence": float(row[4]), "helmet": "unknown", "vest": "unknown"}
                  for box, row in zip(boxes, persons)]
        for kind, cls in (("helmet", 10), ("vest", 16)):
            ppe = rows[rows[:, 5] == cls]
            matches = associate(kind, boxes, list(ppe[:, :4]), list(ppe[:, 4]), cfg["association"])
            for i, match in enumerate(matches):
                if match.status == ASSOCIATED:
                    record[i][kind] = "on"
        if head:
            for p in record:
                verdict = head(image, p["bbox"])
                p["head_verifier"] = verdict
                # Independent explicit evidence can answer off; no detection remains unknown.
                if verdict["state"] != "unknown":
                    p["helmet"] = verdict["state"]
        latency.append((time.perf_counter() - start) * 1000)
        predictions[frame["id"]] = record
        for p in record:
            a,b,c,d=map(int,p["bbox"])
            cv2.rectangle(image,(a,b),(c,d),(0,255,255),2)
            cv2.putText(image,f"H:{p['helmet']} V:{p['vest']}",(a,max(16,b-4)),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,255),1)
        cv2.imwrite(str(out / (frame["id"] + ".jpg")), image)
    metrics = score_reference(data, predictions)
    metrics["model_config"] = cfg["model"]
    metrics["reference_sha256"] = hashlib.sha256(Path(args.reference).read_bytes()).hexdigest()
    metrics["mean_ms_per_reference_frame_including_first_call"] = sum(latency) / len(latency)
    (out / "predictions.json").write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
