"""Extract immutable source frames for independent person/PPE visual review.

Manifest: {"clips": [{"id": "camera", "path": "video.mp4", "split": "test",
"times": [1, 3, 5]}]}. The output manifest remains unreviewed; detections are never labels.
"""
import argparse
import hashlib
import json
from pathlib import Path
import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.out)
    if (root / "reference.json").exists():
        parser.error("reference.json already exists; refusing to overwrite review work")
    root.mkdir(parents=True, exist_ok=True)
    frames = []
    for clip in json.loads(Path(args.manifest).read_text())["clips"]:
        source = Path(clip["path"])
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise OSError(f"cannot open {source}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        try:
            for t in clip["times"]:
                index = round(t * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, image = cap.read()
                if not ok:
                    raise ValueError(f"cannot read {source} at {t}s")
                filename = f"{clip['id']}_f{index:06d}.png"
                if not cv2.imwrite(str(root / filename), image):
                    raise OSError(f"cannot write {filename}")
                frames.append({"id": filename[:-4], "image": filename, "video": source.name,
                               "source_path": str(source.resolve()), "source_sha256": digest,
                               "frame": index, "time_sec": index / fps,
                               "camera_id": clip["id"], "split": clip["split"],
                               "width": image.shape[1], "height": image.shape[0],
                               "reviewed": False, "persons": [], "ignore_boxes": []})
        finally:
            cap.release()
    data = {"schema_version": 1, "reviewed": False, "human_reviewed": False,
            "reviewer": None, "frames": frames,
            "instructions": "Review original pixels. Label every visible person and uncertain areas; do not copy detector boxes. Person fields: person_id, bbox, helmet (on/off/unknown), vest (on/off/unknown), feet_visible, optional head_bbox/torso_bbox. Freeze camera-level development/test splits before experiments."}
    (root / "reference.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Extracted {len(frames)} source frames into {root.resolve()}")


if __name__ == "__main__":
    main()
