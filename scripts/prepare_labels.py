"""Create unreviewed label/mapping templates and timestamped contact sheets from source video.

Never infer ground truth from detector output. Complete the event records after viewing
the original clips; set reviewed=true only after checking every event and no-event interval.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--interval", type=float, default=1)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    labels = {"reviewed": False, "clips": [Path(p).name for p in args.clips], "events": [],
              "instructions": "Empty events is UNLABELLED, not a negative clip. Review original video including gaps between thumbnails. Event fields: video, camera_id, person_id, type, start_ts, end_ts, zone_id (zones only)."}
    mapping = {"reviewed": False, "tracks": {Path(p).name: {} for p in args.clips},
               "instructions": "For this prediction run only: map each confirmed track_id to a ground-truth person_id. Multiple fragmented IDs may map to one person. Leave false-person tracks unmapped."}
    for name, data in (("labels.json", labels), ("identity_map.json", mapping)):
        path = root / name
        if path.exists():
            parser.error(f"refusing to overwrite annotation work: {path}")
    for name, data in (("labels.json", labels), ("identity_map.json", mapping)):
        (root / name).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    for clip in args.clips:
        cap = cv2.VideoCapture(clip)
        if not cap.isOpened():
            raise OSError(f"cannot open {clip}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        step = max(1, round(fps * args.interval))
        tiles, page, index = [], 0, 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if index % step == 0:
                    tile = cv2.resize(frame, (480, 270))
                    cv2.rectangle(tile, (0, 0), (480, 28), (0, 0, 0), -1)
                    cv2.putText(tile, f"{index / fps:.2f}s / frame {index}", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
                    tiles.append(tile)
                index += 1
                if len(tiles) == 12:
                    save_sheet(root, Path(clip).stem, page, tiles)
                    page, tiles = page + 1, []
            if tiles:
                save_sheet(root, Path(clip).stem, page, tiles)
        finally:
            cap.release()
    print(f"Unreviewed annotation workspace: {root.resolve()}")


def save_sheet(root, stem, page, tiles):
    canvas = np.zeros((270 * 4, 480 * 3, 3), np.uint8)
    for i, tile in enumerate(tiles):
        y, x = (i // 3) * 270, (i % 3) * 480
        canvas[y:y + 270, x:x + 480] = tile
    if not cv2.imwrite(str(root / f"{stem}_{page:02d}.jpg"), canvas):
        raise OSError("cannot save annotation contact sheet")


if __name__ == "__main__":
    main()
