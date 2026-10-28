"""Temporal PPE compliance pipeline: SH17 YOLOv8s (960, CPU) + ByteTrack + helmet/vest state machines.

Usage (from the repo root):
    python scripts/run_ppe_pipeline.py --clips VIDEO.mp4 --config configs/ppe_pipeline.yaml
    python scripts/run_ppe_pipeline.py --clips VIDEO.mp4 --max-frames 100
    python scripts/run_ppe_pipeline.py --clips VIDEO.mp4 --allow-unvalidated-absence-alerts --out out_diagnostic
    python scripts/run_ppe_pipeline.py --clips VIDEO.mp4 --backend onnx --threads 4

Pass input videos explicitly with --clips. Source videos are only read.
Outputs go to <output.dir>/<short-scene>/<short-clip-stem>/ (see run_summary.json):
    annotated.mp4, worker_log.csv, candidates.csv, workers.json, events.jsonl, diagnostics.jsonl, evidence/, clips/, snapshots/,
    timeline.md, summary.json
and <output.dir>/run_summary.json + latency_report.md for the whole run.
"""

import argparse
import json
import sys
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ppe_pipeline.detector import Detector  # noqa: E402
from src.ppe_pipeline.pipeline import load_config, process_video, warmup  # noqa: E402

STAGE_ORDER = ("decode", "inference", "tracking", "association", "rules", "render", "write", "total")


def latency_markdown(records: list[dict]) -> str:
    backend = records[0]["backend"] if records else "?"
    lines = [f"# CPU latency report: SH17 YOLOv8s ({backend}) + ByteTrack + PPE rules", "",
             "Median ms per frame (p95 in brackets). Decoding and file writing run on background threads:",
             "`decode` is the reader thread's time per frame, `write` is the main thread's hand-off to the writer,",
             "`total` is the main thread's time per frame including waiting for a decoded frame.",
             "Pipeline FPS = processed frames / wall-clock seconds until writer flush. Decode includes skipped-frame grabs.",
             "Source coverage = video seconds / wall seconds; >= 1 means offline throughput can keep up with the source.", "",
             "| clip | frames | " + " | ".join(STAGE_ORDER) + " | writer thread | pipeline FPS | inference-only FPS |",
             "|---|---|" + "---|" * len(STAGE_ORDER) + "---|---|---|"]
    for r in records:
        lat = r["latency_ms"]
        cells = [f"{lat[s]['median_ms']} ({lat[s]['p95_ms']})" for s in STAGE_ORDER]
        wt = lat.get("writer_thread")
        lines.append(f"| {r['scene']} | {r['frames']} | " + " | ".join(cells)
                     + f" | {wt['median_ms'] if wt else '-'} | {r['pipeline_fps_end_to_end']} | {r['inference_only_fps']} |")
    frames = sum(r["frames"] for r in records)
    wall = sum(r["wall_seconds"] for r in records)
    lines += ["", f"Overall: {frames} frames in {wall:.1f} s wall time, {frames / wall:.2f} FPS end to end "
              f"({'meets' if frames / wall >= 5 else 'BELOW'} the 5 FPS target)."]
    seconds = sum(r.get("video_seconds", r["frames"] / r["fps_video"]) for r in records)
    lines += [f"Source video: {seconds:.2f} s; coverage {seconds / wall:.3f}x real time.",
              "Sampling reduces detector calls; it is not a measured improvement in event recall."]
    lines += ["", "Frame age includes local queueing; paced-file age starts at the source-frame schedule.",
              "It excludes camera/network capture and output delivery, and is not an RTSP measurement.", "",
              "| clip | input mode | median frame age ms | p95 frame age ms | maximum reader queue |",
              "|---|---|---:|---:|---:|"]
    for r in records:
        age = r.get("frame_age_ms", {})
        lines.append(f"| {r['scene']} | {r.get('input_mode', 'offline_file')} | "
                     f"{age.get('median_ms', '-')} | {age.get('p95_ms', '-')} | {r.get('reader_queue_max_depth', '-')} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(ROOT / "configs" / "ppe_pipeline.yaml"))
    parser.add_argument("--clips", nargs="+", required=True, help="one or more input video files")
    parser.add_argument("--out", help="override output.dir")
    parser.add_argument("--max-frames", type=int, help="stop each video after N frames (smoke tests)")
    parser.add_argument("--backend", choices=["torch", "onnx"], help="override model.backend")
    parser.add_argument("--target-fps", type=float, help="detector sampling rate; 0 processes all source frames")
    parser.add_argument("--zones", help="zone JSON override, requires exactly one input clip")
    parser.add_argument("--no-zones", action="store_true", help="disable zones for an ablation run")
    parser.add_argument("--camera-id", help="camera identifier override, requires exactly one input clip")
    parser.add_argument("--imgsz", type=int, help="experimental detector size (multiple of 32); compare accuracy before use")
    parser.add_argument("--threads", type=int, help="override model.cpu_threads")
    parser.add_argument("--person-model", help="optional independent local COCO person detector checkpoint")
    parser.add_argument("--sh17-persons", action="store_true", help="use the original SH17 person branch for comparison")
    parser.add_argument("--realtime", action="store_true", help="pace recorded video at source FPS and report queue-inclusive frame age")
    parser.add_argument("--allow-unvalidated-absence-alerts", action="store_true",
                        help="what-if run: let unvalidated cameras reach SUSPECTED and emit alerts "
                             "(events still record camera_validated_for_absence=false)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.person_model:
        cfg["model"]["person_model"] = {"path": str(Path(args.person_model).resolve()), "imgsz": 640, "conf": .15}
    if args.sh17_persons:
        if args.person_model:parser.error("--person-model and --sh17-persons cannot be combined")
        cfg["model"].pop("person_model",None)
    if args.realtime:
        cfg.setdefault("runtime",{})["realtime"]=True
    if args.target_fps is not None:
        if not math.isfinite(args.target_fps) or args.target_fps < 0:
            parser.error("--target-fps must be nonnegative and finite")
        cfg.setdefault("runtime", {})["target_fps"] = args.target_fps or None
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    if args.allow_unvalidated_absence_alerts:
        cfg["alerts"]["allow_unvalidated_camera_absence_alerts"] = True
    if args.backend:
        cfg["model"]["backend"] = args.backend
    if args.threads:
        cfg["model"]["cpu_threads"] = args.threads
    out_root = Path(args.out) if args.out else ROOT / cfg["output"]["dir"]
    out_root.mkdir(parents=True, exist_ok=True)
    clips = [Path(c) for c in args.clips]
    if not clips:
        parser.error("no videos found; pass --clips VIDEO.mp4")
    if any(not clip.is_file() for clip in clips):
        parser.error("one or more input videos do not exist")
    if (args.zones or args.camera_id) and len(clips) != 1:
        parser.error("--zones and --camera-id require exactly one clip")
    if args.zones and args.no_zones:
        parser.error("--zones and --no-zones cannot be combined")
    if args.zones or args.camera_id:
        from src.ppe_pipeline.pipeline import camera_for
        camera = dict(camera_for(clips[0], cfg, args.camera_id))
        camera["match"] = clips[0].name
        if args.zones:
            camera["zones"] = str(Path(args.zones).resolve())
        if args.camera_id:
            camera["id"] = args.camera_id
        cfg["cameras"] = [camera]
    if args.no_zones:
        for camera in cfg.get("cameras", []):
            camera.pop("zones", None)
    if args.imgsz:
        if args.imgsz < 32 or args.imgsz % 32:
            parser.error("--imgsz must be a positive multiple of 32")
        import cv2
        cap = cv2.VideoCapture(str(clips[0]))
        width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        cap.release()
        if width <= 0 or height <= 0:
            parser.error("cannot read input video dimensions")
        cfg["model"]["imgsz"] = args.imgsz
        scale = args.imgsz / max(width, height)
        cfg["model"]["onnx_imgsz"] = [math.ceil(height * scale / 32) * 32, math.ceil(width * scale / 32) * 32]
        model_path = Path(cfg["model"]["path"])
        cfg["model"]["onnx_path"] = str(model_path.with_name(f"{model_path.stem}_{args.imgsz}.onnx"))

    detector = Detector(cfg["model"], ROOT)
    print(f"backend {detector.backend}; classes verified:",
          {k: detector.names[int(k)] for k in cfg["model"]["class_names"]}, flush=True)
    warmup(detector, clips[0])
    records = []
    for clip in clips:
        print(f"processing {clip.name}", flush=True)
        record = process_video(detector, clip, cfg, out_root, args.max_frames)
        records.append(record)
        print(f"  {record['scene']}: {record['frames']} frames, {record['confirmed_workers']} confirmed workers "
              f"({record['unconfirmed_candidates']} candidates), {record['events']} events "
              f"({record['events_emitted']} emitted), pipeline {record['pipeline_fps_end_to_end']} FPS", flush=True)

    (out_root / "run_summary.json").write_text(
        json.dumps({"config": cfg, "videos": records}, indent=2, default=str), encoding="utf-8")
    (out_root / "latency_report.md").write_text(latency_markdown(records), encoding="utf-8")
    print((out_root / "latency_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
