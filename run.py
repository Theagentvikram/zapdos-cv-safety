"""One command: video in, deduplicated safety alerts out.

python run.py --video clips/x.mp4 --zones zones/x.json --out out/
python run.py --video a.mp4 b.mp4 --zones a.json b.json --out out/   # several cameras, one process
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from zapdos.multi import run_many  # noqa: E402
from zapdos.pipeline import run  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, nargs="+", help="one or more videos (several = one process)")
    ap.add_argument("--zones", nargs="*", default=[],
                    help="zone polygons JSON per video, same order; omit for PPE-only")
    ap.add_argument("--out", default="out")
    ap.add_argument("--config", default="configs/rules.yaml")
    ap.add_argument("--camera-id", help="single video only; defaults to the zones file's camera_id, "
                                        "then the video name")
    ap.add_argument("--device", help="run the .pt detector on this device (cpu | mps | cuda:0) instead of CoreML")
    ap.add_argument("--cpu-only", action="store_true",
                    help="no accelerator: CoreML restricted to the CPU on macOS, else the .pt on CPU")
    ap.add_argument("--save-video", action="store_true",
                    help="single video only: also write an annotated full-length video")
    ap.add_argument("--show", action="store_true",
                    help="single video only: live window at the video's real pace (q quits)")
    args = ap.parse_args()
    if args.zones and len(args.zones) != len(args.video):
        ap.error("give one --zones file per --video, or none")
    single_video_only = any((args.camera_id, args.save_video, args.show))
    if len(args.video) > 1 and single_video_only:
        ap.error("--camera-id, --save-video and --show take a single --video")
    return args


def load_config(args):
    """The config file with the command line's detector choices applied."""
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    det = cfg["detector"]
    if args.device:
        det["device"], det["coreml"] = args.device, None
    if args.cpu_only:
        det["device"], det["coreml_units"] = "cpu", "cpu_only"
    return cfg


def main():
    args = parse_args()
    cfg = load_config(args)
    zones = args.zones or [None] * len(args.video)
    if len(args.video) == 1:
        run(args.video[0], zones[0], args.out, cfg, args.camera_id, args.save_video, show=args.show)
    else:
        _, summary = run_many([(v, z, None) for v, z in zip(args.video, zones)], args.out, cfg)
        print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
