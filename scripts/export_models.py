"""Export the two models for faster inference. With neither --detector nor --classifier, both are exported.

- helmet classifier -> ONNX (ONNX Runtime CPU; PyTorch eager spends most of its time on per-layer overhead
  for a 64x64 MobileNet)
- SH17 detector     -> CoreML fp16 (Apple Neural Engine on M-series Macs; the Apple analogue of TensorRT),
                       or ONNX fp32 with --onnx

python scripts/export_models.py [--detector] [--classifier]
python scripts/export_models.py --detector --square --imgsz 1280 960          # other inputs, for comparison
python scripts/export_models.py --detector [--onnx] --weights weights/sh17/yolo8n.pt
"""
import argparse
import math
import shutil
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from zapdos.helmet import INPUT, build_model  # noqa: E402


def export_classifier(weights):
    m = build_model()
    m.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    m.eval()
    out = weights.with_suffix(".onnx")
    torch.onnx.export(m, torch.zeros(1, 3, INPUT, INPUT), out, input_names=["x"], output_names=["logits"],
                      dynamic_axes={"x": {0: "n"}, "logits": {0: "n"}}, opset_version=17, dynamo=False)
    print("classifier ->", out)


def export_detector(weights, imgsz, aspect=None, fmt="coreml"):
    """Fixed-shape model, saved next to the weights as <stem>_<input>.mlpackage or .onnx. With aspect
    (w/h, e.g. 16/9) the input is a stride-aligned rectangle (1280 -> 736x1280), so a 1080p frame is
    letterboxed with 16 px of padding instead of 560."""
    from ultralytics import YOLO
    shape = [math.ceil(imgsz / aspect / 32) * 32, imgsz] if aspect else imgsz
    opts = {"half": True, "nms": False} if fmt == "coreml" else {"opset": 17, "simplify": True}
    path = Path(YOLO(str(weights)).export(format=fmt, imgsz=shape, **opts))
    tag = f"{shape[0]}x{shape[1]}" if aspect else str(imgsz)
    dest = weights.with_name(f"{weights.stem}_{tag}{path.suffix}")
    if dest.is_dir():
        shutil.rmtree(dest)
    path.replace(dest)
    print("detector ->", dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rules.yaml")
    ap.add_argument("--classifier", action="store_true")
    ap.add_argument("--detector", action="store_true")
    ap.add_argument("--imgsz", type=int, nargs="+", default=[1280])
    ap.add_argument("--square", action="store_true", help="square input instead of 16:9")
    ap.add_argument("--onnx", action="store_true", help="export the detector to ONNX instead of CoreML")
    ap.add_argument("--weights", help="detector .pt to export (default: the config's)")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text())
    if args.classifier or not args.detector:
        export_classifier(ROOT / cfg["ppe"]["classifier"]["weights"])
    if args.detector or not args.classifier:
        weights = ROOT / (args.weights or cfg["detector"]["weights"])
        for s in args.imgsz:
            export_detector(weights, s, None if args.square else 16 / 9, "onnx" if args.onnx else "coreml")


if __name__ == "__main__":
    main()
