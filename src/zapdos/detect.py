"""One-pass PPE detector: persons, heads, helmets and vests from a single SH17 YOLO forward pass."""
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO


def available_device(want):
    """The configured torch device if this machine has it, else CUDA, else CPU (the default config
    asks for mps, which only Apple silicon has)."""
    import torch
    present = {"cpu": True, "mps": torch.backends.mps.is_available(), "cuda": torch.cuda.is_available()}
    if present.get(str(want).split(":")[0], False):
        return want
    fallback = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"device {want} not available here; using {fallback}")
    return fallback


@dataclass
class Detections:
    persons: object      # ultralytics Boxes (numpy), passed straight to the tracker
    heads: np.ndarray    # (N, 5): x1, y1, x2, y2, conf
    helmets: np.ndarray
    vests: np.ndarray


class PPEDetector:
    def __init__(self, cfg):
        coreml = cfg.get("coreml")
        units = cfg.get("coreml_units", "cpu_and_ne")
        if coreml and sys.platform == "darwin" and Path(coreml).exists():
            # input shape comes from the export metadata. ultralytics loads CoreML models for CPU +
            # Neural Engine; only cpu_only changes that (below), other coreml_units values are a label
            self.model, self.device, self.backend = YOLO(coreml, task="detect"), None, f"coreml-{units}"
        else:
            self.model, self.device, self.backend = YOLO(cfg["weights"]), cfg["device"], f"torch-{cfg['device']}"
        self.imgsz = cfg["imgsz"]
        self.conf = cfg["conf"]
        self.cls = cfg["classes"]
        self.min_conf = cfg["min_conf"]
        # first MPS/CPU call compiles kernels; keep it out of the measured latency
        self(np.zeros((1080, 1920, 3), np.uint8))
        if self.backend == "coreml-cpu_only":  # swap the loaded model for a CPU-only one (no NE, no GPU)
            import coremltools as ct
            self.model.predictor.model.backend.model = ct.models.MLModel(coreml, compute_units=ct.ComputeUnit.CPU_ONLY)
            self(np.zeros((1080, 1920, 3), np.uint8))

    def __call__(self, frame):
        r = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, device=self.device,
                               classes=list(self.cls.values()), verbose=False)[0]
        boxes = r.boxes.cpu().numpy()
        cls = boxes.cls.astype(int)

        def pick(name):
            m = (cls == self.cls[name]) & (boxes.conf >= self.min_conf[name])
            return np.hstack([boxes.xyxy[m], boxes.conf[m, None]])

        return Detections(boxes[cls == self.cls["person"]], pick("head"), pick("helmet"), pick("vest"))
