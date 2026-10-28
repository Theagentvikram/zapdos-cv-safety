"""SH17 detector wrapper: one forward pass per frame, PyTorch or ONNX Runtime backend.

Both backends load the same SH17 YOLOv8s weights; the ONNX file is a one-off export of
models/sh17/yolo8s.pt (created on first use if missing). The ONNX path reuses Ultralytics'
letterbox, NMS and box scaling so only the network execution differs.

`onnx_imgsz` must equal the letterboxed input PyTorch uses at `imgsz` for the video aspect ratio
(1920x1080 at 960 -> 544x960); otherwise outputs are still valid but not comparable with PyTorch.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
from ultralytics import YOLO


class Detector:
    def __init__(self, model_cfg: dict, root: Path):
        self.cfg = model_cfg
        self.backend = model_cfg.get("backend", "torch")
        threads = model_cfg.get("cpu_threads")
        pt_path = (root / model_cfg["path"]).resolve()
        if self.backend == "torch":
            if threads:
                import torch

                torch.set_num_threads(int(threads))
            self.model = YOLO(str(pt_path))
            self._names = {int(k): v for k, v in self.model.names.items()}
        elif self.backend == "onnx":
            import onnxruntime as ort
            from ultralytics.data.augment import LetterBox

            self.imgsz = tuple(int(v) for v in model_cfg["onnx_imgsz"])
            onnx_path = (root / model_cfg["onnx_path"]).resolve()
            if not onnx_path.exists():
                exported = YOLO(str(pt_path)).export(format="onnx", imgsz=list(self.imgsz), dynamic=False,
                                                     simplify=False, device="cpu")
                Path(exported).replace(onnx_path)
            options = ort.SessionOptions()
            if threads:
                options.intra_op_num_threads = int(threads)
            if not model_cfg.get("allow_spinning", not bool(model_cfg.get("person_model"))):
                options.add_session_config_entry("session.intra_op.allow_spinning", "0")
                options.add_session_config_entry("session.inter_op.allow_spinning", "0")
            self.session = ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])
            meta = self.session.get_modelmeta().custom_metadata_map
            self._names = {int(k): v for k, v in ast.literal_eval(meta["names"]).items()}
            exported_imgsz = tuple(ast.literal_eval(meta["imgsz"]))
            if exported_imgsz != self.imgsz:
                raise SystemExit(f"{onnx_path} was exported at {exported_imgsz}, config expects {self.imgsz}")
            self.input_name = self.session.get_inputs()[0].name
            self.letterbox = LetterBox(self.imgsz, auto=False, stride=32)
        else:
            raise SystemExit(f"unknown model.backend '{self.backend}'")
        for cls_id, name in model_cfg["class_names"].items():
            if self._names.get(int(cls_id)) != name:
                raise SystemExit(f"class id {cls_id} is '{self._names.get(int(cls_id))}', expected '{name}'")
        self.person_cfg = model_cfg.get("person_model")
        self.person_model = None
        if self.person_cfg:
            path = (root / self.person_cfg["path"]).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"independent person weights not found: {path}")
            pc = self.person_cfg
            # Use the same ONNX path as the PPE detector to avoid a second
            # competing PyTorch CPU thread pool in steady-state inference.
            self.person_model = Detector({"path": str(path), "backend": pc.get("backend", "onnx"),
                                          "onnx_path": pc.get("onnx_path", str(path.with_suffix(".onnx"))),
                                          "onnx_imgsz": pc.get("onnx_imgsz", [384, 640]),
                                          "imgsz": pc.get("imgsz", 640), "conf": pc.get("conf", .15),
                                          "iou": pc.get("iou", .45), "classes": [0],
                                          "class_names": {0: "person"}, "device": "cpu",
                                          "cpu_threads": pc.get("cpu_threads", 4), "allow_spinning": False}, root)

    @property
    def names(self) -> dict:
        return self._names

    def __call__(self, frame) -> np.ndarray:
        rows = self._base(frame)
        if self.person_model is None:
            return rows
        persons = self.person_model(frame)
        person_id = next(int(k) for k,v in self.cfg["class_names"].items() if v == "person")
        persons[:, 5] = person_id
        return np.concatenate([rows[rows[:, 5] != person_id], persons])

    def _base(self, frame) -> np.ndarray:
        """Return (N, 6) float32 rows: x1, y1, x2, y2, conf, cls."""
        mc = self.cfg
        if self.backend == "torch":
            result = self.model.predict(frame, imgsz=mc["imgsz"], conf=mc["conf"], iou=mc["iou"],
                                        classes=mc["classes"], device=mc["device"], verbose=False)[0]
            if result.boxes is None or len(result.boxes) == 0:
                return np.zeros((0, 6), np.float32)
            return result.boxes.data.cpu().numpy().astype(np.float32)[:, :6]
        return self._onnx(frame)

    def _onnx(self, frame) -> np.ndarray:
        import torch
        from ultralytics.utils import nms, ops

        mc = self.cfg
        img = self.letterbox(image=frame)
        blob = np.ascontiguousarray(img[..., ::-1].transpose(2, 0, 1))[None].astype(np.float32) / 255.0
        raw = self.session.run(None, {self.input_name: blob})[0]
        det = nms.non_max_suppression(torch.from_numpy(raw), mc["conf"], mc["iou"], mc["classes"], False,
                                      max_det=300, nc=0)[0]
        if len(det) == 0:
            return np.zeros((0, 6), np.float32)
        det[:, :4] = ops.scale_boxes(blob.shape[2:], det[:, :4], frame.shape)
        return det[:, :6].numpy().astype(np.float32)
