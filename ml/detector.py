"""Asset detection (FR-9).

`Detector` is the swappable interface. `YoloWorldDetector` runs an
open-vocabulary YOLO-World model whose text prompts come from
config `detection.prompts`; `FullFrameDetector` treats the whole frame as one
asset for close-up desk demos. `AutoDetector` needs no operator choice: it uses
YOLO-World boxes when there are any, otherwise it classifies the whole frame's
asset type (close-ups have no object outline for YOLO-World to find).
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from PIL import Image

from ml.config import cfg, resolve_path
from ml.device import resolve_device
from ml.imaging import clip_bbox, pil_to_bgr
from ml.net import ensure_ca_bundle

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    asset_type: str
    confidence: float
    label: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["bbox"] = list(self.bbox)
        return d


class Detector(ABC):
    @abstractmethod
    def detect(self, frame: Image.Image) -> List[Detection]:
        """Return candidate assets in an RGB frame."""


class FullFrameDetector(Detector):
    def __init__(self, asset_type: str, confidence: float):
        self.asset_type = asset_type
        self.confidence = confidence

    def detect(self, frame: Image.Image) -> List[Detection]:
        return [Detection((0, 0, frame.width, frame.height), self.asset_type, self.confidence, "full_frame")]


class YoloWorldDetector(Detector):
    def __init__(self, config: Dict[str, Any]):
        ensure_ca_bundle()
        from ultralytics import YOLO, YOLOWorld

        dcfg = config["detection"]
        self.conf = float(dcfg["confidence"])
        self.iou = float(dcfg["iou"])
        self.imgsz = int(dcfg["image_size"])
        self.max_det = int(dcfg["max_detections"])
        self.min_area = float(dcfg["min_box_area_fraction"])
        device = resolve_device(config["model"]["device"])
        self.device = device.type if device.type != "cuda" else 0
        self._lock = threading.Lock()

        self.labels: List[Tuple[str, str]] = []
        for asset_type, prompts in dcfg["prompts"].items():
            for prompt in prompts:
                self.labels.append((asset_type, prompt))
        if not self.labels:
            raise ValueError("detection.prompts is empty")

        import ultralytics.nn.text_model as text_model

        # The CLIP text encoder otherwise downloads relative to the working directory.
        text_model.WEIGHTS_DIR = resolve_path(dcfg["text_encoder_dir"])
        weights = resolve_path(dcfg["weights"])
        weights.parent.mkdir(parents=True, exist_ok=True)
        world = YOLOWorld(str(weights))
        world.set_classes([p for _, p in self.labels])

        if dcfg["onnx_export"]:
            onnx_path = weights.with_name(f"{weights.stem}_{len(self.labels)}cls.onnx")
            if not onnx_path.exists():
                exported = world.export(format="onnx", imgsz=self.imgsz)
                Path(exported).replace(onnx_path)
            self.model = YOLO(str(onnx_path), task="detect")
            self.device = "cpu"
            self.backend = "onnx"
        else:
            self.model = world
            self.backend = "torch"
        logger.info("YOLO-World detector ready (%s, %d prompts)", self.backend, len(self.labels))

    def detect(self, frame: Image.Image) -> List[Detection]:
        bgr = pil_to_bgr(frame)
        with self._lock:
            results = self.model.predict(
                bgr,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                max_det=self.max_det,
                device=self.device,
                verbose=False,
            )
        out: List[Detection] = []
        frame_area = float(frame.width * frame.height)
        boxes = results[0].boxes
        if boxes is None:
            return out
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        for box, conf, cls in zip(xyxy, confs, classes):
            clipped = clip_bbox(box, frame.width, frame.height)
            if clipped is None:
                continue
            area = (clipped[2] - clipped[0]) * (clipped[3] - clipped[1])
            if area / frame_area < self.min_area:
                continue
            asset_type, label = self.labels[int(cls)]
            out.append(Detection(clipped, asset_type, float(conf), label))
        return out


class AssetTypeClassifier:
    """Logistic regression on the backbone's global embedding (training/train_asset_type_classifier.py)."""

    def __init__(self, path: Path):
        d = np.load(path, allow_pickle=False)
        self.coef, self.intercept = d["coef"], d["intercept"]
        self.classes = [str(c) for c in d["classes"]]
        self.backbone = str(d["backbone"])

    def predict(self, embedding: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        logits = self.coef @ embedding + self.intercept
        p = np.exp(logits - logits.max())
        p /= p.sum()
        i = int(p.argmax())
        return self.classes[i], float(p[i]), {c: float(v) for c, v in zip(self.classes, p)}


class ClipZeroShot:
    """CLIP image–text similarity against per-type prompts plus a "none" class (empty / blurry frames)."""

    def __init__(self, ccfg: Dict[str, Any], device: str = "cpu"):
        import clip
        import torch

        self.torch = torch
        self.model, self.preprocess = clip.load(ccfg["model"], device=device, download_root=str(resolve_path(ccfg["download_root"])))
        self.device = device
        self.names = [(k, p) for k, prompts in ccfg["prompts"].items() for p in prompts]
        with torch.no_grad():
            t = self.model.encode_text(clip.tokenize([p for _, p in self.names]).to(device)).float()
        self.text = t / t.norm(dim=-1, keepdim=True)
        self._lock = threading.Lock()

    def predict(self, frame: Image.Image) -> Dict[str, float]:
        with self._lock, self.torch.no_grad():
            f = self.model.encode_image(self.preprocess(frame)[None].to(self.device)).float()
            f = f / f.norm(dim=-1, keepdim=True)
            p = (100.0 * f @ self.text.T).softmax(-1)[0].cpu().numpy()
        out: Dict[str, float] = {}
        for (k, _), v in zip(self.names, p):
            out[k] = out.get(k, 0.0) + float(v)
        return out


class AutoDetector(Detector):
    """YOLO-World boxes when it finds objects; otherwise the whole frame, typed automatically.

    Type = trained classifier on the DINOv2 embedding, fused with zero-shot CLIP when enabled:
    p = w·CLIP + (1 − w)·classifier. A frame is not an asset when max p < min_confidence or
    CLIP's "none" probability ≥ max_none (empty, blurry, hand in front of the camera …).
    """

    def __init__(self, config: Dict[str, Any], embedder, yolo: Optional[Detector] = None):
        c = cfg(config, "asset_type_classifier")
        self.classifier = AssetTypeClassifier(resolve_path(c["checkpoint"]))
        if self.classifier.backbone != embedder.active_model_name:
            raise ValueError(
                f"asset type classifier was trained on {self.classifier.backbone}, embedder is "
                f"{embedder.active_model_name}: re-run training/train_asset_type_classifier.py"
            )
        self.embedder = embedder
        self.yolo = yolo
        self.min_confidence = float(c["min_confidence"])
        # Once the type is clear the whole frame IS the asset: like full_frame mode, its detection
        # confidence is fixed, so type uncertainty does not scale the defect severity (Cf).
        self.frame_confidence = float(cfg(config, "detection", "full_frame_confidence"))
        clip_cfg = c.get("clip") or {}
        self.clip = ClipZeroShot(clip_cfg) if clip_cfg.get("enabled") else None
        self.clip_weight = float(clip_cfg.get("weight", 0.0))
        self.clip_max_none = float(clip_cfg.get("max_none", 1.0))
        self.last_scores: Dict[str, float] = {}

    def classify(self, frame: Image.Image) -> Tuple[Optional[str], float, Dict[str, float]]:
        _, _, learned = self.classifier.predict(self.embedder.encode_image(frame))
        if self.clip is None:
            scores = learned
        else:
            zero_shot = self.clip.predict(frame)
            w = self.clip_weight
            scores = {k: w * zero_shot.get(k, 0.0) + (1 - w) * v for k, v in learned.items()}
            scores["none"] = zero_shot.get("none", 0.0)
            if scores["none"] >= self.clip_max_none:
                return None, scores["none"], scores
        best = max(learned, key=lambda k: scores[k])
        if scores[best] < self.min_confidence:
            return None, scores[best], scores
        return best, scores[best], scores

    def detect(self, frame: Image.Image) -> List[Detection]:
        if self.yolo is not None:
            found = self.yolo.detect(frame)
            if found:
                return found
        asset_type, p, self.last_scores = self.classify(frame)
        if asset_type is None:
            return []
        return [Detection((0, 0, frame.width, frame.height), asset_type, self.frame_confidence, "auto_full_frame")]


def build_detector(config: Dict[str, Any], embedder=None) -> Detector:
    dcfg = config["detection"]
    if dcfg["mode"] == "full_frame":
        return FullFrameDetector(dcfg["full_frame_asset_type"], float(dcfg["full_frame_confidence"]))
    if dcfg["mode"] == "yolo_world":
        return YoloWorldDetector(config)
    if dcfg["mode"] == "auto":
        if embedder is None:
            raise ValueError("detection.mode auto needs the embedding model")
        return AutoDetector(config, embedder, YoloWorldDetector(config) if dcfg.get("auto_use_yolo", False) else None)
    raise ValueError(f"Unknown detection.mode: {dcfg['mode']}")
