"""Owns the (single) running live session for the API process."""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, Optional, Union

import cv2

from ml.config import cfg
from ml.detector import AutoDetector, Detector, FullFrameDetector, YoloWorldDetector
from ml.live.camera import CameraReader, FrameSource, PushFrameSource
from ml.live.overlay import OverlayRenderer
from ml.live.pipeline import LivePipeline
from ml.semantic_memory import SemanticMemoryEngine

logger = logging.getLogger(__name__)


class LiveStateError(Exception):
    """A live-pipeline request conflicts with the current pipeline state."""


class LiveManager:
    def __init__(
        self,
        config: Dict[str, Any],
        engine: SemanticMemoryEngine,
        store,
        publish: Callable[[Dict[str, Any]], None],
        camera_factory: Optional[Callable[..., FrameSource]] = None,
    ):
        self.config = config
        self.engine = engine
        self.store = store
        self.publish = publish
        self.camera_factory = camera_factory or (lambda source: CameraReader.from_config(config, source))
        self.renderer = OverlayRenderer(config)
        self.pipeline: Optional[LivePipeline] = None
        self._saved_detector: Optional[Detector] = None
        self._lock = threading.Lock()

    @property
    def session_id(self) -> Optional[str]:
        return self.pipeline.session_id if self.pipeline else None

    @property
    def running(self) -> bool:
        return self.pipeline is not None and self.pipeline.running

    def start(
        self,
        session_id: str,
        source: Optional[Union[int, str]] = None,
        mode: str = "camera",
        detection_mode: Optional[str] = None,
        asset_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            if self.running:
                raise LiveStateError(f"Live session already running for {self.session_id}")
            self._teardown()
            s = self.store.get_session(session_id)
            if s is None:
                raise KeyError(session_id)
            if s["status"] != "active":
                raise LiveStateError("Live input needs an active session")
            if detection_mode == "full_frame":
                at = asset_type or cfg(self.config, "detection", "full_frame_asset_type")
                if not self.engine.classifier.supports(at):
                    raise ValueError(f"No condition model for asset_type {at}")
                self._saved_detector = self.engine.detector
                self.engine.detector = FullFrameDetector(at, float(cfg(self.config, "detection", "full_frame_confidence")))
            elif detection_mode == "auto" and not isinstance(self.engine.detector, AutoDetector):
                use_yolo = cfg(self.config, "detection").get("auto_use_yolo", False)
                yolo = self.engine.detector if use_yolo and isinstance(self.engine.detector, YoloWorldDetector) else None
                self._saved_detector = self.engine.detector
                self.engine.detector = AutoDetector(self.config, self.engine.embedder, yolo)
            frame_source: FrameSource = PushFrameSource() if mode == "browser" else self.camera_factory(source)
            self.pipeline = LivePipeline(self.config, self.engine, frame_source, session_id, self.publish)
            self.pipeline.start()
            logger.info("Live session %s started (%s, source=%s)", session_id, mode, source)
            return self.status()

    def stop(self) -> Dict[str, Any]:
        with self._lock:
            final = self._teardown()
            return {**final, "running": False} if final else {"running": False}

    def _teardown(self) -> Optional[Dict[str, Any]]:
        """Stop the pipeline (also one whose video source already finished) and restore the detector."""
        if self.pipeline is None:
            return None
        self.pipeline.stop()
        final = self.pipeline.status()
        if self._saved_detector is not None:
            self.engine.detector = self._saved_detector
            self._saved_detector = None
        self.pipeline = None
        return final

    def push_jpeg(self, data: bytes) -> bool:
        p = self.pipeline
        if p is None or not isinstance(p.source, PushFrameSource):
            return False
        if len(data) > int(cfg(self.config, "live", "max_ws_frame_bytes")):
            return False
        return p.source.push_jpeg(data)

    def capture(self, track_id: Optional[int] = None) -> Optional[int]:
        if self.pipeline is None:
            raise LiveStateError("No live session running")
        return self.pipeline.force_capture(track_id)

    def status(self) -> Dict[str, Any]:
        if self.pipeline is None:
            return {"running": False}
        return self.pipeline.status()

    def annotated_jpeg(self) -> Optional[bytes]:
        p = self.pipeline
        if p is None:
            return None
        frame = p.source.latest()
        if frame is None:
            return None
        st = p.status()
        img = self.renderer.render(frame.image, st["tracks"], st["alerts"], st)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(cfg(self.config, "live", "jpeg_quality"))])
        return buf.tobytes() if ok else None
