"""Persistent semantic memory engine.

Per detected asset:
  crop → embedding → condition classifier → identity cascade → change
  detector → gated severity → one atomic Local DB commit → events.

Identity (re-ID) and condition are separate models (D2). Severity uses the
observation confidence min(detector, classifier) (D1) and the M-gate (D4).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

from memory.store import LocalStore
from memory.vector_memory import PrototypeIndex
from ml.asset_identity import AssetIdentifier, GeoLocation, IdentityResult
from ml.change_detector import (
    ChangeAssessment,
    ChangeDetector,
    ConditionVisualChangeDetector,
    ReferenceCondition,
    render_heatmap,
)
from ml.condition_classifier import NORMAL_LABEL, ConditionClassifier, ConditionPrediction
from ml.config import cfg, resolve_path
from ml.context_risk import ContextRiskService, effective_criticality, summary as context_summary
from ml.detector import Detection, Detector
from ml.embedding_model import AssetEmbeddingModel
from ml.imaging import crop as crop_image
from ml.schemas import new_id, utc_now_iso
from ml.severity import (
    asset_criticality,
    gated_severity,
    is_critical,
    is_priority,
    observation_confidence,
    severity_level,
)
from ml.tracker import SessionTracker

logger = logging.getLogger(__name__)


@dataclass
class AssetEvent:
    type: str                      # new_asset | confirmed | change | duplicate | tracked | error
    session_id: Optional[str]
    timestamp: str
    asset_type: str
    bbox: List[int]
    asset_id: Optional[str] = None
    track_id: Optional[int] = None
    identity: Dict[str, Any] = field(default_factory=dict)
    condition: Dict[str, Any] = field(default_factory=dict)
    change: Optional[Dict[str, Any]] = None
    detector_confidence: float = 0.0
    observation_confidence: float = 0.0
    severity_score: float = 0.0
    severity_level: str = "low"
    priority: bool = False
    critical: bool = False
    persisted: bool = False
    observation_id: Optional[str] = None
    defect: Optional[Dict[str, Any]] = None
    revisit: Optional[Dict[str, float]] = None
    history: Optional[Dict[str, Any]] = None     # FR-12: what memory held for a matched asset
    context: Optional[Dict[str, Any]] = None     # site context risk applied to criticality (ml/context_risk.py)
    image_reference: Optional[str] = None
    heatmap_reference: Optional[str] = None
    latency_ms: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


Listener = Callable[[AssetEvent], None]


def _reference_from_prediction(p: ConditionPrediction) -> Dict[str, Any]:
    return {"label": p.label, "scores": p.scores, "is_defect": p.is_defect, "defect_type": p.defect_type}


def _implied_normal(labels: Sequence[str]) -> Dict[str, Any]:
    """Reference for a never-seen asset: assume it was commissioned in normal condition."""
    return {
        "label": NORMAL_LABEL,
        "scores": {l: (1.0 if l == NORMAL_LABEL else 0.0) for l in labels},
        "is_defect": False,
        "defect_type": None,
    }


class SemanticMemoryEngine:
    def __init__(
        self,
        config: Dict[str, Any],
        store: LocalStore,
        embedder: AssetEmbeddingModel,
        classifier: ConditionClassifier,
        detector: Detector,
        change_detector: Optional[ChangeDetector] = None,
    ):
        self.config = config
        self.store = store
        self.embedder = embedder
        self.classifier = classifier
        self.detector = detector
        self.change_detector = change_detector or ConditionVisualChangeDetector(config, embedder)
        m = cfg(config, "matching")
        self.index = PrototypeIndex(
            embedder.embedding_dim,
            cfg(config, "memory", "vector_backend"),
            int(m["max_prototypes_per_asset"]),
            float(m["prototype_novelty"]),
        )
        self.identifier = AssetIdentifier(config, self.index, self._load_prototype_image)
        self.resolution_missions = int(cfg(config, "defects", "resolution_suggestion_missions"))
        self.jpeg_quality = int(cfg(config, "storage", "image_jpeg_quality"))
        self.heatmap_dir = resolve_path(cfg(config, "change_detection", "heatmap_dir"))
        self.heatmap_opacity = float(cfg(config, "change_detection", "heatmap_opacity"))
        self._listeners: List[Listener] = []
        self._session_cache: Dict[str, Dict[str, Any]] = {}
        self._site_cache: Dict[str, Any] = {}
        self.context: Optional[ContextRiskService] = None   # set by the service container
        self._lock = threading.RLock()
        self._sync_backbone()
        self._load_index()

    # ── setup ─────────────────────────────────────────────────────────────
    def _sync_backbone(self) -> None:
        """Re-embed stored prototypes if the backbone changed since they were written (D12)."""
        current = {"name": self.embedder.active_model_name, "dim": self.embedder.embedding_dim}
        stored = self.store.get_meta("backbone")
        if stored == current:
            return
        protos = self.store.all_prototypes()
        if protos and stored is not None:
            logger.warning("Backbone changed %s → %s; re-embedding %d prototypes", stored, current, len(protos))
            from memory import db

            with self.store.txn() as s:
                for p in protos:
                    path = self.store.image_path(p["image_reference"])
                    row = s.get(db.AssetPrototype, p["proto_id"])
                    if not path.exists():
                        s.delete(row)
                        continue
                    with Image.open(path) as im:
                        vec = self.embedder.encode_image(im.convert("RGB"))
                    row.embedding = vec.astype(np.float32).tobytes()
                    row.dim = int(vec.shape[0])
        self.store.set_meta("backbone", current)

    def _load_index(self) -> None:
        for p in self.store.all_prototypes():
            if p["embedding"].shape[0] != self.embedder.embedding_dim:
                logger.warning("Skipping prototype %s with dim %d", p["proto_id"], p["embedding"].shape[0])
                continue
            self.index.add(p["proto_id"], p["asset_id"], p["site_id"], p["asset_type"], p["embedding"])
        logger.info("Prototype index: %d prototypes for %d assets (%s)", len(self.index), self.index.asset_count(), self.index.backend_name)

    def reload_index(self) -> None:
        with self._lock:
            self.index = PrototypeIndex(
                self.embedder.embedding_dim,
                cfg(self.config, "memory", "vector_backend"),
                self.index.max_prototypes,
                self.index.novelty,
            )
            self.identifier.index = self.index
            self._site_cache.clear()
            self._load_index()

    def prefetch_references(self, site_id: str, stop: Optional[threading.Event] = None) -> int:
        """Extract reference keypoints for the site's prototypes, highest revisit priority first.

        Bounded by `matching.feature_cache_size`, so matching during the flight does not pay
        the extraction cost on the first sighting of a known asset (NFR-2).
        """
        known = self.index.asset_ids(site_id)
        order = [a["asset_id"] for a in self.store.list_assets(site_id) if a["asset_id"] in known]
        budget = int(cfg(self.config, "matching", "feature_cache_size"))
        done = 0
        for aid in order:
            for proto in self.index.prototypes(aid):
                if done >= budget or (stop is not None and stop.is_set()):
                    return done
                if self.identifier.reference_features(proto.proto_id) is not None:
                    done += 1
        return done

    def start_prefetch(self, site_id: str) -> threading.Thread:
        t = threading.Thread(target=self.prefetch_references, args=(site_id,), name=f"prefetch-{site_id}", daemon=True)
        t.start()
        return t

    def _load_prototype_image(self, proto_id: int) -> Optional[Image.Image]:
        path = self.store.prototype_image_path(proto_id)
        if path is None or not path.exists():
            return None
        with Image.open(path) as im:
            return im.convert("RGB")

    def add_listener(self, fn: Listener) -> None:
        self._listeners.append(fn)

    def remove_listener(self, fn: Listener) -> None:
        if fn in self._listeners:
            self._listeners.remove(fn)

    def _emit(self, event: AssetEvent) -> None:
        for fn in list(self._listeners):
            try:
                fn(event)
            except Exception:  # noqa: BLE001 - a failing listener must not break inspection
                logger.exception("Event listener failed")

    def _session(self, session_id: str) -> Dict[str, Any]:
        sess = self._session_cache.get(session_id)
        if sess is None:
            sess = self.store.get_session(session_id)
            if sess is None:
                raise KeyError(f"Unknown session: {session_id}")
            self._session_cache[session_id] = sess
        if sess["status"] != "active":
            fresh = self.store.get_session(session_id)
            self._session_cache[session_id] = fresh
            sess = fresh
        return sess

    def _site_context(self, site_id: str):
        ctx = self._site_cache.get(site_id)
        if ctx is None:
            ctx = (self.store.asset_refs(site_id), self.store.marker_table(site_id))
            self._site_cache[site_id] = ctx
        return ctx

    # ── public API ────────────────────────────────────────────────────────
    def process_frame(
        self,
        frame: Image.Image,
        session_id: str,
        tracker: Optional[SessionTracker] = None,
        location: Optional[Dict[str, float]] = None,
        timestamp: Optional[str] = None,
        force_track_id: Optional[int] = None,
    ) -> List[AssetEvent]:
        """Detect → track → process each new/changed track once (D8)."""
        t0 = time.perf_counter()
        detections = self.detector.detect(frame)
        t_det = (time.perf_counter() - t0) * 1000
        if not detections:
            return []
        crops = [crop_image(frame, d.bbox) for d in detections]
        t1 = time.perf_counter()
        embeddings = self.embedder.encode_batch(crops)
        t_emb = (time.perf_counter() - t1) * 1000
        tracks = (
            tracker.update([d.bbox for d in detections], [d.asset_type for d in detections], embeddings, crops)
            if tracker else [None] * len(detections)
        )

        events: List[AssetEvent] = []
        for det, crop, emb, track in zip(detections, crops, embeddings, tracks):
            forced = track is not None and force_track_id == track.track_id
            if track is not None and track.observed and not forced:
                pred = self.classifier.predict_embeddings(emb[None, :], det.asset_type)[0]
                if pred.label == track.last_label:
                    events.append(self._tracked_event(det, track, session_id, timestamp))
                    continue
            ev = self._process(frame, det, crop, emb, session_id, location, timestamp, True, track, forced)
            ev.latency_ms.update({"detect": t_det, "embed_batch": t_emb})
            events.append(ev)
        return events

    def observe(
        self,
        image: Image.Image,
        asset_type: str,
        session_id: Optional[str],
        persist: bool = True,
        detector_confidence: Optional[float] = None,
        location: Optional[Dict[str, float]] = None,
        timestamp: Optional[str] = None,
        force: bool = False,
    ) -> AssetEvent:
        """Process a single close-up image of one asset (whole image = the asset)."""
        conf = float(cfg(self.config, "detection", "full_frame_confidence")) if detector_confidence is None else detector_confidence
        det = Detection((0, 0, image.width, image.height), asset_type, conf, "image")
        emb = self.embedder.encode_image(image)
        return self._process(image, det, image, emb, session_id, location, timestamp, persist, None, force)

    # ── core ──────────────────────────────────────────────────────────────
    def _tracked_event(self, det: Detection, track, session_id, timestamp) -> AssetEvent:
        info = track.info
        return AssetEvent(
            type="tracked",
            session_id=session_id,
            timestamp=timestamp or utc_now_iso(),
            asset_type=det.asset_type,
            bbox=list(det.bbox),
            asset_id=track.asset_id,
            track_id=track.track_id,
            condition=info.get("condition", {}),
            detector_confidence=det.confidence,
            severity_score=info.get("severity_score", 0.0),
            severity_level=info.get("severity_level", "low"),
            priority=info.get("priority", False),
            critical=info.get("critical", False),
        )

    def _process(
        self,
        frame: Image.Image,
        det: Detection,
        crop: Image.Image,
        embedding: np.ndarray,
        session_id: Optional[str],
        location: Optional[Dict[str, float]],
        timestamp: Optional[str],
        persist: bool,
        track,
        force: bool,
    ) -> AssetEvent:
        ts = timestamp or utc_now_iso()
        lat: Dict[str, float] = {}
        t_start = time.perf_counter()
        ev = AssetEvent(
            type="error",
            session_id=session_id,
            timestamp=ts,
            asset_type=det.asset_type,
            bbox=list(det.bbox),
            track_id=track.track_id if track else None,
            detector_confidence=det.confidence,
        )
        try:
            if not self.classifier.supports(det.asset_type):
                raise ValueError(f"No condition model for asset_type '{det.asset_type}'")
            session = self._session(session_id) if session_id else None
            if persist and (session is None or session["status"] != "active"):
                raise ValueError("Persisting an observation requires an active session (D7)")
            site_id = session["site_id"] if session else cfg(self.config, "sites", "default_site_id")

            t = time.perf_counter()
            pred = self.classifier.predict_embeddings(embedding[None, :], det.asset_type)[0]
            lat["classify"] = (time.perf_counter() - t) * 1000
            cf = observation_confidence(det.confidence, pred.confidence)
            ev.condition = pred.to_dict()
            ev.observation_confidence = cf

            with self._lock:
                t = time.perf_counter()
                refs, markers = self._site_context(site_id)
                geo = GeoLocation.from_dict(location)
                identity: IdentityResult = self.identifier.identify(
                    frame, det.bbox, crop, embedding, det.asset_type, site_id, geo, refs, markers
                )
                lat["identity"] = (time.perf_counter() - t) * 1000
                ev.identity = identity.to_dict()

                if not identity.is_new and persist and session_id and not force:
                    last = self.store.last_session_observation(session_id, identity.asset_id)
                    if last is not None and last["condition_description"] == pred.label:
                        ev.type = "duplicate"
                        ev.asset_id = identity.asset_id
                        ev.severity_score = last["severity_score"]
                        ev.severity_level = last["severity_level"]
                        self._update_track(track, ev, pred)
                        lat["total"] = (time.perf_counter() - t_start) * 1000
                        ev.latency_ms = lat
                        return ev

                labels = self.classifier.label_maps[det.asset_type]
                t = time.perf_counter()
                if identity.is_new:
                    # A healthy-looking new asset is its own baseline (M = 0); a defective one is
                    # compared with the implied commissioning state so the defect is scored.
                    ref_dict = _implied_normal(labels) if pred.is_defect else _reference_from_prediction(pred)
                    reference_image = None
                    homography = None
                else:
                    asset_row = self.store.get_asset(identity.asset_id)
                    ev.history = self._history_summary(asset_row)
                    ref_dict = asset_row["reference_condition"]
                    ref_proto = identity.reference_proto_id or asset_row["reference_proto_id"]
                    reference_image = self._load_prototype_image(ref_proto) if ref_proto else None
                    homography = identity.homography if identity.reference_proto_id == ref_proto else None
                reference = ReferenceCondition(
                    ref_dict["label"], ref_dict["scores"], ref_dict["is_defect"], ref_dict["defect_type"]
                )
                change: ChangeAssessment = self.change_detector.assess(pred, reference, crop, reference_image, homography)
                # Change since the PREVIOUS inspection (the reference above is the last confirmed state,
                # which for a known defect is its healthy baseline).
                previous = None if identity.is_new else self.store.latest_observation(identity.asset_id)
                since_last = self._since_last(pred, crop, previous) if previous else None
                lat["change"] = (time.perf_counter() - t) * 1000

                t = time.perf_counter()
                extent = self._defect_extent(crop, pred, det.asset_type) if pred.is_defect else None
                lat["extent"] = (time.perf_counter() - t) * 1000

                site_context = self.context.for_site(site_id) if self.context else None
                criticality = effective_criticality(asset_criticality(det.asset_type, self.config), site_context)
                ev.context = context_summary(site_context)
                severity, significant = gated_severity(change.magnitude, criticality, cf, self.config, extent)
                ev.change = change.to_dict()
                ev.change["extent"] = extent
                if since_last is not None:
                    prev_extent = (previous.get("meta") or {}).get("extent")
                    growth = extent - prev_extent if extent is not None and prev_extent is not None else None
                    if growth is not None and growth >= float(cfg(self.config, "severity", "extent", "growth_threshold")):
                        since_last.significant = True   # same defect type, but it has grown
                    ev.change["since_last_visit"] = {
                        "change_magnitude": since_last.magnitude,
                        "condition_delta": since_last.condition_delta,
                        "visual_delta": since_last.visual_delta,
                        "significant": since_last.significant,
                        "previous_ts": previous["observation_ts"],
                        "previous_condition": previous["condition_description"],
                        "previous_severity": previous["severity_score"],
                        "previous_extent": prev_extent,
                        "extent_growth": growth,
                    }
                    if significant and not since_last.significant:
                        # Known defect, unchanged since the last visit: it keeps the severity it had.
                        severity = float(previous["severity_score"])
                ev.severity_score = severity
                ev.severity_level = severity_level(severity, self.config)
                ev.priority = is_priority(severity, self.config)
                ev.critical = is_critical(severity, self.config)
                if identity.is_new:
                    ev.type = "new_asset"
                    status = "new"
                else:
                    ev.type = "change" if significant else "confirmed"
                    status = "changed" if significant else "confirmed"
                    ev.asset_id = identity.asset_id

                if persist:
                    t = time.perf_counter()
                    heat_base = reference_image if change.warped else crop
                    self._persist(
                        ev, det, crop, embedding, pred, identity, change, status, significant, session, site_id,
                        geo, criticality, cf, heat_base, labels,
                    )
                    lat["store"] = (time.perf_counter() - t) * 1000
                self._update_track(track, ev, pred)
        except Exception as exc:  # noqa: BLE001 - errors are reported per asset, never crash a mission
            logger.exception("Processing failed")
            ev.type = "error"
            ev.error = str(exc)
        lat["total"] = (time.perf_counter() - t_start) * 1000
        ev.latency_ms = lat
        self._emit(ev)
        return ev

    def _update_track(self, track, ev: AssetEvent, pred: ConditionPrediction) -> None:
        if track is None:
            return
        track.observed = True
        track.asset_id = ev.asset_id
        track.identity_stage = ev.identity.get("stage")
        track.last_label = pred.label
        track.info = {
            "condition": ev.condition,
            "severity_score": ev.severity_score,
            "severity_level": ev.severity_level,
            "priority": ev.priority,
            "critical": ev.critical,
        }

    def _persist(
        self, ev, det, crop, embedding, pred, identity, change, status, significant, session, site_id,
        geo, criticality, cf, heat_base, labels,
    ) -> None:
        obs_id = new_id("obs")
        is_new = identity.is_new
        asset_id = new_id("asset") if is_new else identity.asset_id
        img_dir = self.store.asset_image_dir(site_id, asset_id)
        image_ref = f"{site_id}/{asset_id}/{obs_id}.jpg"
        crop.save(img_dir / f"{obs_id}.jpg", format="JPEG", quality=self.jpeg_quality)
        heat_ref = None
        if change.heatmap is not None:
            heat_ref = f"{site_id}/{asset_id}/{obs_id}_heat.png"
            render_heatmap(heat_base, change.heatmap, self.store.image_path(heat_ref), self.heatmap_opacity)

        add_proto, evict = (True, None) if is_new else self.index.plan_update(asset_id, embedding)
        prototype = {"embedding": embedding.tolist(), "image_reference": image_ref} if add_proto else None

        if is_new:
            reference = _reference_from_prediction(pred) if not pred.is_defect else _implied_normal(labels)
        else:
            reference = _reference_from_prediction(pred) if not significant else None

        observation = {
            "observation_id": obs_id,
            "asset_id": asset_id,
            "session_id": session["session_id"],
            "observation_ts": ev.timestamp,
            "condition_description": pred.label,
            "condition_scores": pred.scores,
            "is_defect": pred.is_defect,
            "defect_type": pred.defect_type,
            "image_reference": image_ref,
            "heatmap_reference": heat_ref,
            "severity_score": ev.severity_score,
            "severity_level": ev.severity_level,
            "change_magnitude": change.magnitude,
            "condition_delta": change.condition_delta,
            "visual_delta": change.visual_delta,
            "observation_confidence": cf,
            "detector_confidence": det.confidence,
            "classifier_confidence": pred.confidence,
            "match_stage": identity.stage,
            "match_confidence": identity.confidence,
            "status": status,
            "latitude": geo.latitude if geo else None,
            "longitude": geo.longitude if geo else None,
            "altitude": geo.altitude if geo else None,
            "bbox": list(det.bbox),
            "simulated": bool(session.get("simulated")),
            "meta": {
                "detector_label": det.label, "identity": identity.to_dict(), "context": ev.context,
                "criticality": criticality, "extent": (ev.change or {}).get("extent"),
            },
        }
        if is_new:
            asset = {
                "asset_id": asset_id,
                "site_id": site_id,
                "asset_type": det.asset_type,
                "marker_id": identity.marker_id,
                "first_observed_ts": ev.timestamp,
                "last_checked_ts": ev.timestamp,
                "current_condition": pred.label,
                "current_is_defect": pred.is_defect,
                "reference_condition": reference,
                "criticality": criticality,
                "revisit_priority_score": 0.0,
                "location_samples": 0,
                "simulated": bool(session.get("simulated")),
                "meta": {},
                "updated_ts": ev.timestamp,
            }
        else:
            asset = {"asset_id": asset_id}

        result = self.store.commit_observation(
            asset=asset,
            observation=observation,
            is_new=is_new,
            confirmed=status == "confirmed",
            reference_condition=reference,
            prototype=prototype,
            evict_proto_id=evict,
            resolution_missions=self.resolution_missions,
        )
        if result["evicted_proto_id"] is not None:
            self.index.remove(result["evicted_proto_id"])
            self.identifier.forget(result["evicted_proto_id"])
        if result["proto_id"] is not None:
            self.index.add(result["proto_id"], asset_id, site_id, det.asset_type, embedding)
        if is_new:
            self._site_cache.pop(site_id, None)

        ev.asset_id = asset_id
        ev.persisted = True
        ev.observation_id = obs_id
        ev.image_reference = image_ref
        ev.heatmap_reference = heat_ref
        ev.defect = result["defect"]
        ev.revisit = result["revisit"]

    def _defect_extent(self, crop: Image.Image, pred: ConditionPrediction, asset_type: str) -> Optional[float]:
        """Defect size E ∈ [0, 1]: mean probability of the predicted defect over a grid of overlapping tiles.

        The classifier only says whether the crop shows a crack, not how much of it; run on tiles,
        the share of the surface it flags tracks the crack's length and width (a hairline crack lights
        up one or two tiles, a long open one most of them).
        """
        if not cfg(self.config, "severity", "extent", "enabled") or pred.label not in pred.scores:
            return None
        n = int(cfg(self.config, "severity", "extent", "grid"))
        w, h = crop.size
        tw, th = 2 * w / (n + 1), 2 * h / (n + 1)
        tiles = [
            crop.crop((int(j * w / (n + 1)), int(i * h / (n + 1)), int(j * w / (n + 1) + tw), int(i * h / (n + 1) + th)))
            for i in range(n)
            for j in range(n)
        ]
        preds = self.classifier.predict_embeddings(self.embedder.encode_batch(tiles), asset_type)
        return float(np.mean([p.scores.get(pred.label, 0.0) for p in preds]))

    def _since_last(self, pred: ConditionPrediction, crop: Image.Image, previous: Dict[str, Any]) -> ChangeAssessment:
        """Change since the previous observation: M' = α'·condition + (1 − α')·aligned visual change.

        α' = change_detection.since_last_alpha. Webcam captures of the same scene vary a lot visually
        (glare, moiré, exposure), so by default (α' = 1) the condition decides and the visual part is
        reported only.
        """
        ref = ReferenceCondition(
            previous["condition_description"], previous["condition_scores"], previous["is_defect"], previous["defect_type"]
        )
        img = H = None
        if previous.get("image_reference"):
            path = self.store.image_path(previous["image_reference"])
            if path.exists():
                img = Image.open(path).convert("RGB")
                v = self.identifier.verifier
                inliers, ratio, h = v.verify(v.extract(crop), v.extract(img))
                H = h if h is not None and v.accepts(inliers, ratio) else None
        c = self.change_detector.assess(pred, ref, crop, img, H)
        alpha = float(cfg(self.config, "change_detection", "since_last_alpha"))
        b = c.visual_delta
        m = c.condition_delta if b is None else alpha * c.condition_delta + (1 - alpha) * b
        c.magnitude = float(m)
        c.significant = m >= float(cfg(self.config, "change_detection", "threshold"))
        return c

    # ── history (FR-12) ───────────────────────────────────────────────────
    def _history_summary(self, asset_row: Dict[str, Any]) -> Dict[str, Any]:
        """Historical condition, latest prior image and unresolved defect notes of a matched asset."""
        last = self.store.latest_observation(asset_row["asset_id"])
        return {
            "previous_condition": asset_row["current_condition"],
            "reference_condition": asset_row["reference_condition"]["label"],
            "last_checked_ts": asset_row["last_checked_ts"],
            "previous_image": last["image_reference"] if last else None,
            "previous_severity": last["severity_score"] if last else None,
            "open_defects": [
                {k: d[k] for k in ("defect_id", "defect_type", "status", "severity_score", "notes")}
                for d in self.store.list_defects(asset_id=asset_row["asset_id"])
                if d["status"] != "Resolved"
            ],
        }

    def get_asset_history(self, asset_id: str) -> Dict[str, Any]:
        asset = self.store.get_asset(asset_id)
        if asset is None:
            return {"asset_id": asset_id, "found": False}
        obs = self.store.observations(asset_id)
        return {
            "asset_id": asset_id,
            "found": True,
            "asset": asset,
            "observations": obs,
            "previous_images": [o["image_reference"] for o in obs],
            "previous_conditions": [o["condition_description"] for o in obs],
            "severity_scores": [o["severity_score"] for o in obs],
            "defects": self.store.list_defects(asset_id=asset_id),
            "mission_severities": self.store.mission_severities(asset_id),
        }

    def end_session_hook(self, session_id: str) -> None:
        self._session_cache.pop(session_id, None)

    def recompute_site(self, site_id: str, now: Optional[datetime] = None) -> List[Dict[str, Any]]:
        return self.store.recompute_revisit(site_id, now)
