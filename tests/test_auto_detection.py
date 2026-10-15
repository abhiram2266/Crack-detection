"""Automatic asset typing (AutoDetector) and the report's comparison with the previous inspection."""

import numpy as np
import pytest

from memory.store import LocalStore
from ml.detector import AutoDetector, Detection, Detector, FullFrameDetector
from ml.reporting import build_report
from ml.schemas import utc_now
from ml.semantic_memory import SemanticMemoryEngine
from tests.conftest import add_crack, add_dust, make_panel


def _classifier_file(tmp_path, embedder, target, classes=("building", "solar_panel")):
    """A classifier whose decision is the sign of the cosine to `target`'s embedding."""
    dim = embedder.embedding_dim
    coef = np.zeros((len(classes), dim), np.float32)
    coef[0] = embedder.encode_image(target) * 40.0
    path = tmp_path / "asset_type.npz"
    np.savez(path, coef=coef, intercept=np.array([-20.0, 0.0], np.float32), classes=np.array(classes),
             backbone=np.array(embedder.active_model_name))
    return path


def _config(config, path, min_conf=0.6):
    config["asset_type_classifier"]["checkpoint"] = str(path)
    config["asset_type_classifier"]["min_confidence"] = min_conf
    config["asset_type_classifier"]["clip"]["enabled"] = False   # CLIP path tested separately below
    return config


class _Yolo(Detector):
    def __init__(self, found):
        self.found = found

    def detect(self, frame):
        return list(self.found)


def test_close_up_typed_without_operator_choice(tmp_path, config, embedder):
    wall = make_panel(3)
    det = AutoDetector(_config(config, _classifier_file(tmp_path, embedder, wall)), embedder)
    (d,) = det.detect(wall)
    assert d.asset_type == "building" and d.label == "auto_full_frame"
    assert d.bbox == (0, 0, wall.width, wall.height)
    assert d.confidence == config["detection"]["full_frame_confidence"]   # type certainty does not scale S


def test_low_confidence_frame_is_not_an_asset(tmp_path, config, embedder):
    det = AutoDetector(_config(config, _classifier_file(tmp_path, embedder, make_panel(3)), min_conf=0.999), embedder)
    det.classifier.coef[:] = 0.0          # 50/50 between the two types
    det.classifier.intercept[:] = 0.0
    assert det.detect(make_panel(4)) == []


def test_yolo_boxes_win_when_present(tmp_path, config, embedder):
    box = Detection((1, 2, 30, 40), "utility_pole", 0.9, "utility pole")
    det = AutoDetector(_config(config, _classifier_file(tmp_path, embedder, make_panel(3))), embedder, _Yolo([box]))
    assert det.detect(make_panel(3)) == [box]


def test_backbone_mismatch_rejected(tmp_path, config, embedder):
    path = _classifier_file(tmp_path, embedder, make_panel(3))
    d = dict(np.load(path))
    d["backbone"] = np.array("some_other_backbone")
    np.savez(path, **d)
    with pytest.raises(ValueError, match="re-run"):
        AutoDetector(_config(config, path), embedder)


def test_report_compares_with_previous_inspection(config, embedder, classifier):
    store = LocalStore(config)
    store.ensure_site("s", "S")
    engine = SemanticMemoryEngine(config, store, embedder, classifier, FullFrameDetector("solar_panel", 0.95))
    s1 = store.create_session("s", "op", "offline", {})["session_id"]
    first = engine.observe(make_panel(9), "solar_panel", s1)
    store.update_session(s1, status="ended", ended_ts=utc_now().isoformat())
    s2 = store.create_session("s", "op", "offline", {})["session_id"]
    again = engine.observe(add_crack(make_panel(9)), "solar_panel", s2)
    assert again.asset_id == first.asset_id

    report = build_report(store, s2, {}, [])
    (r,) = report["revisited_assets"]
    assert r["asset_id"] == first.asset_id
    assert r["previous_inspection_ts"] == first.timestamp
    assert r["previous_condition"] == first.condition["label"]
    assert r["severity_delta"] == pytest.approx(again.severity_score - first.severity_score, abs=1e-4)
    assert report["counts"]["revisited"] == 1
    assert build_report(store, s1, {}, [])["revisited_assets"] == []    # first flight: nothing to compare


def test_unchanged_defect_keeps_its_severity_and_worsening_is_rescored(config, embedder, classifier):
    """Same image at a later visit → same severity, reported as unchanged since the last visit."""
    store = LocalStore(config)
    store.ensure_site("s", "S")
    engine = SemanticMemoryEngine(config, store, embedder, classifier, FullFrameDetector("solar_panel", 0.95))

    def visit(img):
        sid = store.create_session("s", "op", "offline", {})["session_id"]
        ev = engine.observe(img, "solar_panel", sid)
        store.update_session(sid, status="ended", ended_ts=utc_now().isoformat())
        return ev

    cracked = add_crack(make_panel(12))
    first = visit(cracked)
    assert first.type == "new_asset" and first.severity_score > 0
    for _ in range(2):
        again = visit(cracked)
        assert again.asset_id == first.asset_id
        since = again.change["since_last_visit"]
        assert not since["significant"] and since["previous_condition"] == first.condition["label"]
        assert again.severity_score == pytest.approx(first.severity_score)

    # A healthy asset that cracks later: changed since the last visit → scored afresh, not carried over.
    healthy = visit(make_panel(13))
    cracked_later = visit(add_crack(make_panel(13)))
    assert cracked_later.asset_id == healthy.asset_id
    assert cracked_later.change["since_last_visit"]["significant"]
    assert cracked_later.severity_score > healthy.severity_score
