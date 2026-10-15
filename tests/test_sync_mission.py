"""Cloud sync (FR-1/2/3/6/7, NFR-3), mission lifecycle (FR-4/18/20), escalation (FR-19)."""

import time

import pytest
from fastapi.testclient import TestClient

from auth.security import hash_password
from cloud.main import create_app
from memory.cloud_store_interface import CloudStoreInterface, CloudUnavailable, HttpCloudStore
from memory.store import LocalStore
from memory.sync import SyncManager
from ml.detector import FullFrameDetector
from ml.escalation import EscalationService
from ml.mission import MissionService
from ml.reporting import render_html, render_pdf
from ml.semantic_memory import SemanticMemoryEngine
from tests.conftest import add_crack, make_panel

SITE = "site_s"


class InProcessCloud(HttpCloudStore):
    """HTTP client that talks to the in-process cloud app; `up` simulates connectivity."""

    def __init__(self, app, username, password):
        super().__init__("http://cloud.test", username, password)
        self.app = app
        self.up = True
        self.push_calls = 0

    def _client(self, timeout):
        if not self.up:
            raise CloudUnavailable("network down")
        return TestClient(self.app, base_url="http://cloud.test")

    def is_available(self, timeout):
        return self.up and super().is_available(timeout)

    def push_changes(self, changes, timeout):
        self.push_calls += 1
        return super().push_changes(changes, timeout)


class SlowCloud(CloudStoreInterface):
    def __init__(self, delay):
        self.delay = delay

    def is_available(self, timeout):
        return True

    def pull_site(self, site_id, timeout):
        time.sleep(self.delay)
        return {}

    def push_changes(self, changes, timeout):
        time.sleep(self.delay)
        return []


@pytest.fixture
def cloud_app(config):
    app = create_app(config, secret="test-cloud-secret")
    app.state.seed_user("sync", "sync-pass")
    return app


def _stack(config, embedder, classifier, cloud, db_name="local.db"):
    config = {**config, "storage": {**config["storage"], "db_url": config["storage"]["db_url"].replace("local.db", db_name)}}
    store = LocalStore(config)
    store.ensure_site(SITE, "Sync site")
    engine = SemanticMemoryEngine(config, store, embedder, classifier, FullFrameDetector("solar_panel", 0.95))
    sync = SyncManager(config, store, cloud)
    esc = EscalationService(config, store)
    engine.add_listener(esc.on_event)
    return store, engine, sync, MissionService(store, engine, sync, esc)


def test_offline_queue_then_flush_exactly_once(config, embedder, classifier, cloud_app):
    cloud = InProcessCloud(cloud_app, "sync", "sync-pass")
    cloud.up = False
    store, engine, sync, mission = _stack(config, embedder, classifier, cloud)
    started = mission.start(SITE, "op")
    assert started["sync"]["mode"] == "offline" and started["session"]["mode"] == "offline"  # FR-1, FR-3
    sid = started["session"]["session_id"]
    engine.observe(make_panel(1), "solar_panel", sid)
    report = mission.end(sid)
    assert report["sync"]["online"] is False and report["sync"]["queued"] > 0  # FR-7 queued
    queued = store.outbox_counts()["pending"]

    cloud.up = True
    assert sync.flush_once() == queued  # FR-7 sync at next connection
    assert store.outbox_counts()["pending"] == 0
    assert sync.flush_once() == 0
    stats = TestClient(cloud_app).get(
        "/cloud/stats", headers={"Authorization": f"Bearer {cloud._token}"}
    ).json()
    assert stats["applied_changes"] == queued  # each change applied exactly once

    # Re-sending an already-acked batch is idempotent on the cloud side.
    from memory import db
    with store.Session() as s:
        rows = [{c.name: getattr(r, c.name) for c in r.__table__.columns} for r in s.query(db.SyncOutbox).limit(5)]
    acked = cloud.push_changes(rows, 5.0)
    assert len(acked) == len(rows)
    stats2 = TestClient(cloud_app).get("/cloud/stats", headers={"Authorization": f"Bearer {cloud._token}"}).json()
    assert stats2["applied_changes"] == queued


def test_online_end_sync_and_pull_on_second_device(config, embedder, classifier, cloud_app):
    cloud = InProcessCloud(cloud_app, "sync", "sync-pass")
    store, engine, sync, mission = _stack(config, embedder, classifier, cloud)
    sid = mission.start(SITE, "op")["session"]["session_id"]
    first = engine.observe(make_panel(2), "solar_panel", sid)
    report = mission.end(sid)
    assert report["sync"]["online"] and report["sync"]["queued"] == 0  # FR-6

    store2, engine2, sync2, mission2 = _stack(config, embedder, classifier, cloud, "device2.db")
    started = mission2.start(SITE, "op2")  # FR-2 pull
    assert started["sync"]["pulled"]["asset"] >= 1
    assert store2.get_asset(first.asset_id) is not None
    ev = engine2.observe(make_panel(2), "solar_panel", started["session"]["session_id"])
    assert ev.asset_id == first.asset_id  # pulled prototypes support re-ID


def test_nfr3_slow_cloud_times_out_to_offline(config, embedder, classifier):
    config = {**config, "cloud": {**config["cloud"], "sync_timeout_s": 0.5}}
    store, engine, sync, mission = _stack(config, embedder, classifier, SlowCloud(3.0))
    t0 = time.monotonic()
    started = mission.start(SITE, "op")
    elapsed = time.monotonic() - t0
    assert started["sync"]["timed_out"] and started["session"]["mode"] == "offline"
    assert elapsed < 0.5 + 0.5


def test_fr4_briefing_and_fr18_report(config, embedder, classifier):
    store, engine, sync, mission = _stack(config, embedder, classifier, None)
    sid = mission.start(SITE, "op")["session"]["session_id"]
    a = engine.observe(make_panel(3), "solar_panel", sid)
    b = engine.observe(make_panel(4), "solar_panel", sid)
    mission.end(sid)
    sid2 = mission.start(SITE, "op")
    brief = sid2["briefing"]
    assert {x["asset_id"] for x in brief["assets_due"]} == {a.asset_id, b.asset_id}
    rs = [x["revisit_priority"] for x in brief["assets_due"]]
    assert rs == sorted(rs, reverse=True)
    s2 = sid2["session"]["session_id"]
    engine.observe(add_crack(make_panel(3)), "solar_panel", s2)
    engine.observe(make_panel(4), "solar_panel", s2)
    engine.observe(make_panel(5), "solar_panel", s2)
    report = mission.end(s2)
    assert [f["asset_id"] for f in report["flagged_defects"]] == [a.asset_id]
    assert [c["asset_id"] for c in report["confirmed_assets"]] == [b.asset_id]
    assert len(report["new_assets"]) == 1
    assert store.get_session(s2)["assets_processed"] == 3
    assert mission.start(SITE, "op")["briefing"]["unresolved_defects"]
    assert render_pdf(report)[:4] == b"%PDF"
    assert a.asset_id in render_html(report)


def test_fr19_escalation_only_to_subscribed_maintenance(config, embedder, classifier):
    # The weight-free test classifier is low-confidence (Cf ≈ 0.55), so the critical level is
    # lowered to make the synthetic crack critical (and its extent ignored); the escalation path itself is unchanged.
    sev = config["severity"]
    config = {**config, "severity": {**sev, "levels": {**sev["levels"], "critical": 0.3}, "extent": {**sev["extent"], "enabled": False}}}
    store, engine, sync, mission = _stack(config, embedder, classifier, None)
    store.ensure_site("other", "Other site")
    pw = hash_password("x")
    maint = store.create_user("m1", pw, "maintenance")
    other_maint = store.create_user("m2", pw, "maintenance")
    op = store.create_user("o1", pw, "operator")
    admin = store.create_user("a1", pw, "admin")
    store.set_subscriptions(maint["user_id"], [SITE])
    store.set_subscriptions(other_maint["user_id"], ["other"])
    store.set_subscriptions(op["user_id"], [SITE])
    store.set_subscriptions(admin["user_id"], [SITE])

    sid = mission.start(SITE, "op")["session"]["session_id"]
    engine.observe(make_panel(9), "solar_panel", sid)
    mission.end(sid)
    sid = mission.start(SITE, "op")["session"]["session_id"]
    ev = engine.observe(add_crack(make_panel(9), 14), "solar_panel", sid, detector_confidence=1.0)
    assert ev.critical, ev.severity_score
    notes = store.all_notifications()
    assert {n["user_id"] for n in notes} == {maint["user_id"]}
    report = mission.end(sid)
    assert report["counts"]["escalations"] == 1


def test_quick_start_reuses_a_nearby_site_or_creates_one(config, embedder, classifier, monkeypatch):
    import ml.mission as mission_mod

    monkeypatch.setattr(mission_mod, "place_name", lambda lat, lon, cfg: "Tambaram, Chennai")
    store, engine, sync, mission = _stack(config, embedder, classifier, None)
    here = {"latitude": 12.9246, "longitude": 80.1270, "accuracy_m": 30.0}
    first = mission.quick_start("op", here)
    q = first["quick_start"]
    assert q["created"] and q["site"]["name"] == "Tambaram, Chennai"
    sid = first["session"]["session_id"]
    asset = engine.observe(make_panel(6), "solar_panel", sid)
    mission.end(sid)

    # ~50 m away, later: the same site, so the asset is recognised.
    again = mission.quick_start("op", {"latitude": 12.92505, "longitude": 80.1270, "accuracy_m": 30.0})
    assert not again["quick_start"]["created"] and again["session"]["site_id"] == q["site"]["site_id"]
    ev = engine.observe(make_panel(6), "solar_panel", again["session"]["session_id"])
    assert ev.asset_id == asset.asset_id and ev.history is not None
    mission.end(again["session"]["session_id"])

    far = mission.quick_start("op", {"latitude": 13.0806, "longitude": 80.2773})   # ~20 km away
    assert far["quick_start"]["created"] and far["session"]["site_id"] != q["site"]["site_id"]
