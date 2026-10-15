"""Context-aware site risk (ml/context_risk.py): scoring, missing data, caching, effect on S and R."""

from datetime import datetime, timezone

import pytest

from memory.store import LocalStore
from ml.context_risk import ContextRiskService, effective_criticality, load_region, score
from ml.detector import FullFrameDetector
from ml.revisit_priority import compute_revisit
from ml.semantic_memory import SemanticMemoryEngine
from tests.conftest import add_crack, add_dust, make_panel

CHENNAI = (13.0806, 80.2773)   # inside the Chennai region box
TODAY = datetime(2026, 10, 11, tzinfo=timezone.utc)


def _raw(lat=CHENNAI[0], lon=CHENNAI[1], past_mm=3.0, ahead_mm=(0.0, 0.0, 115.6), gust=59.0, quakes=None, osm=None):
    past = [past_mm] * 30
    ahead = list(ahead_mm) + [0.0] * (7 - len(ahead_mm))
    return {
        "fetched_ts": TODAY.isoformat(),
        "latitude": lat,
        "longitude": lon,
        "weather": {
            "dates": [f"d{i}" for i in range(37)],
            "precipitation_mm": past + ahead,
            "wind_gust_kmh": [10.0] * 30 + [gust] * 7,
            "past_days": 30,
        },
        "earthquakes": {
            "events": quakes if quakes is not None else [{"magnitude": 4.75, "place": "x", "date": "1967-03-27", "distance_km": 276.0}],
            "radius_km": 300,
            "min_magnitude": 3.0,
            "since": "1900-01-01",
        },
        "osm": osm if osm is not None else {"enclosing": [{"amenity": "hospital", "name": "GH"}], "nearby_buildings": [{"building": "yes"}]},
    }


def test_factor_scores_follow_the_documented_formulas(config):
    r = score(_raw(), {"construction_year": 1996}, load_region(config), config, TODAY)
    f = {k: v["score"] for k, v in r["factors"].items()}
    assert f["seismic_zone"] == pytest.approx(0.16 / 0.36, abs=1e-4)     # IS 1893 Zone III
    assert f["seismic_history"] == pytest.approx((4.75 - 3.0) / 3.5, abs=1e-4)
    assert f["rain_forecast"] == pytest.approx(1.0)                      # 115.6 mm/day = IMD very heavy
    assert f["rain_antecedent"] == pytest.approx(90.0 / 300.0, abs=1e-4)
    assert f["wind_forecast"] == pytest.approx(59.0 / 118.0, abs=1e-4)
    assert f["occupancy"] == 1.0                                         # hospital
    assert f["building_age"] == pytest.approx(30 / 60)
    w = config["context_risk"]["weights"]
    expected = sum(w[k] * f[k] for k in f) / sum(w.values())
    assert r["score"] == pytest.approx(expected, abs=1e-3)
    assert r["coverage"] == 1.0
    assert r["criticality_multiplier"] == pytest.approx(1 + config["context_risk"]["criticality_gain"] * r["score"], abs=1e-3)
    assert r["reasons"] and r["reasons"][0].startswith("rain forecast")


def test_missing_sources_are_excluded_not_guessed(config):
    raw = _raw(osm={"error": "timeout"})
    raw["weather"] = {"error": "offline"}
    r = score(raw, {}, load_region(config), config, TODAY)
    unavailable = {k for k, v in r["factors"].items() if v["score"] is None}
    assert unavailable == {"rain_forecast", "rain_antecedent", "wind_forecast", "occupancy", "building_age"}
    w = config["context_risk"]["weights"]
    assert r["coverage"] == pytest.approx((w["seismic_zone"] + w["seismic_history"]) / sum(w.values()), abs=1e-3)
    f = r["factors"]
    expected = (w["seismic_zone"] * f["seismic_zone"]["score"] + w["seismic_history"] * f["seismic_history"]["score"]) / (
        w["seismic_zone"] + w["seismic_history"]
    )
    assert r["score"] == pytest.approx(expected, abs=1e-3)


def test_outside_region_has_no_seismic_zone(config):
    r = score(_raw(lat=28.61, lon=77.21), {}, load_region(config), config, TODAY)   # Delhi
    assert r["factors"]["seismic_zone"]["score"] is None


def test_admin_attributes_override_osm(config):
    r = score(_raw(), {"occupancy": "warehouse"}, load_region(config), config, TODAY)
    assert r["factors"]["occupancy"]["score"] == config["context_risk"]["occupancy_scores"]["warehouse"]
    assert r["factors"]["occupancy"]["source"] == "admin"


def test_effective_criticality_is_clipped():
    assert effective_criticality(0.6, None) == 0.6
    assert effective_criticality(0.6, {"criticality_multiplier": 1.25}) == pytest.approx(0.75)
    assert effective_criticality(0.9, {"criticality_multiplier": 1.5}) == 1.0


def test_service_caches_and_keeps_old_result_when_offline(config):
    store = LocalStore(config)
    store.ensure_site("gh", "GH", *CHENNAI)
    calls = []

    def fake_fetch(lat, lon, _cfg):
        calls.append((lat, lon))
        return _raw(lat, lon) if len(calls) == 1 else {"latitude": lat, "longitude": lon, "fetched_ts": TODAY.isoformat(),
                                                       "weather": {"error": "x"}, "earthquakes": {"error": "x"}, "osm": {"error": "x"}}

    svc = ContextRiskService(config, store, fetch=fake_fetch)
    first = svc.refresh("gh", force=True)
    assert first["coverage"] > 0.5 and svc.for_site("gh") == first
    assert svc.refresh("gh", force=True) == first          # every source failed → cached result kept
    assert len(calls) == 2
    updated = svc.set_attributes("gh", {"construction_year": 1950})
    assert updated["factors"]["building_age"]["score"] == 1.0
    # A fresh service (new process) reads the same result back from the Local DB.
    assert ContextRiskService(config, store, fetch=fake_fetch).for_site("gh")["score"] == updated["score"]


def test_same_damage_scores_higher_on_high_risk_site(config, embedder, classifier):
    """Identical visual damage → higher severity and revisit priority where the context is riskier."""
    store = LocalStore(config)
    store.ensure_site("calm", "Low-risk site", *CHENNAI)
    store.ensure_site("risky", "High-risk site", *CHENNAI)
    calm = _raw(past_mm=0.0, ahead_mm=(0.0,), gust=5.0, quakes=[], osm={"enclosing": [], "nearby_buildings": [{"building": "warehouse"}]})
    risky = _raw(past_mm=15.0, ahead_mm=(180.0,), gust=110.0)
    svc = ContextRiskService(config, store, fetch=lambda lat, lon, _c: risky)
    svc.refresh("risky", force=True)
    svc.fetch = lambda lat, lon, _c: calm
    svc.refresh("calm", force=True)
    svc.set_attributes("risky", {"construction_year": 1965})
    assert svc.for_site("risky")["score"] > svc.for_site("calm")["score"]

    engine = SemanticMemoryEngine(config, store, embedder, classifier, FullFrameDetector("solar_panel", 0.95))
    engine.context = svc
    damaged = add_dust(add_crack(make_panel(21)))
    events = {}
    for site in ("calm", "risky"):
        sid = store.create_session(site, "op1", "offline", {})["session_id"]
        events[site] = engine.observe(damaged, "solar_panel", sid)
    lo, hi = events["calm"], events["risky"]
    assert lo.change["change_magnitude"] == pytest.approx(hi.change["change_magnitude"])     # same visual evidence
    assert hi.severity_score > lo.severity_score > 0
    assert hi.context["score"] == svc.for_site("risky")["score"]
    stored = store.observations(hi.asset_id)[-1]["metadata"]["context"]
    assert stored["level"] == hi.context["level"]
    assert hi.revisit["context"] > lo.revisit["context"]


def test_revisit_context_term(config):
    base = compute_revisit([0.4], None, config)
    more = compute_revisit([0.4], None, config, context_score=1.0)
    assert more["revisit_priority"] == pytest.approx(min(1.0, base["revisit_priority"] + config["revisit_priority"]["w6"]))


def test_site_without_coordinates_is_located_automatically(config):
    store = LocalStore(config)
    store.ensure_site("dev", "Device-located")
    store.ensure_site("ip", "IP-located")
    seen = []
    svc = ContextRiskService(
        config, store,
        fetch=lambda lat, lon, _c: seen.append((lat, lon)) or _raw(lat, lon),
        locate=lambda _c: {"latitude": 12.9246, "longitude": 80.1, "source": "ip", "place": "Tambaram", "accuracy_m": None},
    )
    r = svc.refresh("dev", device_location={"latitude": 13.0806, "longitude": 80.2773, "accuracy_m": 35.0})
    assert store.get_site("dev")["latitude"] == 13.0806 and r["location_source"]["source"] == "device"
    r = svc.refresh("ip")
    assert (store.get_site("ip")["latitude"], store.get_site("ip")["longitude"]) == (12.9246, 80.1)
    assert r["location_source"]["place"] == "Tambaram"
    # A located site keeps its coordinates: a later device position does not move it.
    svc.refresh("dev", force=True, device_location={"latitude": 10.0, "longitude": 78.0})
    assert store.get_site("dev")["latitude"] == 13.0806
    assert seen == [(13.0806, 80.2773), (12.9246, 80.1), (13.0806, 80.2773)]


def test_no_location_available_gives_no_context(config):
    store = LocalStore(config)
    store.ensure_site("x", "X")
    config["context_risk"]["auto_locate_ip"] = False
    svc = ContextRiskService(config, store, fetch=lambda *a: pytest.fail("must not fetch"))
    assert svc.refresh("x") is None and store.get_site("x")["latitude"] is None
