"""Context-aware structural risk of an inspection site (extension to SRS §8.1 / §8.2).

The camera only says how damaged an asset looks. The same crack matters more on a
crowded hospital in a seismic zone before a very heavy rain spell than on an empty
shed in dry weather. This module turns real, public data about the site's location
into a context score K ∈ [0, 1]:

  factor            source (all real, fetched live or from an official standard)
  seismic_zone      BIS IS 1893 (Part 1):2016 zone factor of the region (ml/regions/*.json)
  seismic_history   USGS earthquake catalogue, largest event within radius_km since `start`
  rain_forecast     Open-Meteo, largest daily rainfall in the next forecast_days
  rain_antecedent   Open-Meteo, rainfall total of the past past_days (saturated ground / seepage)
  wind_forecast     Open-Meteo, strongest wind gust in the next forecast_days
  occupancy         OpenStreetMap tags of the enclosing campus / nearby building, or admin entry
  building_age      admin-entered construction year, or the OpenStreetMap start_date tag

K = Σ wᵢ·fᵢ / Σ wᵢ over the factors that have data; missing factors are reported as
unavailable instead of being guessed. K is used in two places:
  criticality   C' = min(1, C·(1 + criticality_gain·K))   → severity S = (w1·M + w2·C')·Cf
  revisit       R' = R + w6·K                              (ml/revisit_priority.py)

Network calls happen only in refresh(), never on the per-frame path; the per-frame path
reads the cached result, so offline missions keep working with the last known context.
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

from ml.config import cfg, resolve_path
from ml.net import ensure_ca_bundle
from ml.schemas import utc_now, utc_now_iso

logger = logging.getLogger(__name__)

FACTORS = (
    "seismic_zone",
    "seismic_history",
    "rain_forecast",
    "rain_antecedent",
    "wind_forecast",
    "occupancy",
    "building_age",
)
META_KEY = "context_risk:{}"


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def load_region(config: Dict[str, Any]) -> Dict[str, Any]:
    with open(resolve_path(cfg(config, "context_risk", "region_file")), "r", encoding="utf-8") as f:
        return json.load(f)


def in_region(lat: float, lon: float, region: Dict[str, Any]) -> bool:
    b = region["bbox"]
    return b["min_lat"] <= lat <= b["max_lat"] and b["min_lon"] <= lon <= b["max_lon"]


# ── fetching (real public APIs) ───────────────────────────────────────────
def _http_json(url: str, config: Dict[str, Any], params: Optional[Dict] = None, data: Optional[Dict] = None) -> Any:
    ensure_ca_bundle()
    c = cfg(config, "context_risk")
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    body = urllib.parse.urlencode(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers={"User-Agent": c["user_agent"]})
    with urllib.request.urlopen(req, timeout=float(c["http_timeout_s"])) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_weather(lat: float, lon: float, config: Dict[str, Any]) -> Dict[str, Any]:
    c = cfg(config, "context_risk")
    past, ahead = int(c["rainfall"]["past_days"]), int(c["rainfall"]["forecast_days"])
    d = _http_json(
        c["endpoints"]["open_meteo"],
        config,
        params={
            "latitude": lat,
            "longitude": lon,
            "daily": "precipitation_sum,wind_gusts_10m_max",
            "past_days": past,
            "forecast_days": ahead,
            "timezone": c["timezone"],
        },
    )["daily"]
    # The response holds past_days entries before today, then forecast_days entries from today.
    return {
        "dates": d["time"],
        "precipitation_mm": d["precipitation_sum"],
        "wind_gust_kmh": d["wind_gusts_10m_max"],
        "past_days": past,
    }


def fetch_earthquakes(lat: float, lon: float, config: Dict[str, Any]) -> Dict[str, Any]:
    q = cfg(config, "context_risk", "earthquakes")
    d = _http_json(
        cfg(config, "context_risk", "endpoints", "usgs"),
        config,
        params={
            "format": "geojson",
            "latitude": lat,
            "longitude": lon,
            "maxradiuskm": q["radius_km"],
            "minmagnitude": q["min_magnitude"],
            "starttime": q["start"],
            "orderby": "magnitude",
            "limit": 20,
        },
    )
    events = []
    for f in d.get("features", []):
        p, (elon, elat, _depth) = f["properties"], f["geometry"]["coordinates"]
        when = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(milliseconds=p["time"])
        events.append(
            {
                "magnitude": p["mag"],
                "place": p["place"],
                "date": when.date().isoformat(),
                "distance_km": round(haversine_km(lat, lon, elat, elon), 1),
            }
        )
    return {"events": events, "radius_km": q["radius_km"], "min_magnitude": q["min_magnitude"], "since": q["start"]}


def fetch_osm(lat: float, lon: float, config: Dict[str, Any]) -> Dict[str, Any]:
    """Tags of the areas that contain the point (campus, building, land use) and of nearby buildings."""
    c = cfg(config, "context_risk")
    query = (
        f"[out:json][timeout:25];is_in({lat},{lon})->.a;"
        "(area.a[amenity];area.a[building];area.a[landuse];);out tags;"
        f"nwr(around:{c['osm_building_radius_m']},{lat},{lon})[building];out tags 10;"
    )
    last: Optional[Exception] = None
    for url in c["endpoints"]["overpass"]:  # public Overpass servers are often busy: try mirrors in order
        try:
            elements = _http_json(url, config, data={"data": query})["elements"]
            keep = ("amenity", "building", "landuse", "healthcare", "start_date", "building:levels", "name")
            pick = lambda e: {k: e["tags"][k] for k in keep if k in e.get("tags", {})}  # noqa: E731
            return {
                "enclosing": [pick(e) for e in elements if e["type"] == "area"],
                "nearby_buildings": [pick(e) for e in elements if e["type"] != "area"],
            }
        except Exception as exc:  # noqa: BLE001
            last = exc
    # Fallback: Nominatim reverse geocoding returns the OSM object at the point (one object, fewer tags).
    try:
        h = _http_json(
            c["endpoints"]["nominatim_reverse"],
            config,
            params={"lat": lat, "lon": lon, "format": "json", "zoom": 18, "extratags": 1},
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Overpass failed ({last}); Nominatim reverse failed ({exc})") from exc
    tags: Dict[str, Any] = {}
    if h.get("class") in ("amenity", "building", "landuse", "healthcare"):
        tags[h["class"]] = h["type"]
    extra = h.get("extratags") or {}
    tags.update({k: extra[k] for k in ("building", "start_date", "healthcare") if k in extra})
    if h.get("name"):
        tags["name"] = h["name"]
    return {"enclosing": [tags] if len(tags) > ("name" in tags) else [], "nearby_buildings": [], "via": "nominatim_reverse"}


def geocode(query: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Real place name → coordinates (OpenStreetMap Nominatim)."""
    hits = _http_json(
        cfg(config, "context_risk", "endpoints", "nominatim"),
        config,
        params={"q": query, "format": "json", "limit": 1, "countrycodes": "in"},
    )
    if not hits:
        raise LookupError(f"No OpenStreetMap match for {query!r}")
    h = hits[0]
    return {"latitude": float(h["lat"]), "longitude": float(h["lon"]), "display_name": h["display_name"]}


def locate_by_ip(config: Dict[str, Any]) -> Dict[str, Any]:
    """City-level position of this machine's public IP (fallback when the device gives no location)."""
    d = _http_json(cfg(config, "context_risk", "endpoints", "ip_geolocation"), config)
    if not d.get("success", True) or d.get("latitude") is None:
        raise LookupError(f"IP geolocation failed: {d.get('message', d)}")
    place = ", ".join(x for x in (d.get("city"), d.get("region"), d.get("country")) if x)
    return {"latitude": float(d["latitude"]), "longitude": float(d["longitude"]), "source": "ip", "place": place, "accuracy_m": None}


def place_name(lat: float, lon: float, config: Dict[str, Any]) -> Optional[str]:
    """Short human name of a position (OpenStreetMap reverse geocoding), e.g. "Tambaram, Chennai"."""
    try:
        h = _http_json(
            cfg(config, "context_risk", "endpoints", "nominatim_reverse"),
            config,
            params={"lat": lat, "lon": lon, "format": "json", "zoom": 16},
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Reverse geocoding failed: %s", exc)
        return None
    a = h.get("address") or {}
    local = h.get("name") or a.get("building") or a.get("amenity") or a.get("road")
    area = a.get("suburb") or a.get("neighbourhood") or a.get("town") or a.get("village")
    city = a.get("city") or a.get("state_district") or a.get("county")
    parts = [x for x in (local, area, city) if x]
    return ", ".join(dict.fromkeys(parts)) or None


def fetch_raw(lat: float, lon: float, config: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch every source; a failing source is recorded as an error, the others still count."""
    raw: Dict[str, Any] = {"fetched_ts": utc_now_iso(), "latitude": lat, "longitude": lon}
    for name, fn in (("weather", fetch_weather), ("earthquakes", fetch_earthquakes), ("osm", fetch_osm)):
        try:
            raw[name] = fn(lat, lon, config)
        except Exception as exc:  # noqa: BLE001 - one dead API must not drop the others
            logger.warning("Context source %s failed: %s", name, exc)
            raw[name] = {"error": str(exc)}
    return raw


# ── scoring (pure, recomputed whenever config or attributes change) ──────
def _occupancy(raw_osm: Dict[str, Any], attrs: Dict[str, Any], table: Dict[str, float]):
    if attrs.get("occupancy"):
        o = str(attrs["occupancy"]).lower()
        return table.get(o), f"{o} (entered by admin)", "admin"
    if not raw_osm or "error" in raw_osm:
        return None, "OpenStreetMap unavailable", "OpenStreetMap"
    candidates = []
    for tags in raw_osm.get("enclosing", []):  # a campus decides the occupancy of every block inside it
        candidates += [tags.get("amenity"), tags.get("healthcare"), tags.get("building")]
    for tags in raw_osm.get("nearby_buildings", []):
        candidates += [tags.get("amenity"), tags.get("healthcare"), tags.get("building")]
    for tags in raw_osm.get("enclosing", []):  # land use is the weakest hint
        candidates.append(tags.get("landuse"))
    for value in candidates:
        if value and value in table:
            names = [t["name"] for t in raw_osm.get("enclosing", []) + raw_osm.get("nearby_buildings", []) if t.get("name")]
            where = f" - {names[0]}" if names else ""
            return table[value], f"{value}{where}", "OpenStreetMap"
    return None, "no usable building/amenity tag in OpenStreetMap", "OpenStreetMap"


def _construction_year(raw_osm: Dict[str, Any], attrs: Dict[str, Any]):
    if attrs.get("construction_year"):
        return int(attrs["construction_year"]), "admin"
    for tags in (raw_osm or {}).get("enclosing", []) + (raw_osm or {}).get("nearby_buildings", []):
        m = re.match(r"\s*(\d{4})", str(tags.get("start_date", "")))
        if m:
            return int(m.group(1)), "OpenStreetMap start_date"
    return None, None


def score(
    raw: Dict[str, Any],
    attrs: Dict[str, Any],
    region: Dict[str, Any],
    config: Dict[str, Any],
    today: Optional[datetime] = None,
) -> Dict[str, Any]:
    c = cfg(config, "context_risk")
    lat, lon = raw["latitude"], raw["longitude"]
    today = today or utc_now()
    f: Dict[str, Dict[str, Any]] = {}

    def put(name, value, evidence, source):
        f[name] = {
            "score": None if value is None else round(_clip01(value), 4),
            "weight": float(c["weights"][name]),
            "evidence": evidence,
            "source": source,
        }

    s = region["seismic"]
    if in_region(lat, lon, region):
        put("seismic_zone", s["zone_factor"] / s["max_zone_factor"], f"Zone {s['zone']} (Z = {s['zone_factor']})", s["source"])
    else:
        put("seismic_zone", None, f"outside configured region '{region['name']}'", s["source"])

    eq = raw.get("earthquakes") or {}
    q = c["earthquakes"]
    if "error" in eq or not eq:
        put("seismic_history", None, f"USGS unavailable: {eq.get('error', 'not fetched')}", "USGS")
    elif not eq["events"]:
        put("seismic_history", 0.0, f"no M>={eq['min_magnitude']} event within {eq['radius_km']} km since {eq['since']}", "USGS")
    else:
        top = max(eq["events"], key=lambda e: e["magnitude"])
        put(
            "seismic_history",
            (top["magnitude"] - q["zero_magnitude"]) / (q["full_magnitude"] - q["zero_magnitude"]),
            f"largest: M{top['magnitude']} {top['place']} ({top['date']}, {top['distance_km']} km away); "
            f"{len(eq['events'])} event(s) M>={eq['min_magnitude']} within {eq['radius_km']} km",
            "USGS",
        )

    w = raw.get("weather") or {}
    if "error" in w or not w:
        for name in ("rain_forecast", "rain_antecedent", "wind_forecast"):
            put(name, None, f"Open-Meteo unavailable: {w.get('error', 'not fetched')}", "Open-Meteo")
    else:
        n = w["past_days"]
        past = [x or 0.0 for x in w["precipitation_mm"][:n]]
        ahead = [x or 0.0 for x in w["precipitation_mm"][n:]]
        gusts = [x or 0.0 for x in w["wind_gust_kmh"][n:]]
        rf, rw = c["rainfall"], c["wind"]
        peak = max(ahead, default=0.0)
        peak_day = w["dates"][n + ahead.index(peak)] if ahead else None
        put("rain_forecast", peak / rf["heavy_day_mm"], f"max {peak:.1f} mm/day forecast ({peak_day}); {sum(ahead):.1f} mm next {len(ahead)} days", "Open-Meteo")
        put("rain_antecedent", sum(past) / rf["antecedent_full_mm"], f"{sum(past):.1f} mm in the past {n} days", "Open-Meteo")
        put("wind_forecast", max(gusts, default=0.0) / rw["full_gust_kmh"], f"max gust {max(gusts, default=0.0):.0f} km/h forecast", "Open-Meteo")

    osm = raw.get("osm") or {}
    occ, occ_ev, occ_src = _occupancy(osm, attrs, c["occupancy_scores"])
    put("occupancy", occ, occ_ev, occ_src)

    year, year_src = _construction_year(osm, attrs)
    if year is None:
        put("building_age", None, "construction year unknown (not in OpenStreetMap; admin can enter it)", "-")
    else:
        age = max(0, today.year - year)
        put("building_age", age / float(c["building_age"]["full_age_years"]), f"built {year} ({age} years)", year_src)

    known = {k: v for k, v in f.items() if v["score"] is not None}
    total_w = sum(v["weight"] for v in f.values())
    known_w = sum(v["weight"] for v in known.values())
    k = sum(v["weight"] * v["score"] for v in known.values()) / known_w if known_w else 0.0
    levels = c["levels"]
    level = "high" if k >= levels["high"] else "moderate" if k >= levels["moderate"] else "low"
    drivers = sorted(known.items(), key=lambda kv: kv[1]["weight"] * kv[1]["score"], reverse=True)
    return {
        "score": round(k, 4),
        "level": level,
        "criticality_multiplier": round(1.0 + float(c["criticality_gain"]) * k, 4),
        "coverage": round(known_w / total_w, 4) if total_w else 0.0,
        "region": region["name"],
        "location": {"latitude": lat, "longitude": lon},
        "fetched_ts": raw.get("fetched_ts"),
        "scored_ts": today.isoformat(),
        "reasons": [f"{name.replace('_', ' ')}: {v['evidence']}" for name, v in drivers[:3] if v["score"] > 0],
        "factors": f,
    }


def effective_criticality(base: float, context: Optional[Dict[str, Any]]) -> float:
    if not context:
        return base
    return _clip01(base * float(context["criticality_multiplier"]))


def summary(context: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Compact form attached to live events and stored with every observation."""
    if not context:
        return None
    return {
        k: context[k]
        for k in ("score", "level", "criticality_multiplier", "coverage", "reasons", "fetched_ts")
    } | {"factors": {n: v["score"] for n, v in context["factors"].items()}}


# ── service: cache in the Local DB, refresh in the background ─────────────
class ContextRiskService:
    def __init__(
        self,
        config: Dict[str, Any],
        store,
        fetch: Callable[[float, float, Dict], Dict] = fetch_raw,
        locate: Callable[[Dict], Dict] = locate_by_ip,
    ):
        self.config = config
        self.store = store
        self.fetch = fetch
        self.locate = locate
        self.region = load_region(config)
        self._cache: Dict[str, Optional[Dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self._inflight: set = set()

    @property
    def enabled(self) -> bool:
        return bool(cfg(self.config, "context_risk", "enabled"))

    def _entry(self, site_id: str) -> Dict[str, Any]:
        return self.store.get_meta(META_KEY.format(site_id)) or {}

    def _save(self, site_id: str, entry: Dict[str, Any]) -> None:
        self.store.set_meta(META_KEY.format(site_id), entry)
        with self._lock:
            self._cache[site_id] = entry.get("result")

    def for_site(self, site_id: str) -> Optional[Dict[str, Any]]:
        """Cached context of a site (no network). None when disabled or never fetched."""
        if not self.enabled:
            return None
        with self._lock:
            if site_id in self._cache:
                return self._cache[site_id]
        result = self._entry(site_id).get("result")
        with self._lock:
            self._cache[site_id] = result
        return result

    def attributes(self, site_id: str) -> Dict[str, Any]:
        return self._entry(site_id).get("attributes", {})

    def set_attributes(self, site_id: str, attributes: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Admin-entered facts the public sources lack (construction year, occupancy); re-scores at once."""
        entry = self._entry(site_id)
        attrs = {**entry.get("attributes", {}), **{k: v for k, v in attributes.items() if v is not None}}
        entry["attributes"] = attrs
        if entry.get("raw"):
            entry["result"] = self._score(entry)
        self._save(site_id, entry)
        return entry.get("result")

    def _is_fresh(self, entry: Dict[str, Any], lat: float, lon: float) -> bool:
        raw = entry.get("raw")
        if not raw or (raw["latitude"], raw["longitude"]) != (lat, lon):
            return False
        age_h = (utc_now() - datetime.fromisoformat(raw["fetched_ts"])).total_seconds() / 3600.0
        return age_h < float(cfg(self.config, "context_risk", "cache_ttl_hours"))

    def record_location_source(self, site_id: str, loc: Dict[str, Any]) -> None:
        entry = self._entry(site_id)
        entry["location_source"] = {
            "source": loc.get("source", "device"), "accuracy_m": loc.get("accuracy_m"), "place": loc.get("place"),
            "latitude": loc["latitude"], "longitude": loc["longitude"], "set_ts": utc_now_iso(),
        }
        self._save(site_id, entry)

    def _ensure_location(self, site: Dict[str, Any], device: Optional[Dict[str, Any]], entry: Dict[str, Any]) -> bool:
        """A site without coordinates takes the inspecting device's position, else this machine's IP position."""
        if site["latitude"] is not None and site["longitude"] is not None:
            return True
        loc = None
        if device and device.get("latitude") is not None and device.get("longitude") is not None:
            loc = {"latitude": float(device["latitude"]), "longitude": float(device["longitude"]), "source": "device",
                   "accuracy_m": device.get("accuracy_m"), "place": None}
        elif cfg(self.config, "context_risk", "auto_locate_ip"):
            try:
                loc = self.locate(self.config)
            except Exception as exc:  # noqa: BLE001
                logger.warning("IP geolocation failed for %s: %s", site["site_id"], exc)
        if loc is None:
            return False
        self.store.set_site_location(site["site_id"], loc["latitude"], loc["longitude"])
        site["latitude"], site["longitude"] = loc["latitude"], loc["longitude"]
        entry["location_source"] = {**loc, "set_ts": utc_now_iso()}
        return True

    def refresh(
        self, site_id: str, force: bool = False, device_location: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Fetch real data for the site's coordinates (network). Keeps the old result if every source fails.

        A site without coordinates is located automatically first (device position, else IP).
        """
        site = self.store.get_site(site_id)
        if site is None:
            raise KeyError(site_id)
        entry = self._entry(site_id)
        if not self._ensure_location(site, device_location, entry):
            return None
        lat, lon = float(site["latitude"]), float(site["longitude"])
        if not force and self._is_fresh(entry, lat, lon):
            return entry.get("result")
        raw = self.fetch(lat, lon, self.config)
        if all("error" in (raw.get(k) or {"error": 1}) for k in ("weather", "earthquakes", "osm")) and entry.get("result"):
            logger.warning("Context refresh for %s failed entirely; keeping the cached result", site_id)
            return entry["result"]
        entry["raw"] = raw
        entry["result"] = self._score(entry)
        self._save(site_id, entry)
        return entry["result"]

    def _score(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        result = score(entry["raw"], entry.get("attributes", {}), self.region, self.config)
        result["location_source"] = entry.get("location_source") or {"source": "site"}
        return result

    def refresh_async(self, site_id: str, device_location: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled or not cfg(self.config, "context_risk", "auto_refresh"):
            return
        with self._lock:
            if site_id in self._inflight:
                return
            self._inflight.add(site_id)

        def run():
            try:
                self.refresh(site_id, device_location=device_location)
            except Exception:  # noqa: BLE001
                logger.exception("Context refresh failed for %s", site_id)
            finally:
                with self._lock:
                    self._inflight.discard(site_id)

        threading.Thread(target=run, name=f"context-{site_id}", daemon=True).start()
