"""Inspection session lifecycle.

start: FR-1 connectivity → FR-2 pull (online) / FR-3 offline → session row →
       FR-4 briefing (R recomputed from "now" for every site asset, D9).
end:   FR-20 recompute R for every site asset → FR-6 push / FR-7 queue →
       FR-18 report (includes FR-19 escalations raised during the session).
Interrupted sessions (crash, §6.2) can be resumed or closed.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from memory.store import LocalStore
from memory.sync import SyncManager
from ml.config import cfg
from ml.context_risk import ContextRiskService, haversine_km, locate_by_ip, place_name
from ml.escalation import EscalationService
from ml.reporting import build_report
from ml.schemas import new_id, utc_now
from ml.semantic_memory import SemanticMemoryEngine


class SessionStateError(Exception):
    """The requested session transition is not allowed in the session's current state."""


class MissionService:
    def __init__(
        self,
        store: LocalStore,
        engine: SemanticMemoryEngine,
        sync: SyncManager,
        escalation: EscalationService,
        context: Optional[ContextRiskService] = None,
    ):
        self.store = store
        self.engine = engine
        self.sync = sync
        self.escalation = escalation
        self.context = context

    def briefing(self, site_id: str) -> Dict[str, Any]:
        ranked = self.store.recompute_revisit(site_id, utc_now(), enqueue=False)
        unresolved = [d for d in self.store.list_defects(site_id=site_id) if d["status"] != "Resolved"]
        return {
            "site_id": site_id,
            "generated_ts": utc_now().isoformat(),
            "context": self.context.for_site(site_id) if self.context else None,
            "assets_due": [
                {
                    "asset_id": a["asset_id"],
                    "asset_type": a["asset_type"],
                    "current_condition": a["current_condition"],
                    "last_checked_ts": a["last_checked_ts"],
                    "revisit_priority": a["revisit_priority_score"],
                    "components": a["revisit_components"],
                    "location": a["location"],
                    "marker_id": a["marker_id"],
                    "simulated": a["simulated"],
                }
                for a in ranked
            ],
            "unresolved_defects": unresolved,
        }

    def start(
        self, site_id: str, operator_id: str, simulated: bool = False, location: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if self.store.get_site(site_id) is None:
            raise KeyError(f"Unknown site: {site_id}")
        sync = self.sync.start_sync(site_id)
        if sync.get("pulled") and any(sync["pulled"].values()):
            self.engine.reload_index()
        session = self.store.create_session(site_id, operator_id, sync["mode"], sync, simulated)
        if self.context:
            # Weather changes, and a site without coordinates takes this device's position. Never blocks (NFR-3).
            self.context.refresh_async(site_id, location)
        self.engine.start_prefetch(site_id)
        return {"session": session, "sync": sync, "briefing": self.briefing(site_id)}

    def quick_start(self, operator_id: str, location: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Start a session here without choosing a site.

        Memory is per site, so a site inspected before within sites.quick_start_radius_m of this
        position is reused (its assets can be recognised); otherwise a new site is created at this
        position and named after the place. Without a device position the IP position is used.
        """
        config = self.store.config
        source = "device"
        if location is None:
            try:
                location = locate_by_ip(config)
                source = "ip"
            except Exception as exc:  # noqa: BLE001
                raise LookupError(f"No location: allow location in the browser ({exc})") from exc
        lat, lon = float(location["latitude"]), float(location["longitude"])
        radius_km = float(cfg(config, "sites", "quick_start_radius_m")) / 1000.0
        nearest, best = None, None
        for s in self.store.list_sites():
            if s["latitude"] is None or s["longitude"] is None:
                continue
            d = haversine_km(lat, lon, s["latitude"], s["longitude"])
            if d <= radius_km and (best is None or d < best):
                nearest, best = s, d
        if nearest is not None:
            site = nearest
            created = False
        else:
            name = location.get("place") or place_name(lat, lon, config) or f"Inspection {datetime.now():%d %b %Y %H:%M}"
            site_id = new_id("site")
            site = self.store.ensure_site(site_id, name, lat, lon)
            created = True
            if self.context:
                self.context.record_location_source(site_id, {**location, "source": source})
        out = self.start(site["site_id"], operator_id, location={**location} if source == "device" else None)
        out["quick_start"] = {"site": site, "created": created, "distance_m": None if best is None else round(best * 1000), "location_source": source}
        return out

    def end(self, session_id: str) -> Dict[str, Any]:
        session = self.store.get_session(session_id)
        if session is None:
            raise KeyError(session_id)
        if session["status"] == "ended":
            raise SessionStateError(f"Session {session_id} already ended")
        self.engine.recompute_site(session["site_id"], utc_now())
        self.store.update_session(session_id, status="ended", ended_ts=utc_now().isoformat())
        self.engine.end_session_hook(session_id)
        sync = self.sync.end_sync()
        report = build_report(self.store, session_id, sync, self.escalation.for_session(session_id))
        self.store.update_session(session_id, end_sync=sync, report=report)
        return report

    def interrupted(self) -> list:
        return [s for s in self.store.list_sessions() if s["status"] in {"active", "interrupted"}]

    def resume(self, session_id: str) -> Dict[str, Any]:
        s = self.store.get_session(session_id)
        if s is None:
            raise KeyError(session_id)
        if s["status"] == "ended":
            raise SessionStateError("Cannot resume an ended session")
        self.engine.end_session_hook(session_id)
        return self.store.update_session(session_id, status="active")

    def report(self, session_id: str) -> Optional[Dict[str, Any]]:
        s = self.store.get_session(session_id)
        if s is None:
            raise KeyError(session_id)
        return s["report"] or build_report(self.store, session_id, s.get("end_sync") or {}, self.escalation.for_session(session_id))
