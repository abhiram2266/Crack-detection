"""Local DB access layer (SQLite via SQLAlchemy).

Every per-asset update runs in one transaction that also enqueues idempotent
change records in `sync_outbox` (outbox pattern, FR-5/FR-7, SRS §6.2).
Records pulled from the cloud are applied without re-enqueueing.
"""

from __future__ import annotations

import base64
import logging
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session as OrmSession

from memory import db
from ml.asset_identity import AssetRef, GeoLocation
from ml.config import cfg, resolve_path
from ml.revisit_priority import compute_revisit
from ml.schemas import new_id, utc_now, utc_now_iso

logger = logging.getLogger(__name__)


def _row(obj) -> Dict[str, Any]:
    return {c.name: getattr(obj, c.name) for c in obj.__table__.columns}


def asset_dict(a: db.Asset) -> Dict[str, Any]:
    d = _row(a)
    d["location"] = (
        {"latitude": a.latitude, "longitude": a.longitude, "altitude": a.altitude}
        if a.latitude is not None
        else None
    )
    d["metadata"] = d.pop("meta") or {}
    return d


def observation_dict(o: db.Observation) -> Dict[str, Any]:
    d = _row(o)
    d["metadata"] = d.pop("meta") or {}
    return d


def prototype_payload(p: db.AssetPrototype, images_dir: Path) -> Dict[str, Any]:
    d = _row(p)
    d.pop("proto_id")
    d["embedding"] = base64.b64encode(p.embedding).decode("ascii")
    img = images_dir / p.image_reference
    d["image_b64"] = base64.b64encode(img.read_bytes()).decode("ascii") if img.exists() else None
    return d


class LocalStore:
    def __init__(self, config: Dict[str, Any], db_url: Optional[str] = None, images_dir: Optional[Path] = None):
        self.config = config
        scfg = cfg(config, "storage")
        url = db_url or scfg["db_url"]
        if url.startswith("sqlite:///") and not url.startswith("sqlite:////"):
            path = resolve_path(url[len("sqlite:///"):])
            path.parent.mkdir(parents=True, exist_ok=True)
            url = f"sqlite:///{path}"
        self.db_url = url
        self.engine = db.make_engine(url, int(scfg["busy_timeout_ms"]))
        db.Base.metadata.create_all(self.engine)
        self.Session = db.make_session_factory(self.engine)
        self.images_dir = images_dir or resolve_path(scfg["images_dir"])
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.RLock()
        self.integrity = self.integrity_check()

    # ── infrastructure ────────────────────────────────────────────────────
    @contextmanager
    def txn(self) -> Iterator[OrmSession]:
        with self._write_lock:
            s = self.Session()
            try:
                yield s
                s.commit()
            except Exception:
                s.rollback()
                raise
            finally:
                s.close()

    @contextmanager
    def read(self) -> Iterator[OrmSession]:
        s = self.Session()
        try:
            yield s
        finally:
            s.close()

    def integrity_check(self) -> Dict[str, Any]:
        with self.engine.connect() as conn:
            result: Dict[str, Any] = {}
            if self.db_url.startswith("sqlite"):
                result["integrity"] = conn.execute(text("PRAGMA integrity_check")).scalar()
                result["foreign_key_violations"] = len(conn.execute(text("PRAGMA foreign_key_check")).fetchall())
                result["journal_mode"] = conn.execute(text("PRAGMA journal_mode")).scalar()
            else:
                result["integrity"] = "ok"
                result["foreign_key_violations"] = 0
        with self.read() as s:
            result["orphan_observations"] = s.scalar(
                select(func.count()).select_from(db.Observation).where(
                    ~db.Observation.session_id.in_(select(db.Session.session_id))
                )
            )
            result["active_sessions"] = [
                r.session_id for r in s.scalars(select(db.Session).where(db.Session.status == "active"))
            ]
        result["ok"] = result["integrity"] == "ok" and result["foreign_key_violations"] == 0 and result[
            "orphan_observations"
        ] == 0
        if not result["ok"]:
            logger.error("Local DB integrity check failed: %s", result)
        return result

    def _enqueue(self, s: OrmSession, entity: str, entity_id: str, site_id: str, payload: Dict[str, Any]) -> None:
        s.add(
            db.SyncOutbox(
                change_id=new_id("chg"),
                entity=entity,
                entity_id=entity_id,
                site_id=site_id,
                payload=payload,
                created_ts=utc_now_iso(),
            )
        )

    def get_meta(self, key: str) -> Any:
        with self.read() as s:
            m = s.get(db.Meta, key)
            return m.value if m else None

    def set_meta(self, key: str, value: Any) -> None:
        with self.txn() as s:
            s.merge(db.Meta(key=key, value=value))

    def record_config_version(self, username: str, changes: Dict[str, Any], snapshot: Dict[str, Any]) -> Dict:
        with self.txn() as s:
            v = db.ConfigVersion(created_ts=utc_now_iso(), username=username, changes=changes, snapshot=snapshot)
            s.add(v)
            s.flush()
            return _row(v)

    def list_config_versions(self, include_snapshot: bool = False) -> List[Dict]:
        with self.read() as s:
            out = []
            for v in s.scalars(select(db.ConfigVersion).order_by(db.ConfigVersion.version.desc())):
                d = _row(v)
                if not include_snapshot:
                    d.pop("snapshot")
                out.append(d)
            return out

    # ── sites ─────────────────────────────────────────────────────────────
    def ensure_site(self, site_id: str, name: str, latitude=None, longitude=None, enqueue: bool = True) -> Dict:
        with self.txn() as s:
            site = s.get(db.Site, site_id)
            if site is None:
                site = db.Site(site_id=site_id, name=name, latitude=latitude, longitude=longitude, created_ts=utc_now_iso())
                s.add(site)
                if enqueue:
                    self._enqueue(s, "site", site_id, site_id, _row(site))
            return _row(site)

    def set_site_location(self, site_id: str, latitude: float, longitude: float, enqueue: bool = True) -> Dict:
        with self.txn() as s:
            site = s.get(db.Site, site_id)
            if site is None:
                raise KeyError(site_id)
            site.latitude, site.longitude = float(latitude), float(longitude)
            if enqueue:
                self._enqueue(s, "site", site_id, site_id, _row(site))
            return _row(site)

    def list_sites(self) -> List[Dict]:
        with self.read() as s:
            return [_row(x) for x in s.scalars(select(db.Site).order_by(db.Site.site_id))]

    def get_site(self, site_id: str) -> Optional[Dict]:
        with self.read() as s:
            x = s.get(db.Site, site_id)
            return _row(x) if x else None

    # ── users ─────────────────────────────────────────────────────────────
    def create_user(self, username: str, password_hash: str, role: str, email: Optional[str] = None) -> Dict:
        if role not in db.ROLES:
            raise ValueError(f"Unknown role: {role}")
        with self.txn() as s:
            u = db.User(username=username, password_hash=password_hash, role=role, email=email, active=True, created_ts=utc_now_iso())
            s.add(u)
            s.flush()
            return _row(u)

    def get_user(self, username: str) -> Optional[Dict]:
        with self.read() as s:
            u = s.scalar(select(db.User).where(db.User.username == username))
            return _row(u) if u else None

    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        with self.read() as s:
            u = s.get(db.User, user_id)
            return _row(u) if u else None

    def list_users(self) -> List[Dict]:
        with self.read() as s:
            users = []
            for u in s.scalars(select(db.User).order_by(db.User.user_id)):
                d = _row(u)
                d.pop("password_hash")
                d["sites"] = [
                    r.site_id for r in s.scalars(select(db.SiteSubscription).where(db.SiteSubscription.user_id == u.user_id))
                ]
                users.append(d)
            return users

    def update_user(self, user_id: int, **fields) -> Dict:
        allowed = {"role", "email", "active", "password_hash"}
        with self.txn() as s:
            u = s.get(db.User, user_id)
            if u is None:
                raise KeyError(user_id)
            for k, v in fields.items():
                if k not in allowed:
                    raise ValueError(f"Field not updatable: {k}")
                if k == "role" and v not in db.ROLES:
                    raise ValueError(f"Unknown role: {v}")
                setattr(u, k, v)
            d = _row(u)
            d.pop("password_hash")
            return d

    def set_subscriptions(self, user_id: int, site_ids: Sequence[str]) -> None:
        with self.txn() as s:
            for sub in s.scalars(select(db.SiteSubscription).where(db.SiteSubscription.user_id == user_id)):
                s.delete(sub)
            for sid in site_ids:
                if s.get(db.Site, sid) is None:
                    raise KeyError(f"Unknown site: {sid}")
                s.add(db.SiteSubscription(user_id=user_id, site_id=sid))

    def escalation_recipients(self, site_id: str) -> List[Dict]:
        """Active maintenance users subscribed to the site (FR-19)."""
        with self.read() as s:
            rows = s.scalars(
                select(db.User)
                .join(db.SiteSubscription, db.SiteSubscription.user_id == db.User.user_id)
                .where(db.SiteSubscription.site_id == site_id, db.User.role == "maintenance", db.User.active.is_(True))
            )
            out = []
            for u in rows:
                d = _row(u)
                d.pop("password_hash")
                out.append(d)
            return out

    # ── sessions ──────────────────────────────────────────────────────────
    def create_session(
        self, site_id: str, operator_id: str, mode: str, start_sync: Dict, simulated: bool = False
    ) -> Dict:
        now = utc_now()
        with self.txn() as s:
            if s.get(db.Site, site_id) is None:
                raise KeyError(f"Unknown site: {site_id}")
            sess = db.Session(
                session_id=new_id("sess"),
                site_id=site_id,
                operator_id=operator_id,
                mission_date=now.date().isoformat(),
                started_ts=now.isoformat(),
                status="active",
                mode=mode,
                assets_processed=0,
                simulated=simulated,
                start_sync=start_sync,
            )
            s.add(sess)
            s.flush()
            self._enqueue(s, "session", sess.session_id, site_id, _row(sess))
            return _row(sess)

    def get_session(self, session_id: str) -> Optional[Dict]:
        with self.read() as s:
            x = s.get(db.Session, session_id)
            return _row(x) if x else None

    def list_sessions(self, site_id: Optional[str] = None) -> List[Dict]:
        with self.read() as s:
            q = select(db.Session).order_by(db.Session.started_ts.desc())
            if site_id:
                q = q.where(db.Session.site_id == site_id)
            return [_row(x) for x in s.scalars(q)]

    def count_sessions(self) -> int:
        with self.read() as s:
            return int(s.scalar(select(func.count()).select_from(db.Session)))

    def update_session(self, session_id: str, **fields) -> Dict:
        with self.txn() as s:
            sess = s.get(db.Session, session_id)
            if sess is None:
                raise KeyError(session_id)
            for k, v in fields.items():
                setattr(sess, k, v)
            s.flush()
            self._enqueue(s, "session", session_id, sess.site_id, _row(sess))
            return _row(sess)

    def session_assets(self, session_id: str) -> List[str]:
        with self.read() as s:
            return list(
                s.scalars(select(db.Observation.asset_id).where(db.Observation.session_id == session_id).distinct())
            )

    def last_session_observation(self, session_id: str, asset_id: str) -> Optional[Dict]:
        with self.read() as s:
            o = s.scalar(
                select(db.Observation)
                .where(db.Observation.session_id == session_id, db.Observation.asset_id == asset_id)
                .order_by(db.Observation.observation_ts.desc())
                .limit(1)
            )
            return observation_dict(o) if o else None

    def session_observations(self, session_id: str) -> List[Dict]:
        with self.read() as s:
            q = select(db.Observation).where(db.Observation.session_id == session_id).order_by(db.Observation.observation_ts)
            return [observation_dict(o) for o in s.scalars(q)]

    # ── assets ────────────────────────────────────────────────────────────
    def get_asset(self, asset_id: str) -> Optional[Dict]:
        with self.read() as s:
            a = s.get(db.Asset, asset_id)
            return asset_dict(a) if a else None

    def list_assets(self, site_id: Optional[str] = None) -> List[Dict]:
        with self.read() as s:
            q = select(db.Asset).order_by(db.Asset.revisit_priority_score.desc())
            if site_id:
                q = q.where(db.Asset.site_id == site_id)
            return [asset_dict(a) for a in s.scalars(q)]

    def asset_refs(self, site_id: str) -> List[AssetRef]:
        with self.read() as s:
            return [
                AssetRef(
                    a.asset_id,
                    a.asset_type,
                    GeoLocation(a.latitude, a.longitude, a.altitude or 0.0) if a.latitude is not None else None,
                    a.marker_id,
                )
                for a in s.scalars(select(db.Asset).where(db.Asset.site_id == site_id))
            ]

    def marker_table(self, site_id: str) -> Dict[int, str]:
        with self.read() as s:
            rows = s.execute(
                select(db.Asset.marker_id, db.Asset.asset_id).where(
                    db.Asset.site_id == site_id, db.Asset.marker_id.is_not(None)
                )
            )
            return {int(m): aid for m, aid in rows}

    def observations(self, asset_id: str) -> List[Dict]:
        with self.read() as s:
            q = select(db.Observation).where(db.Observation.asset_id == asset_id).order_by(db.Observation.observation_ts)
            return [observation_dict(o) for o in s.scalars(q)]

    def latest_observation(self, asset_id: str) -> Optional[Dict]:
        with self.read() as s:
            o = s.scalar(
                select(db.Observation)
                .where(db.Observation.asset_id == asset_id)
                .order_by(db.Observation.observation_ts.desc())
                .limit(1)
            )
            return observation_dict(o) if o else None

    def _mission_severities(self, s: OrmSession, asset_id: str) -> List[float]:
        rows = s.execute(
            select(db.Observation.session_id, func.max(db.Observation.severity_score), func.min(db.Observation.observation_ts))
            .where(db.Observation.asset_id == asset_id)
            .group_by(db.Observation.session_id)
            .order_by(func.min(db.Observation.observation_ts))
        ).all()
        return [float(r[1]) for r in rows]

    def mission_severities(self, asset_id: str) -> List[float]:
        with self.read() as s:
            return self._mission_severities(s, asset_id)

    def _apply_revisit(self, s: OrmSession, asset: db.Asset, now: Optional[datetime]) -> Dict[str, float]:
        enabled = (self.config.get("context_risk") or {}).get("enabled", False)
        ctx = s.get(db.Meta, f"context_risk:{asset.site_id}") if enabled else None
        k = ((ctx.value or {}).get("result") or {}).get("score", 0.0) if ctx else 0.0
        comp = compute_revisit(self._mission_severities(s, asset.asset_id), asset.last_checked_ts, self.config, now, k)
        asset.revisit_priority_score = comp["revisit_priority"]
        asset.revisit_components = comp
        return comp

    def recompute_revisit(self, site_id: str, now: Optional[datetime] = None, enqueue: bool = True) -> List[Dict]:
        """FR-20 / D9: recompute R for every asset at the site, seen or not."""
        now = now or utc_now()
        with self.txn() as s:
            out = []
            for a in s.scalars(select(db.Asset).where(db.Asset.site_id == site_id)):
                self._apply_revisit(s, a, now)
                if enqueue:
                    a.updated_ts = now.isoformat()
                    self._enqueue(s, "asset", a.asset_id, site_id, _row(a))
                out.append(asset_dict(a))
            return sorted(out, key=lambda d: d["revisit_priority_score"], reverse=True)

    # ── prototypes ────────────────────────────────────────────────────────
    def all_prototypes(self) -> List[Dict]:
        with self.read() as s:
            rows = s.execute(
                select(db.AssetPrototype, db.Asset.site_id, db.Asset.asset_type).join(
                    db.Asset, db.Asset.asset_id == db.AssetPrototype.asset_id
                )
            ).all()
            return [
                {
                    "proto_id": p.proto_id,
                    "asset_id": p.asset_id,
                    "site_id": site_id,
                    "asset_type": atype,
                    "embedding": np.frombuffer(p.embedding, dtype=np.float32).copy(),
                    "image_reference": p.image_reference,
                }
                for p, site_id, atype in rows
            ]

    def prototype_image_path(self, proto_id: int) -> Optional[Path]:
        with self.read() as s:
            p = s.get(db.AssetPrototype, proto_id)
            return (self.images_dir / p.image_reference) if p else None

    def image_path(self, ref: str) -> Path:
        return self.images_dir / ref

    def asset_image_dir(self, site_id: str, asset_id: str) -> Path:
        d = self.images_dir / site_id / asset_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ── observation commit (one transaction per processed asset) ─────────
    def commit_observation(
        self,
        *,
        asset: Dict[str, Any],
        observation: Dict[str, Any],
        is_new: bool,
        confirmed: bool,
        reference_condition: Optional[Dict[str, Any]],
        prototype: Optional[Dict[str, Any]],
        evict_proto_id: Optional[int],
        resolution_missions: int,
    ) -> Dict[str, Any]:
        """Persist one processed asset atomically and return the stored state.

        `asset` holds the asset fields for a new asset, or {"asset_id": …} for
        an existing one. Defects are opened/extended/streak-counted here so
        the defect log can never disagree with the observation log.
        """
        with self.txn() as s:
            ts = observation["observation_ts"]
            if is_new:
                a = db.Asset(**asset)
                s.add(a)
                s.flush()
            else:
                a = s.get(db.Asset, asset["asset_id"])
                if a is None:
                    raise KeyError(asset["asset_id"])
            site_id = a.site_id

            obs = db.Observation(**observation)
            s.add(obs)

            a.last_checked_ts = ts
            a.current_condition = obs.condition_description
            a.current_is_defect = obs.is_defect
            a.updated_ts = ts
            if reference_condition is not None:
                a.reference_condition = reference_condition
            if (confirmed or is_new) and obs.latitude is not None:
                # D3: asset location is the running mean of confirmed observations only.
                n = a.location_samples or 0
                if n == 0 or a.latitude is None:
                    a.latitude, a.longitude, a.altitude = obs.latitude, obs.longitude, obs.altitude
                    a.location_samples = 1
                else:
                    a.latitude = (a.latitude * n + obs.latitude) / (n + 1)
                    a.longitude = (a.longitude * n + obs.longitude) / (n + 1)
                    a.altitude = ((a.altitude or 0.0) * n + (obs.altitude or 0.0)) / (n + 1)
                    a.location_samples = n + 1

            evicted = None
            new_proto_id = None
            if evict_proto_id is not None:
                victim = s.get(db.AssetPrototype, evict_proto_id)
                if victim is not None:
                    evicted = evict_proto_id
                    s.delete(victim)
            if prototype is not None:
                p = db.AssetPrototype(
                    proto_uid=new_id("proto"),
                    asset_id=a.asset_id,
                    observation_id=obs.observation_id,
                    embedding=np.asarray(prototype["embedding"], dtype=np.float32).tobytes(),
                    dim=int(len(prototype["embedding"])),
                    image_reference=prototype["image_reference"],
                    created_ts=ts,
                )
                s.add(p)
                s.flush()
                new_proto_id = p.proto_id
                if a.reference_proto_id is None or is_new:
                    a.reference_proto_id = p.proto_id
                self._enqueue(s, "prototype", p.proto_uid, site_id, prototype_payload(p, self.images_dir))
            if evicted is not None and a.reference_proto_id == evicted:
                a.reference_proto_id = new_proto_id or s.scalar(
                    select(db.AssetPrototype.proto_id).where(db.AssetPrototype.asset_id == a.asset_id).limit(1)
                )

            defect_event = self._apply_defects(s, a, obs, resolution_missions)
            s.flush()
            revisit = self._apply_revisit(s, a, None)

            self._enqueue(s, "asset", a.asset_id, site_id, _row(a))
            self._enqueue(s, "observation", obs.observation_id, site_id, _row(obs))
            if defect_event:
                for d in defect_event["defects"]:
                    self._enqueue(s, "defect", d["defect_id"], site_id, d)
                for t in defect_event["trend"]:
                    self._enqueue(s, "defect_trend", t["trend_id"], site_id, t)

            assets_processed = s.scalar(
                select(func.count(func.distinct(db.Observation.asset_id))).where(
                    db.Observation.session_id == obs.session_id
                )
            )
            sess = s.get(db.Session, obs.session_id)
            sess.assets_processed = int(assets_processed)

            return {
                "asset": asset_dict(a),
                "observation": observation_dict(obs),
                "revisit": revisit,
                "defect": defect_event,
                "proto_id": new_proto_id,
                "evicted_proto_id": evicted,
            }

    def _apply_defects(
        self, s: OrmSession, a: db.Asset, obs: db.Observation, resolution_missions: int
    ) -> Optional[Dict[str, Any]]:
        open_defects = list(
            s.scalars(select(db.Defect).where(db.Defect.asset_id == a.asset_id, db.Defect.status != "Resolved"))
        )
        touched: List[db.Defect] = []
        trend: List[db.DefectTrend] = []
        opened = None
        if obs.is_defect and obs.defect_type:
            d = next((x for x in open_defects if x.defect_type == obs.defect_type), None)
            if d is None:
                d = db.Defect(
                    defect_id=new_id("def"),
                    asset_id=a.asset_id,
                    defect_type=obs.defect_type,
                    severity_score=obs.severity_score,
                    max_severity=obs.severity_score,
                    first_detected_ts=obs.observation_ts,
                    last_seen_ts=obs.observation_ts,
                    status="Open",
                    normal_streak=0,
                    resolution_suggested=False,
                    updated_ts=obs.observation_ts,
                )
                s.add(d)
                opened = d
            else:
                d.severity_score = obs.severity_score
                d.max_severity = max(d.max_severity, obs.severity_score)
                d.last_seen_ts = obs.observation_ts
                d.normal_streak = 0
                d.resolution_suggested = False
                d.updated_ts = obs.observation_ts
            t = db.DefectTrend(
                trend_id=new_id("trd"),
                defect_id=d.defect_id,
                observation_id=obs.observation_id,
                session_id=obs.session_id,
                ts=obs.observation_ts,
                severity_score=obs.severity_score,
            )
            s.add(t)
            trend.append(t)
            touched.append(d)
        elif not obs.is_defect:
            for d in open_defects:
                if d.last_normal_session_id != obs.session_id:
                    d.normal_streak += 1
                    d.last_normal_session_id = obs.session_id
                    d.resolution_suggested = d.normal_streak >= resolution_missions
                    d.updated_ts = obs.observation_ts
                    touched.append(d)
        if not touched:
            return None
        s.flush()
        return {
            "opened": _row(opened) if opened else None,
            "defects": [_row(d) for d in touched],
            "trend": [_row(t) for t in trend],
        }

    # ── defects ───────────────────────────────────────────────────────────
    def list_defects(
        self, site_id: Optional[str] = None, status: Optional[str] = None, asset_id: Optional[str] = None
    ) -> List[Dict]:
        with self.read() as s:
            q = select(db.Defect, db.Asset.site_id, db.Asset.asset_type).join(db.Asset, db.Asset.asset_id == db.Defect.asset_id)
            if site_id:
                q = q.where(db.Asset.site_id == site_id)
            if status:
                q = q.where(db.Defect.status == status)
            if asset_id:
                q = q.where(db.Defect.asset_id == asset_id)
            q = q.order_by(db.Defect.severity_score.desc())
            out = []
            for d, sid, atype in s.execute(q).all():
                row = _row(d)
                row["site_id"] = sid
                row["asset_type"] = atype
                out.append(row)
            return out

    def get_defect(self, defect_id: str) -> Optional[Dict]:
        with self.read() as s:
            d = s.get(db.Defect, defect_id)
            if d is None:
                return None
            row = _row(d)
            row["trend"] = [
                _row(t)
                for t in s.scalars(select(db.DefectTrend).where(db.DefectTrend.defect_id == defect_id).order_by(db.DefectTrend.ts))
            ]
            return row

    def update_defect_status(self, defect_id: str, status: str, notes: Optional[str] = None) -> Dict:
        if status not in db.DEFECT_STATUSES:
            raise ValueError(f"Invalid status: {status}")
        with self.txn() as s:
            d = s.get(db.Defect, defect_id)
            if d is None:
                raise KeyError(defect_id)
            now = utc_now_iso()
            d.status = status
            d.updated_ts = now
            if notes is not None:
                d.notes = notes
            d.resolved_ts = now if status == "Resolved" else None
            if status == "Resolved":
                d.resolution_suggested = False
            self._enqueue(s, "defect", d.defect_id, s.get(db.Asset, d.asset_id).site_id, _row(d))
            return _row(d)

    # ── notifications ─────────────────────────────────────────────────────
    def add_notification(self, **fields) -> Dict:
        with self.txn() as s:
            n = db.Notification(notification_id=new_id("ntf"), created_ts=utc_now_iso(), read=False, **fields)
            s.add(n)
            s.flush()
            return _row(n)

    def set_notification_email_status(self, notification_id: str, status: str) -> None:
        with self.txn() as s:
            n = s.get(db.Notification, notification_id)
            if n is not None:
                n.email_status = status

    def list_notifications(self, user_id: int, unread_only: bool = False) -> List[Dict]:
        with self.read() as s:
            q = select(db.Notification).where(db.Notification.user_id == user_id)
            if unread_only:
                q = q.where(db.Notification.read.is_(False))
            return [_row(n) for n in s.scalars(q.order_by(db.Notification.created_ts.desc()))]

    def session_notifications(self, session_id: str) -> List[Dict]:
        with self.read() as s:
            rows = s.execute(
                select(db.Notification, db.User.username)
                .join(db.User, db.User.user_id == db.Notification.user_id)
                .where(db.Notification.session_id == session_id)
                .order_by(db.Notification.created_ts)
            ).all()
            return [{**_row(n), "username": u} for n, u in rows]

    def all_notifications(self) -> List[Dict]:
        with self.read() as s:
            return [_row(n) for n in s.scalars(select(db.Notification))]

    def mark_notification_read(self, user_id: int, notification_id: str) -> Dict:
        with self.txn() as s:
            n = s.get(db.Notification, notification_id)
            if n is None or n.user_id != user_id:
                raise KeyError(notification_id)
            n.read = True
            return _row(n)

    # ── outbox ────────────────────────────────────────────────────────────
    def pending_outbox(self, limit: int) -> List[Dict]:
        with self.read() as s:
            q = select(db.SyncOutbox).where(db.SyncOutbox.sent_ts.is_(None)).order_by(db.SyncOutbox.seq).limit(limit)
            return [_row(r) for r in s.scalars(q)]

    def outbox_counts(self) -> Dict[str, int]:
        with self.read() as s:
            pending = s.scalar(select(func.count()).select_from(db.SyncOutbox).where(db.SyncOutbox.sent_ts.is_(None)))
            sent = s.scalar(select(func.count()).select_from(db.SyncOutbox).where(db.SyncOutbox.sent_ts.is_not(None)))
            return {"pending": int(pending), "sent": int(sent)}

    def mark_outbox_sent(self, change_ids: Sequence[str]) -> None:
        if not change_ids:
            return
        with self.txn() as s:
            now = utc_now_iso()
            for r in s.scalars(select(db.SyncOutbox).where(db.SyncOutbox.change_id.in_(list(change_ids)))):
                r.sent_ts = now

    def bump_outbox_attempts(self, change_ids: Sequence[str]) -> None:
        with self.txn() as s:
            for r in s.scalars(select(db.SyncOutbox).where(db.SyncOutbox.change_id.in_(list(change_ids)))):
                r.attempts += 1

    # ── pulled cloud records (FR-2) ───────────────────────────────────────
    def apply_remote(self, records: Dict[str, List[Dict]]) -> Dict[str, int]:
        """Merge a cloud snapshot: LWW for assets/defects, append-only for the rest."""
        counts = {k: 0 for k in records}
        with self.txn() as s:
            for site in records.get("site", []):
                if s.get(db.Site, site["site_id"]) is None:
                    s.add(db.Site(**site))
                    counts["site"] += 1
            for sess in records.get("session", []):
                if s.get(db.Session, sess["session_id"]) is None:
                    s.add(db.Session(**sess))
                    counts["session"] += 1
            s.flush()
            for a in records.get("asset", []):
                cur = s.get(db.Asset, a["asset_id"])
                if cur is None:
                    s.add(db.Asset(**a))
                    counts["asset"] += 1
                elif a["updated_ts"] > cur.updated_ts:
                    for k, v in a.items():
                        setattr(cur, k, v)
                    counts["asset"] += 1
            s.flush()
            for o in records.get("observation", []):
                if s.get(db.Observation, o["observation_id"]) is None:
                    s.add(db.Observation(**o))
                    counts["observation"] += 1
            for d in records.get("defect", []):
                cur = s.get(db.Defect, d["defect_id"])
                if cur is None:
                    s.add(db.Defect(**d))
                    counts["defect"] += 1
                elif d["updated_ts"] > cur.updated_ts:
                    for k, v in d.items():
                        setattr(cur, k, v)
                    counts["defect"] += 1
            s.flush()
            for t in records.get("defect_trend", []):
                if s.get(db.DefectTrend, t["trend_id"]) is None:
                    s.add(db.DefectTrend(**t))
                    counts["defect_trend"] += 1
            for p in records.get("prototype", []):
                exists = s.scalar(select(db.AssetPrototype).where(db.AssetPrototype.proto_uid == p["proto_uid"]))
                if exists is not None:
                    continue
                if p.get("image_b64"):
                    path = self.images_dir / p["image_reference"]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(base64.b64decode(p["image_b64"]))
                s.add(
                    db.AssetPrototype(
                        proto_uid=p["proto_uid"],
                        asset_id=p["asset_id"],
                        observation_id=p.get("observation_id"),
                        embedding=base64.b64decode(p["embedding"]),
                        dim=p["dim"],
                        image_reference=p["image_reference"],
                        created_ts=p["created_ts"],
                    )
                )
                counts["prototype"] += 1
        return counts

    # ── aggregates ────────────────────────────────────────────────────────
    def summary(self, site_id: Optional[str] = None) -> Dict[str, Any]:
        threshold = float(cfg(self.config, "severity", "priority_threshold"))
        with self.read() as s:
            aq = select(func.count()).select_from(db.Asset)
            dq = (
                select(func.count())
                .select_from(db.Defect)
                .join(db.Asset, db.Asset.asset_id == db.Defect.asset_id)
                .where(db.Defect.status != "Resolved")
            )
            sq = select(func.count()).select_from(db.Session)
            if site_id:
                aq = aq.where(db.Asset.site_id == site_id)
                dq = dq.where(db.Asset.site_id == site_id)
                sq = sq.where(db.Session.site_id == site_id)
            latest = (
                select(db.Observation.asset_id, func.max(db.Observation.observation_ts).label("ts"))
                .group_by(db.Observation.asset_id)
                .subquery()
            )
            pq = (
                select(func.count())
                .select_from(db.Observation)
                .join(latest, (latest.c.asset_id == db.Observation.asset_id) & (latest.c.ts == db.Observation.observation_ts))
                .join(db.Asset, db.Asset.asset_id == db.Observation.asset_id)
                .where(db.Observation.severity_score > threshold)
            )
            if site_id:
                pq = pq.where(db.Asset.site_id == site_id)
            return {
                "total_assets": int(s.scalar(aq)),
                "active_defects": int(s.scalar(dq)),
                "priority_assets": int(s.scalar(pq)),
                "missions": int(s.scalar(sq)),
            }
