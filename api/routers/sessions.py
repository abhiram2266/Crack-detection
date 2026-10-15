"""Inspection sessions: start (FR-1–4), end (FR-6/7, FR-18–20), reports, crash recovery."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from ml.mission import SessionStateError
from ml.reporting import render_html, render_pdf

from api.deps import any_user, field_roles, get_services, operator_only
from api.services import Services

router = APIRouter(tags=["sessions"])


class StartRequest(BaseModel):
    site_id: str
    simulated: bool = False
    latitude: Optional[float] = None     # device position (browser geolocation), used to locate a site
    longitude: Optional[float] = None    # that has no coordinates yet
    accuracy_m: Optional[float] = None


@router.get("/api/sites")
def list_sites(_: Dict = Depends(any_user), services: Services = Depends(get_services)) -> List[Dict[str, Any]]:
    return services.store.list_sites()


@router.get("/api/sites/{site_id}/briefing")
def briefing(site_id: str, _: Dict = Depends(field_roles), services: Services = Depends(get_services)) -> Dict[str, Any]:
    if services.store.get_site(site_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown site {site_id}")
    return services.mission.briefing(site_id)


@router.get("/api/sites/{site_id}/context")
def site_context(site_id: str, _: Dict = Depends(any_user), services: Services = Depends(get_services)) -> Dict[str, Any]:
    """Cached context risk of a site (real weather, seismic, building data) and admin-entered attributes."""
    site = services.store.get_site(site_id)
    if site is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown site {site_id}")
    return {"site": site, "context": services.context.for_site(site_id), "attributes": services.context.attributes(site_id)}


@router.post("/api/sessions/start")
def start_session(body: StartRequest, user: Dict = Depends(operator_only), services: Services = Depends(get_services)) -> Dict[str, Any]:
    try:
        location = (
            {"latitude": body.latitude, "longitude": body.longitude, "accuracy_m": body.accuracy_m}
            if body.latitude is not None and body.longitude is not None
            else None
        )
        return services.mission.start(body.site_id, user["username"], simulated=body.simulated, location=location)
    except KeyError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc


@router.post("/api/sessions/{session_id}/end")
def end_session(session_id: str, _: Dict = Depends(operator_only), services: Services = Depends(get_services)) -> Dict[str, Any]:
    if services.live is not None and services.live.session_id == session_id:
        services.live.stop()
    try:
        return services.mission.end(session_id)
    except KeyError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown session {session_id}") from exc
    except SessionStateError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc


class QuickStartRequest(BaseModel):
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    accuracy_m: Optional[float] = None


@router.post("/api/sessions/quick-start")
def quick_start(body: QuickStartRequest, user: Dict = Depends(operator_only), services: Services = Depends(get_services)) -> Dict[str, Any]:
    """Start inspecting here: reuse the site inspected before near this position, or create one."""
    location = (
        {"latitude": body.latitude, "longitude": body.longitude, "accuracy_m": body.accuracy_m}
        if body.latitude is not None and body.longitude is not None
        else None
    )
    try:
        return services.mission.quick_start(user["username"], location)
    except LookupError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc


@router.get("/api/sessions")
def list_sessions(site_id: Optional[str] = None, _: Dict = Depends(any_user), services: Services = Depends(get_services)) -> List[Dict[str, Any]]:
    out = []
    for s in services.store.list_sessions(site_id):
        s = dict(s)
        s.pop("report", None)
        out.append(s)
    return out


@router.get("/api/sessions/interrupted")
def interrupted(_: Dict = Depends(operator_only), services: Services = Depends(get_services)) -> List[Dict[str, Any]]:
    return [{k: v for k, v in s.items() if k != "report"} for s in services.mission.interrupted()]


@router.post("/api/sessions/{session_id}/resume")
def resume(session_id: str, _: Dict = Depends(operator_only), services: Services = Depends(get_services)) -> Dict[str, Any]:
    try:
        return services.mission.resume(session_id)
    except KeyError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown session {session_id}") from exc
    except SessionStateError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc


@router.get("/api/sessions/{session_id}")
def get_session(session_id: str, _: Dict = Depends(any_user), services: Services = Depends(get_services)) -> Dict[str, Any]:
    s = services.store.get_session(session_id)
    if s is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown session {session_id}")
    return {**s, "observations": services.store.session_observations(session_id)}


@router.get("/api/sessions/{session_id}/report")
def report(
    session_id: str,
    format: str = Query("json", pattern="^(json|html|pdf)$"),
    _: Dict = Depends(any_user),
    services: Services = Depends(get_services),
):
    try:
        rep = services.mission.report(session_id)
    except KeyError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown session {session_id}") from exc
    if format == "html":
        return HTMLResponse(render_html(rep))
    if format == "pdf":
        return Response(
            render_pdf(rep), media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="audi_report_{session_id}.pdf"'},
        )
    return rep
