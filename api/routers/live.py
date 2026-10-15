"""Live camera endpoints: start/stop/status, MJPEG overlay stream, capture, /ws/live."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from fastapi.exceptions import HTTPException as FastAPIHTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ml.config import cfg
from ml.live.manager import LiveStateError

from api.deps import field_roles, get_services, operator_only, user_from_token
from api.services import Services

logger = logging.getLogger(__name__)
router = APIRouter(tags=["live"])

BOUNDARY = "frame"


class LiveStart(BaseModel):
    session_id: str
    camera_source: Optional[Union[int, str]] = None
    mode: str = "camera"                    # camera | browser
    detection_mode: Optional[str] = None    # None = configured detector | full_frame | auto
    asset_type: Optional[str] = None        # for full_frame


class CaptureRequest(BaseModel):
    track_id: Optional[int] = None


@router.post("/api/live/start")
def live_start(body: LiveStart, _: Dict = Depends(operator_only), services: Services = Depends(get_services)) -> Dict[str, Any]:
    if body.mode not in {"camera", "browser"}:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "mode must be camera or browser")
    if body.detection_mode not in {None, "full_frame", "yolo_world", "auto"}:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "detection_mode must be auto, full_frame or yolo_world")
    try:
        return services.live.start(
            body.session_id, body.camera_source, body.mode,
            body.detection_mode if body.detection_mode in {"full_frame", "auto"} else None, body.asset_type,
        )
    except KeyError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown session {body.session_id}") from exc
    except LiveStateError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc


@router.post("/api/live/stop")
def live_stop(_: Dict = Depends(operator_only), services: Services = Depends(get_services)) -> Dict[str, Any]:
    return services.live.stop()


@router.get("/api/live/status")
def live_status(_: Dict = Depends(field_roles), services: Services = Depends(get_services)) -> Dict[str, Any]:
    return services.live.status()


@router.post("/api/live/capture")
def live_capture(body: CaptureRequest, _: Dict = Depends(operator_only), services: Services = Depends(get_services)) -> Dict[str, Any]:
    try:
        track = services.live.capture(body.track_id)
    except LiveStateError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    return {"forced_track_id": track}


@router.get("/api/live/stream")
async def live_stream(token: str = Query(...), services: Services = Depends(get_services)):
    """MJPEG of the annotated frame (an <img> cannot send headers, so the JWT comes as ?token=)."""
    user = user_from_token(services, token)
    if user["role"] not in {"operator", "maintenance", "admin"}:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Forbidden")
    period = 1.0 / float(cfg(services.config, "live", "stream_fps"))

    async def gen():
        while services.live.running:
            jpg = await asyncio.to_thread(services.live.annotated_jpeg)
            if jpg:
                yield (f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpg)}\r\n\r\n").encode() + jpg + b"\r\n"
            await asyncio.sleep(period)

    return StreamingResponse(gen(), media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}")


@router.websocket("/ws/live")
async def ws_live(ws: WebSocket) -> None:
    """JSON events out; binary JPEG frames in (browser-camera mode)."""
    services: Services = ws.app.state.services
    try:
        user = user_from_token(services, ws.query_params.get("token", ""))
    except FastAPIHTTPException:
        await ws.close(code=4401)
        return
    await ws.accept()
    client = services.hub.register(ws, user["user_id"], user["role"], {"live", "notifications"})
    pump = asyncio.create_task(services.hub.pump(client))
    interval = float(cfg(services.config, "live", "status_interval_s"))

    async def status_loop():
        while True:
            await asyncio.sleep(interval)
            st = services.live.status()
            st.pop("tracks", None)
            services.hub.publish("live", {"type": "status", "status": st})

    status_task = asyncio.create_task(status_loop())
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes") is not None and user["role"] in {"operator", "admin"}:
                ok = await asyncio.to_thread(services.live.push_jpeg, msg["bytes"])
                if not ok:
                    await ws.send_json({"type": "frame_rejected"})
    except WebSocketDisconnect:
        logger.debug("live socket closed for %s", user["username"])
    finally:
        status_task.cancel()
        pump.cancel()
        services.hub.unregister(client)
