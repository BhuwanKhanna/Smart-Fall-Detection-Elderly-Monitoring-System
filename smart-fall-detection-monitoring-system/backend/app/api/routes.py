from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response

from backend.app.models.schemas import ControlResponse, DashboardSnapshot, MonitorStartRequest, SimulatedAlertRequest


router = APIRouter()


def _service(request: Request):
    return request.app.state.monitoring


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/monitor/start", response_model=ControlResponse)
def start_monitoring(payload: MonitorStartRequest, request: Request) -> ControlResponse:
    try:
        status = _service(request).start(payload)
        return ControlResponse(ok=True, message="Monitoring started.", status=status)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/monitor/stop", response_model=ControlResponse)
def stop_monitoring(request: Request) -> ControlResponse:
    status = _service(request).stop()
    return ControlResponse(ok=True, message="Monitoring stopped.", status=status)


@router.get("/monitor/status")
def monitor_status(request: Request) -> dict:
    return _service(request).get_status()


@router.get("/monitor/frame")
def monitor_frame(request: Request) -> Response:
    frame = _service(request).get_frame()
    if frame is None:
        raise HTTPException(status_code=404, detail="No frame available. Start monitoring first.")
    return Response(content=frame, media_type="image/jpeg")


@router.post("/monitor/simulate-alert")
def simulate_alert(payload: SimulatedAlertRequest, request: Request) -> dict:
    return _service(request).simulate_alert(payload)


@router.get("/events")
def list_events(request: Request, limit: int = Query(default=50, ge=1, le=200)) -> list[dict]:
    return _service(request).list_events(limit)


@router.get("/alerts")
def list_alerts(request: Request, limit: int = Query(default=25, ge=1, le=100)) -> list[dict]:
    return _service(request).list_alerts(limit)


@router.get("/dashboard", response_model=DashboardSnapshot)
def dashboard_snapshot(request: Request) -> DashboardSnapshot:
    return DashboardSnapshot(**_service(request).dashboard_snapshot())
