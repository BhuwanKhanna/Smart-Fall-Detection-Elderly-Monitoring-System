from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class MonitorStartRequest(BaseModel):
    camera_index: int = 0
    pose_backend: Literal["auto", "mediapipe", "yolo"] = "auto"
    inactivity_seconds: int = Field(default=12, ge=5, le=120)


class SimulatedAlertRequest(BaseModel):
    event_type: Literal["fall", "collapse", "inactivity", "abnormal_motion"] = "fall"
    severity: Literal["warning", "critical"] = "critical"
    confidence: float = Field(default=0.91, ge=0.0, le=1.0)


class ControlResponse(BaseModel):
    ok: bool
    message: str
    status: dict[str, Any]


class EventRecord(BaseModel):
    id: int
    event_type: str
    severity: str
    confidence: float
    pose_backend: str
    status: str
    details: dict[str, Any]
    notification_channels: list[str]
    created_at: str


class DashboardSnapshot(BaseModel):
    monitoring: dict[str, Any]
    summary: dict[str, Any]
    events: list[EventRecord]
    alerts: list[EventRecord]
