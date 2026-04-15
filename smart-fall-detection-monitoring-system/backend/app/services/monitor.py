from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any

import cv2
from backend.app.core.config import Settings
from backend.app.models.schemas import MonitorStartRequest, SimulatedAlertRequest
from backend.app.services.alerts import AlertDispatcher
from backend.app.services.pose import PoseResult, create_pose_estimator, draw_pose
from backend.app.services.storage import EventStore
from backend.app.services.temporal import TemporalAnalyzer


class MonitoringService:
    def __init__(self, settings: Settings, store: EventStore) -> None:
        self.settings = settings
        self.store = store
        self.alert_dispatcher = AlertDispatcher(settings)
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._latest_frame: bytes | None = None
        self._latest_status: dict[str, Any] = self._default_status()
        self._session_config = MonitorStartRequest(
            camera_index=settings.camera_index,
            pose_backend=settings.pose_backend if settings.pose_backend in {"auto", "mediapipe", "yolo"} else "auto",
            inactivity_seconds=settings.inactivity_seconds,
        )
        self._pose_estimator = None
        self._temporal: TemporalAnalyzer | None = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, payload: MonitorStartRequest) -> dict[str, Any]:
        with self._lock:
            if self.is_running:
                return self.get_status()

            self._session_config = payload
            self._pose_estimator = create_pose_estimator(payload.pose_backend)
            self._temporal = TemporalAnalyzer(
                model_path=self.settings.model_path,
                inactivity_seconds=payload.inactivity_seconds,
            )
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            self._latest_status.update(
                {
                    "running": True,
                    "message": "Monitoring started.",
                    "pose_backend": self._pose_estimator.backend_name,
                    "camera_index": payload.camera_index,
                }
            )
            return self.get_status()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if not self.is_running:
                self._latest_status["message"] = "Monitoring is already stopped."
                return self.get_status()

            self._stop_event.set()
            if self._thread:
                self._thread.join(timeout=2)
            self._thread = None
            self._latest_status.update({"running": False, "message": "Monitoring stopped."})
            return self.get_status()

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            status = dict(self._latest_status)
        status["summary"] = self.store.dashboard_summary()
        return status

    def get_frame(self) -> bytes | None:
        with self._lock:
            return self._latest_frame

    def list_events(self, limit: int = 50) -> list[dict[str, Any]]:
        return self.store.list_events(limit)

    def list_alerts(self, limit: int = 25) -> list[dict[str, Any]]:
        return self.store.list_alerts(limit)

    def dashboard_snapshot(self) -> dict[str, Any]:
        return {
            "monitoring": self.get_status(),
            "summary": self.store.dashboard_summary(),
            "events": self.store.list_events(50),
            "alerts": self.store.list_alerts(20),
        }

    def simulate_alert(self, payload: SimulatedAlertRequest) -> dict[str, Any]:
        created_at = datetime.now(timezone.utc).isoformat()
        event = {
            "event_type": payload.event_type,
            "severity": payload.severity,
            "confidence": payload.confidence,
            "created_at": created_at,
        }
        channels = self.alert_dispatcher.dispatch(event)
        event_id = self.store.log_event(
            event_type=payload.event_type,
            severity=payload.severity,
            confidence=payload.confidence,
            pose_backend=self._latest_status.get("pose_backend", "simulation"),
            status="simulated-alerted",
            details={
                "source": "manual-demo-trigger",
                "note": "Generated from dashboard controls.",
            },
            notification_channels=channels,
        )
        return {"id": event_id, **event, "notification_channels": channels}

    def _run_loop(self) -> None:
        capture = cv2.VideoCapture(self._session_config.camera_index)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.frame_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.frame_height)

        previous_pose: PoseResult | None = None
        previous_tick = time.perf_counter()

        if not capture.isOpened():
            with self._lock:
                self._latest_status.update(
                    {
                        "running": False,
                        "message": "Unable to access webcam. Check permissions or camera index.",
                    }
                )
            self._thread = None
            return

        while not self._stop_event.is_set():
            ok, frame = capture.read()
            if not ok:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            now = time.time()
            tick = time.perf_counter()
            fps = 1.0 / max(tick - previous_tick, 1e-6)
            previous_tick = tick

            pose = self._pose_estimator.estimate(frame, previous_pose) if self._pose_estimator else None
            analysis = self._temporal.update(pose, now) if self._temporal else self._default_analysis()
            if pose is not None:
                previous_pose = pose

            for event_type in analysis["triggers"]:
                self._handle_detected_event(event_type, analysis, pose)

            annotated = self._render_overlay(frame, pose, analysis, fps)
            success, buffer = cv2.imencode(".jpg", annotated)
            if success:
                with self._lock:
                    self._latest_frame = buffer.tobytes()
                    self._latest_status = {
                        "running": True,
                        "message": "Monitoring live.",
                        "pose_backend": pose.pose_backend if pose else self._latest_status.get("pose_backend", "auto"),
                        "camera_index": self._session_config.camera_index,
                        "person_detected": analysis["detected"],
                        "activity_state": analysis["activity_state"],
                        "risk_level": analysis["risk_level"],
                        "confidence": analysis["confidence"],
                        "inactive_for": analysis["inactive_for"],
                        "probabilities": analysis["probabilities"],
                        "fps": round(fps, 1),
                        "last_event_types": analysis["triggers"],
                        "last_updated_at": datetime.now(timezone.utc).isoformat(),
                    }

            time.sleep(0.01)

        capture.release()

    def _handle_detected_event(self, event_type: str, analysis: dict[str, Any], pose: PoseResult | None) -> None:
        severity = "critical" if event_type in {"fall", "collapse"} else "warning"
        confidence = float(analysis["probabilities"].get(event_type, analysis["confidence"]))
        created_at = datetime.now(timezone.utc).isoformat()
        details = {
            "activity_state": analysis["activity_state"],
            "risk_level": analysis["risk_level"],
            "probabilities": analysis["probabilities"],
            "pose_snapshot": {
                "vertical_ratio": pose.vertical_ratio if pose else None,
                "torso_angle": pose.torso_angle if pose else None,
                "motion_magnitude": pose.motion_magnitude if pose else None,
                "hip_velocity": pose.hip_velocity if pose else None,
            },
        }
        event = {
            "event_type": event_type,
            "severity": severity,
            "confidence": confidence,
            "created_at": created_at,
        }
        channels = self.alert_dispatcher.dispatch(event)
        self.store.log_event(
            event_type=event_type,
            severity=severity,
            confidence=confidence,
            pose_backend=pose.pose_backend if pose else "unknown",
            status="alerted",
            details=details,
            notification_channels=channels,
        )

    def _render_overlay(
        self,
        frame: np.ndarray,
        pose: PoseResult | None,
        analysis: dict[str, Any],
        fps: float,
    ) -> np.ndarray:
        overlay = draw_pose(frame.copy(), pose)
        cv2.rectangle(overlay, (20, 18), (510, 178), (15, 23, 42), -1)
        cv2.putText(overlay, "Elderly Safety Monitor", (36, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (245, 247, 250), 2)
        cv2.putText(
            overlay,
            f"State: {analysis['activity_state']}  Risk: {analysis['risk_level']}",
            (36, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (144, 211, 255),
            2,
        )
        cv2.putText(
            overlay,
            f"Confidence: {analysis['confidence']:.2f}  FPS: {fps:.1f}",
            (36, 108),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (235, 245, 255),
            2,
        )
        cv2.putText(
            overlay,
            f"Fall {analysis['probabilities']['fall']:.2f} | Collapse {analysis['probabilities']['collapse']:.2f}",
            (36, 136),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.57,
            (255, 214, 102),
            2,
        )
        cv2.putText(
            overlay,
            f"Inactivity {analysis['probabilities']['inactivity']:.2f} | Abnormal {analysis['probabilities']['abnormal_motion']:.2f}",
            (36, 164),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.57,
            (255, 214, 102),
            2,
        )

        if analysis["triggers"]:
            cv2.rectangle(overlay, (20, 198), (420, 245), (36, 28, 83), -1)
            cv2.putText(
                overlay,
                f"ALERT: {', '.join(event.replace('_', ' ').title() for event in analysis['triggers'])}",
                (32, 228),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (255, 112, 112),
                2,
            )

        return overlay

    @staticmethod
    def _default_status() -> dict[str, Any]:
        return {
            "running": False,
            "message": "Monitoring has not been started.",
            "pose_backend": "auto",
            "camera_index": 0,
            "person_detected": False,
            "activity_state": "Idle",
            "risk_level": "Normal",
            "confidence": 0.0,
            "inactive_for": 0.0,
            "probabilities": {
                "fall": 0.0,
                "collapse": 0.0,
                "inactivity": 0.0,
                "abnormal_motion": 0.0,
            },
            "fps": 0.0,
            "last_event_types": [],
            "last_updated_at": None,
        }

    @staticmethod
    def _default_analysis() -> dict[str, Any]:
        return {
            "detected": False,
            "activity_state": "Idle",
            "risk_level": "Normal",
            "confidence": 0.0,
            "inactive_for": 0.0,
            "probabilities": {
                "fall": 0.0,
                "collapse": 0.0,
                "inactivity": 0.0,
                "abnormal_motion": 0.0,
            },
            "triggers": [],
        }
