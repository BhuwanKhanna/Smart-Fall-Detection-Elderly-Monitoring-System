from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class EventStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(database_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

    def initialize(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    pose_backend TEXT NOT NULL,
                    status TEXT NOT NULL,
                    details TEXT NOT NULL,
                    notification_channels TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            self._conn.commit()

    def log_event(
        self,
        *,
        event_type: str,
        severity: str,
        confidence: float,
        pose_backend: str,
        status: str,
        details: dict[str, Any],
        notification_channels: list[str],
    ) -> int:
        payload = (
            event_type,
            severity,
            confidence,
            pose_backend,
            status,
            json.dumps(details),
            json.dumps(notification_channels),
            datetime.now(timezone.utc).isoformat(),
        )
        with self._lock:
            cursor = self._conn.execute(
                """
                INSERT INTO events (
                    event_type,
                    severity,
                    confidence,
                    pose_backend,
                    status,
                    details,
                    notification_channels,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )
            self._conn.commit()
            return int(cursor.lastrowid)

    def list_events(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT *
                FROM events
                ORDER BY datetime(created_at) DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def list_alerts(self, limit: int = 25) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT *
                FROM events
                WHERE severity IN ('critical', 'warning')
                ORDER BY datetime(created_at) DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def dashboard_summary(self) -> dict[str, Any]:
        with self._lock:
            total_events = self._conn.execute("SELECT COUNT(*) AS count FROM events").fetchone()["count"]
            critical_events = self._conn.execute(
                "SELECT COUNT(*) AS count FROM events WHERE severity = 'critical'"
            ).fetchone()["count"]
            warning_events = self._conn.execute(
                "SELECT COUNT(*) AS count FROM events WHERE severity = 'warning'"
            ).fetchone()["count"]
            latest = self._conn.execute(
                """
                SELECT event_type, severity, confidence, created_at
                FROM events
                ORDER BY datetime(created_at) DESC
                LIMIT 1
                """
            ).fetchone()

        return {
            "total_events": total_events,
            "critical_events": critical_events,
            "warning_events": warning_events,
            "latest_event": dict(latest) if latest else None,
        }

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["details"] = json.loads(data["details"])
        data["notification_channels"] = json.loads(data["notification_channels"])
        return data
