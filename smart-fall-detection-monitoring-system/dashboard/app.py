from __future__ import annotations

import os
from io import StringIO

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from streamlit_autorefresh import st_autorefresh

from dashboard.components.api_client import APIClient
from dashboard.components.theme import hero_banner, inject_theme, metric_card


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT_DIR, ".env"))

DEFAULT_API_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Smart Fall Detection Dashboard",
    page_icon="SF",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_theme()
st_autorefresh(interval=3000, key="dashboard_refresh")


def _safe_api_client() -> APIClient:
    base_url = st.session_state.get("api_url", DEFAULT_API_URL)
    return APIClient(base_url)


def _fetch_snapshot() -> dict:
    try:
        return _safe_api_client().get("/api/dashboard")
    except Exception as exc:
        st.error(f"Unable to reach backend API: {exc}")
        return {
            "monitoring": {
                "running": False,
                "message": "API unavailable",
                "pose_backend": "unknown",
                "activity_state": "Unavailable",
                "risk_level": "Unknown",
                "confidence": 0.0,
                "inactive_for": 0.0,
                "fps": 0.0,
                "probabilities": {
                    "fall": 0.0,
                    "collapse": 0.0,
                    "inactivity": 0.0,
                    "abnormal_motion": 0.0,
                },
                "summary": {
                    "total_events": 0,
                    "critical_events": 0,
                    "warning_events": 0,
                    "latest_event": None,
                },
            },
            "summary": {
                "total_events": 0,
                "critical_events": 0,
                "warning_events": 0,
                "latest_event": None,
            },
            "events": [],
            "alerts": [],
        }


def _render_live_feed(api_url: str, running: bool) -> None:
    if not running:
        st.info("Start monitoring to stream the annotated camera feed here.")
        return

    components.html(
        f"""
        <div class="stream-shell">
            <img id="live-feed" src="{api_url}/api/monitor/frame?tick={os.urandom(4).hex()}"
                 style="width:100%;height:520px;object-fit:cover;display:block;" />
        </div>
        <script>
            const feed = document.getElementById("live-feed");
            const refresh = () => {{
                feed.src = "{api_url}/api/monitor/frame?tick=" + Date.now();
            }};
            setInterval(refresh, 900);
        </script>
        """,
        height=540,
    )


def _post_action(path: str, payload: dict | None = None, success_message: str | None = None) -> None:
    try:
        _safe_api_client().post(path, payload)
        if success_message:
            st.success(success_message)
    except Exception as exc:
        st.error(f"Request failed: {exc}")


if "api_url" not in st.session_state:
    st.session_state["api_url"] = DEFAULT_API_URL

with st.sidebar:
    st.header("Control Center")
    st.session_state["api_url"] = st.text_input("FastAPI base URL", value=st.session_state["api_url"])
    camera_index = st.number_input("Camera index", min_value=0, max_value=10, value=0, step=1)
    pose_backend = st.selectbox("Pose backend", ["auto", "mediapipe", "yolo"], index=0)
    inactivity_seconds = st.slider("Inactivity threshold (seconds)", min_value=5, max_value=60, value=12)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Start", use_container_width=True):
            _post_action(
                "/api/monitor/start",
                {
                    "camera_index": int(camera_index),
                    "pose_backend": pose_backend,
                    "inactivity_seconds": int(inactivity_seconds),
                },
                "Monitoring started.",
            )
    with col_b:
        if st.button("Stop", use_container_width=True):
            _post_action("/api/monitor/stop", success_message="Monitoring stopped.")

    st.divider()
    st.subheader("Demo Alerts")
    if st.button("Simulate Fall", use_container_width=True):
        _post_action(
            "/api/monitor/simulate-alert",
            {"event_type": "fall", "severity": "critical", "confidence": 0.93},
            "Simulated fall alert dispatched.",
        )
    if st.button("Simulate Inactivity", use_container_width=True):
        _post_action(
            "/api/monitor/simulate-alert",
            {"event_type": "inactivity", "severity": "warning", "confidence": 0.88},
            "Simulated inactivity alert dispatched.",
        )
    if st.button("Simulate Abnormal Motion", use_container_width=True):
        _post_action(
            "/api/monitor/simulate-alert",
            {"event_type": "abnormal_motion", "severity": "warning", "confidence": 0.82},
            "Simulated abnormal motion alert dispatched.",
        )

snapshot = _fetch_snapshot()
monitoring = snapshot["monitoring"]
summary = snapshot["summary"]
events_df = pd.DataFrame(snapshot["events"])
alerts_df = pd.DataFrame(snapshot["alerts"])

st.markdown(hero_banner(monitoring["running"], monitoring["message"]), unsafe_allow_html=True)

metric_markup = "".join(
    [
        metric_card("Activity", monitoring["activity_state"]),
        metric_card("Risk Level", monitoring["risk_level"]),
        metric_card("Confidence", f"{monitoring['confidence']:.2f}"),
        metric_card("FPS", f"{monitoring['fps']:.1f}"),
        metric_card("Pose Backend", monitoring["pose_backend"].upper()),
        metric_card("Inactive For", f"{monitoring['inactive_for']:.1f}s"),
        metric_card("Total Events", str(summary["total_events"])),
        metric_card("Critical Alerts", str(summary["critical_events"])),
    ]
)
st.markdown(f'<div class="metric-grid">{metric_markup}</div>', unsafe_allow_html=True)

left, right = st.columns([1.4, 1], gap="large")

with left:
    st.markdown('<h3 class="section-title">Live Camera Feed</h3>', unsafe_allow_html=True)
    _render_live_feed(st.session_state["api_url"], monitoring["running"])

with right:
    st.markdown('<h3 class="section-title">Risk Breakdown</h3>', unsafe_allow_html=True)
    probabilities = monitoring["probabilities"]
    risk_df = pd.DataFrame(
        {
            "Pattern": ["Fall", "Collapse", "Inactivity", "Abnormal motion"],
            "Score": [
                probabilities["fall"],
                probabilities["collapse"],
                probabilities["inactivity"],
                probabilities["abnormal_motion"],
            ],
        }
    )
    st.bar_chart(risk_df.set_index("Pattern"))

    st.markdown('<h3 class="section-title">System Snapshot</h3>', unsafe_allow_html=True)
    latest_event = summary.get("latest_event") or {}
    st.markdown(
        f"""
        <div class="panel-card table-shell">
            <p><strong>Person detected:</strong> {monitoring.get("person_detected", False)}</p>
            <p><strong>Last update:</strong> {monitoring.get("last_updated_at") or 'Not available'}</p>
            <p><strong>Latest incident:</strong> {latest_event.get('event_type', 'No incidents logged')}</p>
            <p><strong>Warning alerts:</strong> {summary.get('warning_events', 0)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<h3 class="section-title">Alert History</h3>', unsafe_allow_html=True)
if alerts_df.empty:
    st.info("No alerts recorded yet. Use the simulation buttons or start live monitoring.")
else:
    st.dataframe(alerts_df, use_container_width=True, hide_index=True)

st.markdown('<h3 class="section-title">Event Log Dashboard</h3>', unsafe_allow_html=True)
if events_df.empty:
    st.info("Event log is empty. Live incidents and simulations will appear here.")
else:
    st.dataframe(events_df, use_container_width=True, hide_index=True)
    csv_buffer = StringIO()
    events_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download event history CSV",
        data=csv_buffer.getvalue(),
        file_name="fall_detection_event_history.csv",
        mime="text/csv",
    )
