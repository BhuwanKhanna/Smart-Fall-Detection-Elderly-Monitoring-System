from __future__ import annotations

import streamlit as st


def inject_theme() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Manrope:wght@400;500;600;700&display=swap');

            :root {
                --bg: #08111f;
                --panel: rgba(14, 24, 42, 0.82);
                --panel-strong: rgba(12, 18, 34, 0.92);
                --border: rgba(133, 193, 255, 0.16);
                --text: #eff6ff;
                --muted: #93abc8;
                --accent: #6ee7b7;
                --warning: #fbbf24;
                --danger: #fb7185;
                --highlight: #7dd3fc;
            }

            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(109, 211, 255, 0.18), transparent 28%),
                    radial-gradient(circle at top right, rgba(110, 231, 183, 0.18), transparent 25%),
                    linear-gradient(180deg, #091120 0%, #050b15 100%);
                color: var(--text);
                font-family: 'Manrope', sans-serif;
            }

            h1, h2, h3, h4 {
                font-family: 'Space Grotesk', sans-serif;
                letter-spacing: -0.03em;
            }

            [data-testid="stSidebar"] {
                background: rgba(6, 11, 23, 0.92);
                border-right: 1px solid var(--border);
            }

            .hero-card,
            .metric-card,
            .panel-card {
                background: var(--panel);
                border: 1px solid var(--border);
                border-radius: 24px;
                box-shadow: 0 18px 55px rgba(0, 0, 0, 0.28);
                backdrop-filter: blur(18px);
            }

            .hero-card {
                padding: 28px;
                margin-bottom: 18px;
            }

            .metric-grid {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 14px;
                margin: 12px 0 22px 0;
            }

            .metric-card {
                padding: 18px;
            }

            .metric-label {
                color: var(--muted);
                font-size: 0.82rem;
                text-transform: uppercase;
                letter-spacing: 0.12em;
                margin-bottom: 10px;
            }

            .metric-value {
                font-size: 1.7rem;
                font-weight: 700;
                color: var(--text);
            }

            .status-pill {
                display: inline-flex;
                align-items: center;
                gap: 10px;
                border-radius: 999px;
                padding: 8px 16px;
                background: rgba(110, 231, 183, 0.12);
                border: 1px solid rgba(110, 231, 183, 0.28);
                font-size: 0.9rem;
                color: #c5ffe8;
            }

            .status-pill.danger {
                background: rgba(251, 113, 133, 0.12);
                border-color: rgba(251, 113, 133, 0.3);
                color: #ffd1d8;
            }

            .subcopy {
                color: var(--muted);
                line-height: 1.6;
                max-width: 860px;
            }

            .section-title {
                margin-top: 6px;
                margin-bottom: 10px;
            }

            .stream-shell {
                border-radius: 24px;
                overflow: hidden;
                border: 1px solid var(--border);
                background: var(--panel-strong);
                min-height: 520px;
            }

            .table-shell {
                padding: 18px;
            }

            @media (max-width: 900px) {
                .metric-grid {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str) -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """


def hero_banner(running: bool, message: str) -> str:
    pill_class = "status-pill" if running else "status-pill danger"
    state = "Live monitoring active" if running else "Monitoring offline"
    return f"""
    <div class="hero-card">
        <div class="{pill_class}">{state}</div>
        <h1 style="margin: 18px 0 10px 0;">AI Smart Fall Detection & Elderly Monitoring</h1>
        <p class="subcopy">
            Real-time webcam surveillance with skeleton tracking, posture intelligence, temporal anomaly scoring,
            and emergency alert simulation for fall prevention demos and internship showcases.
        </p>
        <p class="subcopy" style="margin-top: 10px;"><strong>System note:</strong> {message}</p>
    </div>
    """
