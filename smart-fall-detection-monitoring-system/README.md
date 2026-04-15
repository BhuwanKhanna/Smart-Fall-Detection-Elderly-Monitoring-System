# Smart Fall Detection & Elderly Monitoring System

A production-style internship portfolio project that uses live webcam video, pose estimation, temporal sequence analysis, and a modern dashboard to detect falls, sudden collapses, prolonged inactivity, and abnormal motion patterns in real time.

## What this project includes

- Live webcam monitoring with OpenCV
- Pose estimation abstraction with `MediaPipe` first and `YOLO Pose` support when available
- Temporal analysis pipeline with an LSTM-ready analyzer and heuristic fallback for portfolio demos
- FastAPI backend for monitoring control, status APIs, alert simulation, and event history
- Streamlit dashboard with live camera feed, skeleton overlay, confidence score, alert history, and responsive UI
- SQLite event logging for incidents and notifications
- Simulated email and SMS emergency alerts with real credential support through environment variables
- Clean project structure for local development, demos, and containerized deployment

## Architecture

1. FastAPI starts a monitoring worker that captures frames from the webcam.
2. A pose estimator extracts landmarks and posture features from each frame.
3. A temporal analyzer builds a rolling feature sequence and scores risky events.
4. Critical events are logged and routed to simulated email/SMS notification channels.
5. Streamlit consumes the API for the live dashboard, history, and controls.

## Project structure

```text
smart-fall-detection-monitoring-system/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── models/
│   │   └── services/
│   ├── artifacts/
│   └── Dockerfile
├── dashboard/
│   ├── components/
│   └── Dockerfile
├── .env.example
├── docker-compose.yml
├── requirements.txt
└── requirements-ml.txt
```

## Local setup

1. Create a virtual environment.
2. Install the core dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Optionally install the heavier ML extras for TensorFlow and YOLO Pose:

   ```bash
   pip install -r requirements-ml.txt
   ```

4. Copy `.env.example` to `.env` and update any alert credentials you want to use.
5. Start the API:

   ```bash
   uvicorn backend.app.main:app --reload --port 8000
   ```

6. Start the dashboard:

   ```bash
   streamlit run dashboard/app.py
   ```

## Demo mode notes

- If `tensorflow` or a trained LSTM checkpoint is not installed, the system still runs with a heuristic temporal scorer.
- If outbound email or SMS credentials are missing, notifications are safely simulated and still logged.
- If `ultralytics` is missing, YOLO Pose automatically falls back to MediaPipe or the selected available backend.

## Environment variables

Important settings are documented in `.env.example`, including:

- `POSE_BACKEND`
- `MODEL_PATH`
- `ALERT_EMAIL_ENABLED`
- `SMTP_HOST`
- `SMTP_USERNAME`
- `TWILIO_ACCOUNT_SID`
- `TWILIO_AUTH_TOKEN`
- `EMERGENCY_CONTACT_EMAIL`
- `EMERGENCY_CONTACT_PHONE`

## Portfolio talking points

- Real-time computer vision monitoring pipeline
- Human pose estimation plus temporal sequence intelligence
- Safety workflow design with explainable alerts and persistent event logs
- End-to-end product thinking: backend APIs, monitoring engine, data storage, dashboard UI, and deployment support
