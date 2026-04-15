from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from backend.app.services.pose import PoseResult

try:
    import tensorflow as tf
except Exception:  # pragma: no cover - optional dependency
    tf = None


class TemporalAnalyzer:
    def __init__(self, model_path: Path, inactivity_seconds: int = 12, sequence_length: int = 24) -> None:
        self.model_path = model_path
        self.inactivity_seconds = inactivity_seconds
        self.sequence_length = sequence_length
        self.sequence: deque[np.ndarray] = deque(maxlen=sequence_length)
        self.last_active_timestamp: float | None = None
        self.last_alert_timestamp: dict[str, float] = {}
        self.model = self._load_model()

    def update(self, pose: PoseResult | None, timestamp: float) -> dict[str, Any]:
        if self.last_active_timestamp is None:
            self.last_active_timestamp = timestamp

        if pose is None:
            inactive_for = timestamp - self.last_active_timestamp
            return self._build_response(
                pose=None,
                probabilities=self._empty_probabilities(inactive_for),
                inactive_for=inactive_for,
                triggers=[],
                detected=False,
            )

        feature_vector = np.array(
            [
                pose.vertical_ratio,
                pose.torso_angle / 90.0,
                pose.body_width_ratio,
                pose.motion_magnitude,
                pose.hip_velocity,
                pose.confidence,
            ],
            dtype=np.float32,
        )
        self.sequence.append(feature_vector)

        if pose.motion_magnitude > 0.012 or pose.hip_velocity > 0.012:
            self.last_active_timestamp = timestamp

        inactive_for = timestamp - self.last_active_timestamp
        heuristic_probabilities = self._heuristic_probabilities(pose, inactive_for)
        model_probabilities = self._model_probabilities()
        combined = self._combine_probabilities(heuristic_probabilities, model_probabilities)
        triggers = self._select_triggers(combined, timestamp)

        return self._build_response(
            pose=pose,
            probabilities=combined,
            inactive_for=inactive_for,
            triggers=triggers,
            detected=True,
        )

    def _heuristic_probabilities(self, pose: PoseResult, inactive_for: float) -> dict[str, float]:
        horizontal_factor = _clamp((0.62 - pose.vertical_ratio) / 0.18, 0.0, 1.0)
        torso_factor = _clamp(pose.torso_angle / 80.0, 0.0, 1.0)
        motion_factor = _clamp(pose.motion_magnitude / 0.08, 0.0, 1.0)
        drop_factor = _clamp(pose.hip_velocity / 0.08, 0.0, 1.0)
        inactivity_factor = _clamp(inactive_for / max(self.inactivity_seconds, 1), 0.0, 1.0)

        recent_motion_values = [float(vector[3]) for vector in self.sequence]
        motion_variation = float(np.std(recent_motion_values)) if recent_motion_values else 0.0
        abnormal_factor = _clamp((motion_factor * 0.6) + (motion_variation / 0.04) * 0.4, 0.0, 1.0)

        return {
            "fall": round(_clamp(horizontal_factor * 0.45 + torso_factor * 0.25 + drop_factor * 0.2 + motion_factor * 0.1), 3),
            "collapse": round(_clamp(drop_factor * 0.5 + torso_factor * 0.2 + motion_factor * 0.2 + horizontal_factor * 0.1), 3),
            "inactivity": round(inactivity_factor, 3),
            "abnormal_motion": round(abnormal_factor, 3),
        }

    def _model_probabilities(self) -> dict[str, float] | None:
        if self.model is None or len(self.sequence) < self.sequence_length:
            return None

        sequence = np.expand_dims(np.array(self.sequence, dtype=np.float32), axis=0)
        prediction = self.model.predict(sequence, verbose=0)[0]
        return {
            "fall": round(float(prediction[0]), 3),
            "collapse": round(float(prediction[1]), 3),
            "inactivity": round(float(prediction[2]), 3),
            "abnormal_motion": round(float(prediction[3]), 3),
        }

    def _combine_probabilities(
        self,
        heuristic: dict[str, float],
        model: dict[str, float] | None,
    ) -> dict[str, float]:
        if model is None:
            return heuristic
        return {
            key: round((heuristic[key] * 0.55) + (model[key] * 0.45), 3)
            for key in heuristic
        }

    def _select_triggers(self, probabilities: dict[str, float], timestamp: float) -> list[str]:
        thresholds = {
            "fall": 0.72,
            "collapse": 0.74,
            "inactivity": 0.96,
            "abnormal_motion": 0.78,
        }
        triggers: list[str] = []

        for event_type, threshold in thresholds.items():
            if probabilities[event_type] >= threshold and self._cooldown_ready(event_type, timestamp):
                self.last_alert_timestamp[event_type] = timestamp
                triggers.append(event_type)
        return triggers

    def _cooldown_ready(self, event_type: str, timestamp: float) -> bool:
        return (timestamp - self.last_alert_timestamp.get(event_type, 0.0)) >= 8.0

    def _build_response(
        self,
        *,
        pose: PoseResult | None,
        probabilities: dict[str, float],
        inactive_for: float,
        triggers: list[str],
        detected: bool,
    ) -> dict[str, Any]:
        confidence = max(probabilities.values()) if probabilities else 0.0
        risk_level = "Normal"
        if confidence >= 0.75:
            risk_level = "High"
        elif confidence >= 0.5:
            risk_level = "Elevated"

        activity_state = "No person detected"
        if pose is not None:
            if inactive_for >= self.inactivity_seconds:
                activity_state = "Inactive"
            elif pose.motion_magnitude > 0.025:
                activity_state = "Moving"
            else:
                activity_state = "Stable"

        return {
            "detected": detected,
            "activity_state": activity_state,
            "risk_level": risk_level,
            "confidence": round(confidence, 3),
            "inactive_for": round(inactive_for, 2),
            "probabilities": probabilities,
            "triggers": triggers,
        }

    def _empty_probabilities(self, inactive_for: float) -> dict[str, float]:
        return {
            "fall": 0.0,
            "collapse": 0.0,
            "inactivity": round(_clamp(inactive_for / max(self.inactivity_seconds, 1), 0.0, 1.0), 3),
            "abnormal_motion": 0.0,
        }

    def _load_model(self):
        if tf is None or not self.model_path.exists():
            return None
        try:
            return tf.keras.models.load_model(self.model_path)
        except Exception:
            return None


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def build_lstm_architecture(sequence_length: int = 24, feature_count: int = 6):
    if tf is None:
        raise RuntimeError("TensorFlow is not installed.")

    inputs = tf.keras.Input(shape=(sequence_length, feature_count))
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(inputs)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.LSTM(32)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(4, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model
