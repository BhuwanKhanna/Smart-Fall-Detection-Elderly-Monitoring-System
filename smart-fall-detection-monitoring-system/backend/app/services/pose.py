from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


YOLO_KEYPOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

SKELETON_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]


@dataclass(slots=True)
class PoseResult:
    keypoints: dict[str, tuple[float, float, float]]
    bbox: tuple[int, int, int, int] | None
    confidence: float
    vertical_ratio: float
    torso_angle: float
    body_width_ratio: float
    motion_magnitude: float
    hip_velocity: float
    center: tuple[float, float]
    hip_y: float
    pose_backend: str


def create_pose_estimator(preferred_backend: str):
    choices = ["mediapipe", "yolo"] if preferred_backend == "auto" else [preferred_backend]

    for backend in choices:
        try:
            if backend == "mediapipe":
                return MediaPipePoseEstimator()
            if backend == "yolo":
                return YoloPoseEstimator()
        except Exception:
            continue

    raise RuntimeError(
        "No pose backend is available. Install mediapipe for the default backend or ultralytics for YOLO Pose."
    )


def draw_pose(frame: np.ndarray, pose: PoseResult | None) -> np.ndarray:
    if pose is None:
        return frame

    if pose.bbox:
        x1, y1, x2, y2 = pose.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (33, 194, 129), 2)

    for start, end in SKELETON_CONNECTIONS:
        if start in pose.keypoints and end in pose.keypoints:
            pt1 = (int(pose.keypoints[start][0]), int(pose.keypoints[start][1]))
            pt2 = (int(pose.keypoints[end][0]), int(pose.keypoints[end][1]))
            cv2.line(frame, pt1, pt2, (69, 149, 236), 2)

    for x, y, visibility in pose.keypoints.values():
        color = (33, 194, 129) if visibility >= 0.5 else (0, 145, 255)
        cv2.circle(frame, (int(x), int(y)), 4, color, -1)

    return frame


class MediaPipePoseEstimator:
    def __init__(self) -> None:
        import mediapipe as mp

        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.backend_name = "mediapipe"

    def estimate(self, frame: np.ndarray, previous: PoseResult | None = None) -> PoseResult | None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)
        if not result.pose_landmarks:
            return None

        height, width = frame.shape[:2]
        keypoints: dict[str, tuple[float, float, float]] = {}
        xs: list[float] = []
        ys: list[float] = []
        visibilities: list[float] = []

        landmark_names = [
            "nose",
            "left_eye_inner",
            "left_eye",
            "left_eye_outer",
            "right_eye_inner",
            "right_eye",
            "right_eye_outer",
            "left_ear",
            "right_ear",
            "mouth_left",
            "mouth_right",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_pinky",
            "right_pinky",
            "left_index",
            "right_index",
            "left_thumb",
            "right_thumb",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
            "left_heel",
            "right_heel",
            "left_foot_index",
            "right_foot_index",
        ]

        for index, landmark in enumerate(result.pose_landmarks.landmark):
            x = landmark.x * width
            y = landmark.y * height
            visibility = float(landmark.visibility)
            name = landmark_names[index]
            keypoints[name] = (x, y, visibility)
            if visibility >= 0.4:
                xs.append(x)
                ys.append(y)
                visibilities.append(visibility)

        bbox = None
        if xs and ys:
            bbox = (
                max(int(min(xs)) - 20, 0),
                max(int(min(ys)) - 20, 0),
                min(int(max(xs)) + 20, width),
                min(int(max(ys)) + 20, height),
            )

        return _build_pose_result(keypoints, bbox, visibilities, previous, self.backend_name, frame.shape)


class YoloPoseEstimator:
    def __init__(self) -> None:
        from ultralytics import YOLO

        self.model = YOLO("yolov8n-pose.pt")
        self.backend_name = "yolo"

    def estimate(self, frame: np.ndarray, previous: PoseResult | None = None) -> PoseResult | None:
        predictions = self.model.predict(frame, verbose=False, conf=0.35)
        if not predictions:
            return None

        prediction = predictions[0]
        if prediction.keypoints is None or prediction.boxes is None or len(prediction.boxes) == 0:
            return None

        points = prediction.keypoints.xy[0].tolist()
        confidences = prediction.keypoints.conf[0].tolist()
        box = prediction.boxes.xyxy[0].tolist()
        keypoints = {
            name: (float(point[0]), float(point[1]), float(score))
            for name, point, score in zip(YOLO_KEYPOINTS, points, confidences)
        }
        bbox = tuple(int(value) for value in box)
        visible = [float(score) for score in confidences if float(score) >= 0.4]
        return _build_pose_result(keypoints, bbox, visible, previous, self.backend_name, frame.shape)


def _build_pose_result(
    keypoints: dict[str, tuple[float, float, float]],
    bbox: tuple[int, int, int, int] | None,
    visibilities: Iterable[float],
    previous: PoseResult | None,
    backend_name: str,
    shape: tuple[int, int, int],
) -> PoseResult | None:
    required = ["left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_ankle", "right_ankle"]
    if not all(name in keypoints for name in required):
        return None

    visibility_list = list(visibilities)
    frame_height, frame_width = shape[:2]
    shoulder_center = _midpoint(keypoints["left_shoulder"], keypoints["right_shoulder"])
    hip_center = _midpoint(keypoints["left_hip"], keypoints["right_hip"])
    ankle_center = _midpoint(keypoints["left_ankle"], keypoints["right_ankle"])

    body_height = max(_distance(shoulder_center, ankle_center), 1.0)
    body_width = max(_distance(keypoints["left_shoulder"], keypoints["right_shoulder"]), 1.0)
    vertical_ratio = float(abs(ankle_center[1] - shoulder_center[1]) / frame_height)
    body_width_ratio = float(body_width / body_height)
    torso_angle = _torso_angle_degrees(shoulder_center, hip_center)

    center_x = float((shoulder_center[0] + hip_center[0]) / 2 / frame_width)
    center_y = float((shoulder_center[1] + hip_center[1]) / 2 / frame_height)
    hip_y = float(hip_center[1] / frame_height)
    motion_magnitude = 0.0
    hip_velocity = 0.0

    if previous is not None:
        motion_magnitude = _distance((center_x, center_y), previous.center)
        hip_velocity = abs(hip_y - previous.hip_y)

    confidence = float(sum(visibility_list) / max(len(visibility_list), 1))
    if confidence == 0.0:
        confidence = 0.55

    return PoseResult(
        keypoints=keypoints,
        bbox=bbox,
        confidence=round(confidence, 3),
        vertical_ratio=round(vertical_ratio, 3),
        torso_angle=round(torso_angle, 3),
        body_width_ratio=round(body_width_ratio, 3),
        motion_magnitude=round(motion_magnitude, 4),
        hip_velocity=round(hip_velocity, 4),
        center=(round(center_x, 4), round(center_y, 4)),
        hip_y=round(hip_y, 4),
        pose_backend=backend_name,
    )


def _midpoint(point_a: tuple[float, float, float], point_b: tuple[float, float, float]) -> tuple[float, float]:
    return (float((point_a[0] + point_b[0]) / 2), float((point_a[1] + point_b[1]) / 2))


def _distance(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    return math.dist(point_a, point_b)


def _torso_angle_degrees(shoulder_center: tuple[float, float], hip_center: tuple[float, float]) -> float:
    dx = hip_center[0] - shoulder_center[0]
    dy = hip_center[1] - shoulder_center[1]
    return float(math.degrees(math.atan2(abs(dx), max(abs(dy), 1e-5))))
