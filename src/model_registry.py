from dataclasses import dataclass
from typing import Any


@dataclass
class ModelSpec:
    name: str
    family: str
    params: dict[str, Any]


class ModelRegistry:
    def __init__(self) -> None:
        self._factories = {
            "mediapipe_hand_landmarks": self._build_landmarks,
            "pose_estimation": self._build_pose,
            "video_classifier": self._build_video_classifier,
        }

    def list_models(self) -> list[str]:
        return sorted(self._factories.keys())

    def create(self, name: str, params: dict[str, Any]) -> ModelSpec:
        if name not in self._factories:
            raise ValueError(f"Unknown model '{name}'")
        return self._factories[name](params)

    def _build_landmarks(self, params: dict[str, Any]) -> ModelSpec:
        return ModelSpec(name="mediapipe_hand_landmarks", family="keypoint", params=params)

    def _build_pose(self, params: dict[str, Any]) -> ModelSpec:
        return ModelSpec(name="pose_estimation", family="pose", params=params)

    def _build_video_classifier(self, params: dict[str, Any]) -> ModelSpec:
        return ModelSpec(name="video_classifier", family="video", params=params)
