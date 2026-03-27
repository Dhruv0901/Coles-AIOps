from typing import Any


class ROIExtractor:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def extract(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        hand_enabled = bool(self.config.get("hand_roi", True))
        chest_enabled = bool(self.config.get("chest_roi", True))
        output = []
        for sample in samples:
            rois = {}
            if hand_enabled:
                rois["hand"] = [0.1, 0.1, 0.6, 0.6]
            if chest_enabled:
                rois["chest"] = [0.3, 0.3, 0.5, 0.5]
            output.append({**sample, "rois": rois})
        return output
