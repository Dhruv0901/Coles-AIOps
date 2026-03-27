from typing import Any


class FeatureEngineer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def build(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        temporal_window = int(self.config.get("temporal_window", 8))
        include_geometry = bool(self.config.get("include_geometry", True))
        output = []
        for idx, sample in enumerate(samples):
            features = {
                "motion_delta": round((idx + 1) / max(len(samples), 1), 4),
                "temporal_window": temporal_window,
            }
            if include_geometry:
                features["geometry_score"] = round(0.5 + (idx % 5) * 0.1, 4)
            output.append({**sample, "features": features})
        return output
