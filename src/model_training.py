from typing import Any

from src.model_registry import ModelSpec


class ModelTrainer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def train(self, model: ModelSpec, features: list[dict[str, Any]]) -> dict[str, Any]:
        epochs = int(self.config.get("epochs", 1))
        batch_size = int(self.config.get("batch_size", 8))
        quality = min(0.99, 0.55 + 0.03 * len(features) / max(batch_size, 1))
        return {
            "model_name": model.name,
            "family": model.family,
            "epochs": epochs,
            "batch_size": batch_size,
            "seen_samples": len(features),
            "quality_signal": round(quality, 4),
            "model_params": model.params,
        }
