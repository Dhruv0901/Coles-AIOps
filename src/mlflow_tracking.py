import importlib
from pathlib import Path
from typing import Any

from src.utils import ensure_dir, write_json


class MLflowTracker:
    def __init__(self, tracking_uri: str, experiment_name: str) -> None:
        self.mlflow = importlib.import_module("mlflow")
        self.mlflow.set_tracking_uri(tracking_uri)
        self.mlflow.set_experiment(experiment_name)

    def log_run(
        self,
        run_name: str,
        model_name: str,
        params: dict[str, Any],
        metrics: dict[str, float],
        tags: dict[str, str],
        artifact_payload: dict[str, Any],
        artifacts_dir: str,
    ) -> str:
        ensure_dir(artifacts_dir)
        with self.mlflow.start_run(run_name=run_name) as run:
            self.mlflow.log_param("model_name", model_name)
            self.mlflow.log_params(params)
            self.mlflow.set_tags(tags)
            self.mlflow.log_metrics(metrics)
            artifact_path = Path(artifacts_dir) / f"{run_name}_summary.json"
            write_json(artifact_path, artifact_payload)
            self.mlflow.log_artifact(str(artifact_path))
            return run.info.run_id
