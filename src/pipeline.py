from typing import Any

from src.data_ingestion import DataIngestion
from src.feature_engineering import FeatureEngineer
from src.mlflow_tracking import MLflowTracker
from src.model_evaluation import evaluate_model
from src.model_registry import ModelRegistry
from src.model_training import ModelTrainer
from src.preprocessing import Preprocessor
from src.roi_extraction import ROIExtractor


def run_pipeline(config: dict[str, Any], global_params: dict[str, Any]) -> dict[str, Any]:
    run_name = config.get("run", {}).get("name", "gesture_run")
    model_name = config.get("model", {}).get("name", "mediapipe_hand_landmarks")
    model_params = config.get("model", {}).get("params", {})

    ingestion = DataIngestion(config.get("data", {}))
    preprocessor = Preprocessor(config.get("preprocessing", {}))
    roi_extractor = ROIExtractor(config.get("roi", {}))
    feature_builder = FeatureEngineer(config.get("features", {}))

    records = ingestion.load()
    processed = preprocessor.transform(records)
    rois = roi_extractor.extract(processed)
    features = feature_builder.build(rois)

    registry = ModelRegistry()
    model = registry.create(model_name, model_params)
    trainer = ModelTrainer(config.get("training", {}))
    train_output = trainer.train(model, features)
    metrics = evaluate_model(train_output, features)

    tracker = MLflowTracker(
        tracking_uri=global_params.get("default_tracking_uri", "./mlruns"),
        experiment_name=global_params.get("experiment_name", "gesture_vision_experiments"),
    )

    result = {
        "run_name": run_name,
        "model_name": model_name,
        "train_output": train_output,
        "metrics": metrics,
        "config": config,
    }
    run_id = tracker.log_run(
        run_name=run_name,
        model_name=model_name,
        params=model_params,
        metrics=metrics,
        tags={"pipeline": "gesture_vision", "family": model.family},
        artifact_payload=result,
        artifacts_dir=global_params.get("artifacts_dir", "artifacts"),
    )
    result["mlflow_run_id"] = run_id
    return result
