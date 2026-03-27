from src.pipeline import run_pipeline


class _RunInfo:
    def __init__(self) -> None:
        self.run_id = "test-run-id"


class _RunObj:
    def __init__(self) -> None:
        self.info = _RunInfo()


class _RunContext:
    def __enter__(self) -> _RunObj:
        return _RunObj()

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeMLflow:
    def set_tracking_uri(self, uri: str) -> None:
        self.uri = uri

    def set_experiment(self, name: str) -> None:
        self.experiment = name

    def start_run(self, run_name: str):
        self.run_name = run_name
        return _RunContext()

    def log_param(self, key: str, value):
        return None

    def log_params(self, params):
        return None

    def set_tags(self, tags):
        return None

    def log_metrics(self, metrics):
        return None

    def log_artifact(self, path: str):
        return None


def test_pipeline_returns_expected_keys(monkeypatch) -> None:
    import importlib

    monkeypatch.setattr(importlib, "import_module", lambda name: _FakeMLflow())
    config = {
        "run": {"name": "test_run"},
        "data": {"source": {"sample_count": 4}},
        "model": {"name": "video_classifier", "params": {"clip_length": 8}},
    }
    globals_cfg = {
        "default_tracking_uri": "./mlruns",
        "experiment_name": "test_experiment",
        "artifacts_dir": "artifacts",
    }
    result = run_pipeline(config, globals_cfg)
    assert "metrics" in result
    assert "mlflow_run_id" in result
    assert result["model_name"] == "video_classifier"
