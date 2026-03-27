from pathlib import Path
from typing import Any

from src.pipeline import run_pipeline
from src.utils import ensure_dir, read_yaml, utc_timestamp, write_json


def run_experiment(config_path: str, global_params: dict[str, Any] | None = None) -> dict[str, Any]:
    global_params = global_params or {}
    cfg = read_yaml(config_path)
    result = run_pipeline(cfg, global_params)
    output_dir = ensure_dir(global_params.get("output_dir", "experiments"))
    file_name = f"{result['run_name']}_{utc_timestamp()}.json"
    out_path = Path(output_dir) / file_name
    write_json(out_path, result)
    return result


def run_many(config_paths: list[str], global_params: dict[str, Any]) -> list[dict[str, Any]]:
    results = []
    for path in config_paths:
        results.append(run_experiment(path, global_params))
    summary_path = Path(global_params.get("output_dir", "experiments")) / "comparison_summary.json"
    ranked = sorted(results, key=lambda x: x["metrics"].get("gesture_accuracy", 0.0), reverse=True)
    write_json(summary_path, {"results": ranked})
    return ranked
