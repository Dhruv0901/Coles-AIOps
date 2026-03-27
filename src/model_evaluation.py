from typing import Any


def evaluate_model(train_output: dict[str, Any], samples: list[dict[str, Any]]) -> dict[str, float]:
    seen = max(int(train_output.get("seen_samples", 1)), 1)
    quality = float(train_output.get("quality_signal", 0.5))
    gesture_accuracy = round(min(0.99, quality), 4)
    flick_f1 = round(min(0.99, quality - 0.05), 4)
    badge_roi_iou = round(min(0.99, 0.45 + seen / 200.0), 4)
    return {
        "gesture_accuracy": gesture_accuracy,
        "flick_f1": max(flick_f1, 0.0),
        "badge_roi_iou": badge_roi_iou,
    }


def rank_experiments(results: list[dict[str, Any]], metric: str = "gesture_accuracy") -> list[dict[str, Any]]:
    return sorted(results, key=lambda x: x.get("metrics", {}).get(metric, 0.0), reverse=True)
