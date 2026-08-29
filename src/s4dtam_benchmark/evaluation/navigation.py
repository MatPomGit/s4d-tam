from __future__ import annotations

from typing import Any


def navigation_metrics(trace: dict[str, Any], ground_truth: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key in ("mission_success", "collision_count", "near_miss_count", "min_clearance_m", "energy_wh"):
        if key in trace:
            metrics[f"navigation/{key}"] = float(trace[key])
    length = float(trace.get("path_length_m", 0.0))
    shortest = float(ground_truth.get("shortest_path_m", 0.0))
    if length > 0:
        metrics["navigation/path_length_m"] = length
        metrics["navigation/collisions_per_km"] = 1000.0 * float(trace.get("collision_count", 0)) / length
    if length > 0 and shortest > 0:
        metrics["navigation/path_efficiency"] = shortest / length
    return metrics
