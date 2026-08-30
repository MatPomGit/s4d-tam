from __future__ import annotations

from typing import Any

import numpy as np


def navigation_metrics(
    trace: dict[str, Any],
    ground_truth: dict[str, Any],
    planned_trajectory: Any = None,
    planner_cost: dict[str, float] | None = None,
) -> dict[str, float]:
    """Compute closed-loop and optional planner-specific navigation metrics.

    Planner arguments are optional so results from localization-only algorithms
    continue to use the original navigation metric contract.
    """
    metrics: dict[str, float] = {}
    for key in (
        "mission_success",
        "collision_count",
        "near_miss_count",
        "min_clearance_m",
        "energy_wh",
    ):
        if key in trace:
            metrics[f"navigation/{key}"] = float(trace[key])
    length = float(trace.get("path_length_m", 0.0))
    shortest = float(ground_truth.get("shortest_path_m", 0.0))
    if length > 0:
        metrics["navigation/path_length_m"] = length
        metrics["navigation/collisions_per_km"] = (
            1000.0 * float(trace.get("collision_count", 0)) / length
        )
    if length > 0 and shortest > 0:
        metrics["navigation/path_efficiency"] = shortest / length
    if planned_trajectory is not None:
        trajectory = np.asarray(planned_trajectory, dtype=float)
        metrics["navigation/planned_path_length_m"] = (
            float(np.linalg.norm(np.diff(trajectory, axis=0), axis=1).sum())
            if len(trajectory) > 1
            else 0.0
        )
    if planner_cost:
        for key, value in planner_cost.items():
            metrics[f"navigation/planner_cost/{key}"] = float(value)
    return metrics
