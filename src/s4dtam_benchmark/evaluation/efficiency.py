from __future__ import annotations

import numpy as np


def efficiency_metrics(
    latency_ms: np.ndarray | None,
    resource: dict[str, float],
    planner_cost: dict[str, float] | None = None,
) -> dict[str, float]:
    """Summarize runtime resources, latency, and optional planning resources."""
    metrics = {f"efficiency/{key}": float(value) for key, value in resource.items()}
    if latency_ms is not None and len(latency_ms):
        metrics.update(
            {
                "efficiency/latency_median_ms": float(np.median(latency_ms)),
                "efficiency/latency_p90_ms": float(np.quantile(latency_ms, 0.90)),
                "efficiency/latency_p95_ms": float(np.quantile(latency_ms, 0.95)),
                "efficiency/latency_p99_ms": float(np.quantile(latency_ms, 0.99)),
                "efficiency/fps": 1000.0 / max(float(np.mean(latency_ms)), 1e-12),
            }
        )
    if planner_cost:
        for key in ("energy", "time", "energy_remaining_wh"):
            if key in planner_cost:
                metrics[f"efficiency/planner_{key}"] = float(planner_cost[key])
    return metrics
