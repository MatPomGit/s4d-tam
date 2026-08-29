from __future__ import annotations

import numpy as np


def paired_bootstrap(
    candidate: np.ndarray, baseline: np.ndarray, seed: int = 7, resamples: int = 10000
) -> dict[str, float]:
    """Bootstrap paired mission/sequence summaries, never individual frames."""
    if candidate.shape != baseline.shape or candidate.ndim != 1:
        raise ValueError("paired arrays must have identical [N] shape")
    differences = candidate - baseline
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(resamples, len(differences)))
    samples = differences[indices].mean(axis=1)
    sd = float(np.std(differences, ddof=1)) if len(differences) > 1 else float("nan")
    return {
        "mean_difference": float(np.mean(differences)),
        "median_difference": float(np.median(differences)),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "cohen_dz": float(np.mean(differences) / sd) if sd > 0 else float("nan"),
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min((len(p_values) - rank) * p_values[index], 1.0)
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()
