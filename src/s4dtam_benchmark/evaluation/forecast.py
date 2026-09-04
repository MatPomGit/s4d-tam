from __future__ import annotations

import numpy as np


def expected_calibration_error(
    target: np.ndarray, probability: np.ndarray, bins: int = 10
) -> float:
    target = np.asarray(target)
    probability = np.asarray(probability, dtype=float)
    _require_matching_arrays("occupancy", target, probability)
    target_flat, prob_flat = target.astype(float).ravel(), probability.ravel()
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        selected = (prob_flat >= lower) & (prob_flat < upper if upper < 1.0 else prob_flat <= upper)
        if np.any(selected):
            result += float(np.mean(selected)) * abs(
                float(np.mean(target_flat[selected])) - float(np.mean(prob_flat[selected]))
            )
    return result


def _require_matching_arrays(name: str, target: np.ndarray, prediction: np.ndarray) -> None:
    if target.shape != prediction.shape:
        raise ValueError(
            f"{name} target and prediction shapes must match exactly: "
            f"{target.shape} != {prediction.shape}"
        )
    if target.size == 0:
        raise ValueError(f"{name} target and prediction must not be empty")
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(prediction)):
        raise ValueError(f"{name} target and prediction must be finite")


def occupancy_metrics(
    target: np.ndarray, probability: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    target = np.asarray(target)
    probability = np.asarray(probability, dtype=float)
    _require_matching_arrays("occupancy", target, probability)
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("occupancy probabilities must be in [0, 1]")
    truth, pred = target.astype(bool), probability >= threshold
    tp = int(np.sum(truth & pred))
    fp = int(np.sum(~truth & pred))
    fn = int(np.sum(truth & ~pred))
    eps = 1e-12
    clipped = np.clip(probability, 1e-7, 1 - 1e-7)
    return {
        "iou": tp / (tp + fp + fn + eps),
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
        "f1": 2 * tp / (2 * tp + fp + fn + eps),
        "brier": float(np.mean((probability - target) ** 2)),
        "nll": float(-np.mean(target * np.log(clipped) + (1 - target) * np.log(1 - clipped))),
        "ece": expected_calibration_error(target, probability),
    }


def flow_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    _require_matching_arrays("flow", target, prediction)
    if target.ndim < 1 or target.shape[-1] == 0:
        raise ValueError("flow arrays must have a non-empty final vector axis")
    endpoint = np.linalg.norm(prediction - target, axis=-1)
    return {"epe_mean": float(np.mean(endpoint)), "epe_p95": float(np.quantile(endpoint, 0.95))}
