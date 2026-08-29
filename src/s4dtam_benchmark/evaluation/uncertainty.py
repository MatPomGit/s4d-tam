from __future__ import annotations

import numpy as np


def pose_uncertainty_metrics(
    reference: np.ndarray, estimate: np.ndarray, covariance: np.ndarray
) -> dict[str, float]:
    if covariance.shape != (len(reference), 3, 3):
        raise ValueError("pose covariance must have shape [N,3,3]")
    errors = reference - estimate
    nees, nll = [], []
    for error, matrix in zip(errors, covariance, strict=True):
        stable = matrix + np.eye(3) * 1e-9
        inverse = np.linalg.inv(stable)
        value = float(error.T @ inverse @ error)
        nees.append(value)
        _, logdet = np.linalg.slogdet(stable)
        nll.append(0.5 * (3 * np.log(2 * np.pi) + logdet + value))
    nees_array = np.asarray(nees)
    return {
        "uncertainty/pose_nees_mean": float(np.mean(nees_array)),
        "uncertainty/pose_nees_median": float(np.median(nees_array)),
        "uncertainty/pose_95pct_coverage": float(np.mean(nees_array <= 7.8147279)),
        "uncertainty/pose_nll_mean": float(np.mean(nll)),
    }


def binary_risk_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    truth = target.astype(int).ravel()
    score = np.asarray(probability, dtype=float).ravel()
    if truth.shape != score.shape:
        raise ValueError("risk target and prediction must have matching shape")
    clipped = np.clip(score, 1e-7, 1 - 1e-7)
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    positive, negative = truth == 1, truth == 0
    if positive.any() and negative.any():
        auc = (ranks[positive].sum() - positive.sum() * (positive.sum() + 1) / 2) / (
            positive.sum() * negative.sum()
        )
    else:
        auc = float("nan")
    predicted = score >= 0.5
    return {
        "risk/brier": float(np.mean((score - truth) ** 2)),
        "risk/nll": float(-np.mean(truth * np.log(clipped) + (1 - truth) * np.log(1 - clipped))),
        "risk/auroc": float(auc),
        "risk/false_alarm_rate": float(np.mean(predicted[negative])) if negative.any() else float("nan"),
        "risk/miss_rate": float(np.mean(~predicted[positive])) if positive.any() else float("nan"),
    }


def ood_metrics(target: np.ndarray, score: np.ndarray) -> dict[str, float]:
    """Threshold-free OOD ranking metrics (larger scores mean more anomalous)."""
    truth = np.asarray(target, dtype=int).ravel()
    values = np.asarray(score, dtype=float).ravel()
    if truth.shape != values.shape or not np.all(np.isfinite(values)):
        raise ValueError("OOD labels and finite scores must have matching shape")
    if set(truth.tolist()) - {0, 1}:
        raise ValueError("OOD labels must be binary")
    positive, negative = truth == 1, truth == 0
    if not positive.any() or not negative.any():
        return {"ood/auroc": float("nan"), "ood/auprc": float("nan")}
    # Pairwise definition handles tied scores without a test-derived threshold.
    comparisons = values[positive, None] - values[negative][None, :]
    auroc = np.mean((comparisons > 0) + 0.5 * (comparisons == 0))
    order = np.argsort(-values, kind="stable")
    sorted_truth = truth[order]
    precision = np.cumsum(sorted_truth) / np.arange(1, len(truth) + 1)
    auprc = np.sum(precision * sorted_truth) / positive.sum()
    return {"ood/auroc": float(auroc), "ood/auprc": float(auprc)}


def selective_risk_metrics(
    reference: np.ndarray, estimate: np.ndarray, uncertainty: np.ndarray
) -> dict[str, float]:
    """Risk/coverage summary using a fixed coverage grid, never tuned on test labels."""
    errors = np.linalg.norm(np.asarray(reference) - np.asarray(estimate), axis=1)
    scores = np.asarray(uncertainty, dtype=float).ravel()
    if scores.shape != errors.shape:
        raise ValueError("uncertainty score must have one value per pose")
    order = np.argsort(scores, kind="stable")
    coverages = np.asarray([0.25, 0.5, 0.75, 1.0])
    risks = [float(np.mean(errors[order[: max(1, int(np.ceil(len(order) * c)))]])) for c in coverages]
    result = {f"selective/risk_at_{int(c * 100)}pct": r for c, r in zip(coverages, risks)}
    result["selective/aurc"] = float(np.trapezoid(risks, coverages))
    return result


def pose_calibration_metrics(
    reference: np.ndarray, estimate: np.ndarray, covariance: np.ndarray
) -> dict[str, float]:
    """Fixed-bin calibration curve and mean absolute coverage error."""
    errors = np.asarray(reference) - np.asarray(estimate)
    nees = np.asarray([e @ np.linalg.pinv(c) @ e for e, c in zip(errors, covariance, strict=True)])
    # Predeclared chi-square(3) quantiles: no calibration choices use test data.
    levels = np.asarray([0.5, 0.8, 0.9, 0.95])
    thresholds = np.asarray([2.365974, 4.641628, 6.251389, 7.814728])
    observed = np.asarray([np.mean(nees <= threshold) for threshold in thresholds])
    result = {f"calibration/coverage_{int(level * 100)}pct": float(value)
              for level, value in zip(levels, observed)}
    result["calibration/ece"] = float(np.mean(np.abs(observed - levels)))
    return result
