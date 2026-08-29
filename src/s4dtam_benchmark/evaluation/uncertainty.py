from __future__ import annotations

import numpy as np
from scipy.stats import chi2, rankdata


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
        "risk/false_alarm_rate": float(np.mean(predicted[negative]))
        if negative.any()
        else float("nan"),
        "risk/miss_rate": float(np.mean(~predicted[positive])) if positive.any() else float("nan"),
    }


def ood_metrics(target: np.ndarray, score: np.ndarray) -> dict[str, float]:
    """Compute threshold-free OOD AUROC and average precision.

    Args:
        target: Binary labels where one denotes an OOD sample.
        score: Finite anomaly scores where larger values are more OOD-like.

    Returns:
        AUROC and area under the precision-recall curve (average precision).

    Raises:
        ValueError: If inputs have different shapes, are empty/non-finite, or labels
            are not binary.
    """
    truth = np.asarray(target, dtype=int).ravel()
    values = np.asarray(score, dtype=float).ravel()
    if truth.shape != values.shape or truth.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("OOD labels and finite scores must have matching shape")
    if set(truth.tolist()) - {0, 1}:
        raise ValueError("OOD labels must be binary")
    positive, negative = truth == 1, truth == 0
    if not positive.any() or not negative.any():
        return {"ood/auroc": float("nan"), "ood/auprc": float("nan")}
    ranks = rankdata(values, method="average")
    auroc = (ranks[positive].sum() - positive.sum() * (positive.sum() + 1) / 2) / (
        positive.sum() * negative.sum()
    )
    order = np.argsort(-values, kind="mergesort")
    sorted_truth = truth[order]
    sorted_scores = values[order]
    threshold_ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(values) - 1]
    true_positives = np.cumsum(sorted_truth)[threshold_ends]
    recall = true_positives / positive.sum()
    precision = true_positives / (threshold_ends + 1)
    auprc = np.sum(np.diff(np.r_[0.0, recall]) * precision)
    return {"ood/auroc": float(auroc), "ood/auprc": float(auprc)}


def selective_risk_metrics(
    reference: np.ndarray, estimate: np.ndarray, uncertainty: np.ndarray
) -> dict[str, float]:
    """Compute selective risk at fixed coverages and exact discrete AURC.

    Args:
        reference: Ground-truth positions with shape ``[N, 3]``.
        estimate: Estimated positions with shape ``[N, 3]``.
        uncertainty: One uncertainty score per estimate; lower means more confident.

    Returns:
        Mean position error at fixed, predeclared coverages and AURC over all prefixes.
    """
    errors = np.linalg.norm(np.asarray(reference) - np.asarray(estimate), axis=1)
    scores = np.asarray(uncertainty, dtype=float).ravel()
    if scores.shape != errors.shape or not len(scores) or not np.all(np.isfinite(scores)):
        raise ValueError("uncertainty score must have one value per pose")
    order = np.argsort(scores, kind="stable")
    coverages = np.asarray([0.25, 0.5, 0.75, 1.0])
    cumulative_risk = np.cumsum(errors[order]) / np.arange(1, len(order) + 1)
    risks = [float(cumulative_risk[max(0, int(np.ceil(len(order) * c)) - 1)]) for c in coverages]
    result = {f"selective/risk_at_{int(c * 100)}pct": r for c, r in zip(coverages, risks)}
    result["selective/aurc"] = float(np.mean(cumulative_risk))
    return result


def pose_calibration_metrics(
    reference: np.ndarray, estimate: np.ndarray, covariance: np.ndarray
) -> dict[str, float]:
    """Compute fixed-level pose coverage and expected calibration error.

    Args:
        reference: Ground-truth positions with shape ``[N, 3]``.
        estimate: Estimated positions with shape ``[N, 3]``.
        covariance: Positive-definite position covariances with shape ``[N, 3, 3]``.

    Returns:
        Empirical coverage at predeclared confidence levels and mean absolute ECE.
    """
    errors = np.asarray(reference) - np.asarray(estimate)
    nees = np.asarray([e @ np.linalg.pinv(c) @ e for e, c in zip(errors, covariance, strict=True)])
    levels = np.asarray([0.5, 0.8, 0.9, 0.95])
    # Levels are fixed before evaluation; no threshold is selected from test outcomes.
    thresholds = chi2.ppf(levels, df=3)
    observed = np.asarray([np.mean(nees <= threshold) for threshold in thresholds])
    result = {
        f"calibration/coverage_{int(level * 100)}pct": float(value)
        for level, value in zip(levels, observed)
    }
    result["calibration/ece"] = float(np.mean(np.abs(observed - levels)))
    return result
