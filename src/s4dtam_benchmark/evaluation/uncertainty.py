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
