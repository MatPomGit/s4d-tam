"""Held-out uncertainty and input-distribution calibration primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class CalibrationParameters:
    """Serializable parameters learned exclusively from a calibration split.

    Args:
        feature_mean: Per-feature in-distribution mean.
        feature_scale: Per-feature in-distribution standard deviation.
        covariance_scale: Positive multiplier applied to predicted covariance.
        sample_count: Number of calibration samples used by the fit.
    """

    feature_mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    feature_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    covariance_scale: float = 1.0
    sample_count: int = 0

    def __post_init__(self) -> None:
        mean = np.asarray(self.feature_mean, dtype=float)
        scale = np.asarray(self.feature_scale, dtype=float)
        if mean.shape != (3,) or scale.shape != (3,):
            raise ValueError("calibration feature statistics must contain three values")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
            raise ValueError("calibration feature statistics must be finite")
        if np.any(scale <= 0):
            raise ValueError("calibration feature scales must be positive")
        if not np.isfinite(self.covariance_scale) or self.covariance_scale <= 0:
            raise ValueError("calibration covariance scale must be finite and positive")
        if self.sample_count < 0:
            raise ValueError("calibration sample_count must be non-negative")

    def ood_score(self, features: np.ndarray) -> float:
        """Return standardized input energy, where larger values are more OOD-like."""
        vector = np.asarray(features, dtype=float)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError("OOD features must be a finite three-dimensional vector")
        standardized = (vector - self.feature_mean_array) / self.feature_scale_array
        return float(np.mean(np.square(standardized)))

    @property
    def feature_mean_array(self) -> np.ndarray:
        """Return the feature mean as a NumPy vector."""
        return np.asarray(self.feature_mean, dtype=float)

    @property
    def feature_scale_array(self) -> np.ndarray:
        """Return the feature scale as a NumPy vector."""
        return np.asarray(self.feature_scale, dtype=float)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation for artifacts and manifests."""
        return {
            "feature_mean": list(self.feature_mean),
            "feature_scale": list(self.feature_scale),
            "covariance_scale": self.covariance_scale,
            "sample_count": self.sample_count,
        }


def fit_calibration(
    features: np.ndarray,
    errors: np.ndarray,
    predicted_covariances: np.ndarray,
    *,
    minimum_feature_scale: float = 1e-6,
    minimum_covariance_scale: float = 1e-6,
) -> CalibrationParameters:
    """Fit OOD standardization and covariance scaling on held-out samples.

    The covariance multiplier is the mean normalized estimation error divided by
    the three pose dimensions. It therefore corrects the scale of the supplied
    covariance predictions rather than replacing their physical units.

    Args:
        features: Input features with shape ``[N, 3]``.
        errors: Position residuals with shape ``[N, 3]``.
        predicted_covariances: Uncalibrated covariances with shape ``[N, 3, 3]``.
        minimum_feature_scale: Lower numerical bound for feature standard deviations.
        minimum_covariance_scale: Lower numerical bound for the covariance multiplier.

    Returns:
        Validated, serializable calibration parameters.

    Raises:
        ValueError: If inputs are empty, non-finite, incorrectly shaped, or bounds
            are not strictly positive.
    """
    feature_array = np.asarray(features, dtype=float)
    error_array = np.asarray(errors, dtype=float)
    covariance_array = np.asarray(predicted_covariances, dtype=float)
    sample_count = len(feature_array)
    if sample_count == 0:
        raise ValueError("calibration requires at least one usable sample")
    if feature_array.shape != (sample_count, 3) or error_array.shape != (sample_count, 3):
        raise ValueError("calibration features and errors must have shape [N,3]")
    if covariance_array.shape != (sample_count, 3, 3):
        raise ValueError("calibration covariances must have shape [N,3,3]")
    if not all(
        np.all(np.isfinite(item)) for item in (feature_array, error_array, covariance_array)
    ):
        raise ValueError("calibration inputs must be finite")
    if not np.allclose(covariance_array, np.swapaxes(covariance_array, 1, 2), atol=1e-10):
        raise ValueError("calibration covariances must be symmetric")
    if np.any(np.linalg.eigvalsh(covariance_array) <= 0):
        raise ValueError("calibration covariances must be positive definite")
    if minimum_feature_scale <= 0 or minimum_covariance_scale <= 0:
        raise ValueError("calibration numerical bounds must be positive")

    normalized_errors = np.asarray(
        [
            error @ np.linalg.solve(covariance, error)
            for error, covariance in zip(error_array, covariance_array, strict=True)
        ]
    )
    covariance_scale = max(float(np.mean(normalized_errors) / 3.0), minimum_covariance_scale)
    feature_scale = np.maximum(np.std(feature_array, axis=0), minimum_feature_scale)
    feature_mean = np.mean(feature_array, axis=0)
    return CalibrationParameters(
        feature_mean=(float(feature_mean[0]), float(feature_mean[1]), float(feature_mean[2])),
        feature_scale=(float(feature_scale[0]), float(feature_scale[1]), float(feature_scale[2])),
        covariance_scale=covariance_scale,
        sample_count=sample_count,
    )
