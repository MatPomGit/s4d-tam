from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np


MODALITIES = ("rgb", "thermal", "lidar", "imu", "gnss")


class AvailabilityState(IntEnum):
    """Stable integer values used by per-sample modality availability masks."""

    STREAM_ABSENT = 0
    SAMPLE_MISSING = 1
    QUALITY_REJECTED = 2
    AVAILABLE = 3


@dataclass(slots=True)
class SequenceData:
    dataset: str
    sequence_id: str
    timestamps: np.ndarray
    gt_positions: np.ndarray
    gt_quaternions: np.ndarray | None = None
    observations: np.ndarray | None = None
    semantic_observations: np.ndarray | None = None
    semantic_gt: np.ndarray | None = None
    occupancy_observations: np.ndarray | None = None
    occupancy_gt: dict[float, np.ndarray] = field(default_factory=dict)
    flow_gt: dict[float, np.ndarray] = field(default_factory=dict)
    risk_gt: np.ndarray | None = None
    navigation_gt: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    rgb: np.ndarray | None = None
    thermal: np.ndarray | None = None
    lidar: np.ndarray | None = None
    imu: np.ndarray | None = None
    gnss: np.ndarray | None = None
    calibration: dict[str, np.ndarray] = field(default_factory=dict)
    availability_masks: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize sequence arrays and validate synchronized sample contracts."""
        self.timestamps = np.asarray(self.timestamps, dtype=float)
        self.gt_positions = np.asarray(self.gt_positions, dtype=float)
        if self.timestamps.ndim != 1 or not np.all(np.isfinite(self.timestamps)):
            raise ValueError("timestamps must be a finite one-dimensional array")
        length = len(self.timestamps)
        if self.gt_positions.shape != (length, 3):
            raise ValueError("gt_positions must have shape (samples, 3)")
        if length > 1 and np.any(np.diff(self.timestamps) <= 0):
            raise ValueError("timestamps must be strictly increasing")
        if self.gt_quaternions is not None and np.shape(self.gt_quaternions) != (length, 4):
            raise ValueError("gt_quaternions must have shape (samples, 4)")

        unknown = set(self.availability_masks) - set(MODALITIES)
        if unknown:
            raise ValueError(f"unknown modalities in availability_masks: {sorted(unknown)}")
        valid_states = {
            AvailabilityState.STREAM_ABSENT,
            AvailabilityState.SAMPLE_MISSING,
            AvailabilityState.QUALITY_REJECTED,
            AvailabilityState.AVAILABLE,
        }
        for modality in MODALITIES:
            stream = getattr(self, modality)
            if stream is not None:
                stream = np.asarray(stream)
                if stream.ndim == 0 or stream.shape[0] != length:
                    raise ValueError(f"{modality} must have {length} samples on its first axis")
                setattr(self, modality, stream)
            default = (
                AvailabilityState.STREAM_ABSENT if stream is None else AvailabilityState.AVAILABLE
            )
            mask = np.asarray(
                self.availability_masks.get(modality, np.full(length, default)), dtype=np.int8
            )
            if mask.shape != (length,):
                raise ValueError(f"availability mask for {modality} must have shape ({length},)")
            if not set(mask.tolist()) <= valid_states:
                raise ValueError(f"availability mask for {modality} contains an invalid state")
            if stream is None and np.any(mask != AvailabilityState.STREAM_ABSENT):
                raise ValueError(f"absent {modality} stream must use STREAM_ABSENT throughout")
            if stream is not None and np.any(mask == AvailabilityState.STREAM_ABSENT):
                raise ValueError(f"present {modality} stream cannot use STREAM_ABSENT")
            self.availability_masks[modality] = mask

        for name, matrix in self.calibration.items():
            value = np.asarray(matrix, dtype=float)
            if value.ndim != 2 or not np.all(np.isfinite(value)):
                raise ValueError(f"calibration {name!r} must be a finite matrix")
            self.calibration[name] = value


@dataclass(slots=True)
class AlgorithmResult:
    """Normalized output contract shared by algorithms and evaluators.

    Per-sample arrays use the same leading dimension as ``timestamps``. Forecast
    mappings are keyed by horizon in seconds; occupancy predictions parameterize
    Bernoulli distributions while flow prediction/uncertainty pairs parameterize
    axis-independent Gaussian distributions. Forecast masks identify valid
    spatiotemporal regions and have the occupancy shape (equivalently, the flow
    shape without its final vector axis). Non-zero estimated quaternions are
    normalized to unit length. Pose
    covariance matrices are required to be symmetric positive definite because
    uncertainty evaluation uses their inverse and log determinant.
    """

    algorithm: str
    timestamps: np.ndarray
    estimated_positions: np.ndarray
    estimated_quaternions: np.ndarray | None = None
    pose_covariances: np.ndarray | None = None
    ood_scores: np.ndarray | None = None
    semantic_pred: np.ndarray | None = None
    occupancy_pred: dict[float, np.ndarray] = field(default_factory=dict)
    flow_pred: dict[float, np.ndarray] = field(default_factory=dict)
    occupancy_uncertainty: dict[float, np.ndarray] = field(default_factory=dict)
    flow_uncertainty: dict[float, np.ndarray] = field(default_factory=dict)
    forecast_observable_mask: dict[float, np.ndarray] = field(default_factory=dict)
    risk_pred: np.ndarray | None = None
    latency_ms: np.ndarray | None = None
    resource: dict[str, float] = field(default_factory=dict)
    navigation: dict[str, Any] = field(default_factory=dict)
    planned_trajectory: np.ndarray | None = None
    planner_cost_diagnostics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize arrays and reject malformed algorithm output early."""
        self.timestamps = np.asarray(self.timestamps, dtype=float)
        self.estimated_positions = np.asarray(self.estimated_positions, dtype=float)
        if self.timestamps.ndim != 1:
            raise ValueError("result timestamps must be a one-dimensional array")
        if self.timestamps.size == 0:
            raise ValueError("result timestamps must not be empty")
        if not np.all(np.isfinite(self.timestamps)):
            raise ValueError("result timestamps must be a finite one-dimensional array")
        if np.any(np.diff(self.timestamps) <= 0):
            raise ValueError("result timestamps must be strictly increasing")
        count = len(self.timestamps)
        if self.estimated_positions.shape != (count, 3) or not np.all(
            np.isfinite(self.estimated_positions)
        ):
            raise ValueError("estimated_positions must be a finite array with shape [N,3]")
        if self.estimated_quaternions is not None:
            quaternions = np.asarray(self.estimated_quaternions, dtype=float)
            if quaternions.shape != (count, 4):
                raise ValueError("estimated_quaternions must have shape [N,4]")
            if not np.all(np.isfinite(quaternions)):
                raise ValueError("estimated_quaternions must be finite")
            norms = np.linalg.norm(quaternions, axis=1)
            if np.any(norms == 0):
                raise ValueError("estimated_quaternions must not contain zero-norm quaternions")
            self.estimated_quaternions = quaternions / norms[:, None]
        if self.pose_covariances is not None:
            covariance = np.asarray(self.pose_covariances, dtype=float)
            if covariance.shape != (count, 3, 3):
                raise ValueError("pose_covariances must have shape [N,3,3]")
            if not np.all(np.isfinite(covariance)):
                raise ValueError("pose_covariances must be finite")
            if not np.allclose(covariance, np.swapaxes(covariance, 1, 2), atol=1e-10):
                raise ValueError("pose_covariances must be symmetric")
            if np.any(np.linalg.eigvalsh(covariance) <= 0):
                raise ValueError("pose_covariances must be positive definite")
            self.pose_covariances = covariance
        if self.ood_scores is not None:
            scores = np.asarray(self.ood_scores, dtype=float)
            if scores.shape != (count,) or not np.all(np.isfinite(scores)):
                raise ValueError("ood_scores must be a finite vector with one value per estimate")
            self.ood_scores = scores
        if self.semantic_pred is not None:
            semantic = np.asarray(self.semantic_pred)
            if semantic.ndim == 0 or semantic.shape[0] != count:
                raise ValueError("semantic_pred must have N samples on its first axis")
            if not np.issubdtype(semantic.dtype, np.number) or not np.all(np.isfinite(semantic)):
                raise ValueError("semantic_pred must contain finite numeric values")
            self.semantic_pred = semantic

        def normalize_forecasts(
            name: str, values: dict[float, np.ndarray], *, probability: bool = False
        ) -> dict[float, np.ndarray]:
            normalized: dict[float, np.ndarray] = {}
            for raw_horizon, raw_value in values.items():
                horizon = float(raw_horizon)
                if not np.isfinite(horizon) or horizon <= 0:
                    raise ValueError(f"{name} horizons must be positive and finite")
                value = np.asarray(raw_value, dtype=float)
                if value.ndim == 0 or value.shape[0] != count:
                    raise ValueError(f"{name}[{horizon:g}] must have N samples on its first axis")
                if not np.all(np.isfinite(value)):
                    raise ValueError(f"{name}[{horizon:g}] must be finite")
                if probability and np.any((value < 0) | (value > 1)):
                    raise ValueError(f"{name}[{horizon:g}] probabilities must be in [0, 1]")
                if name.endswith("uncertainty") and np.any(value < 0):
                    raise ValueError(f"{name}[{horizon:g}] must be non-negative")
                normalized[horizon] = value
            return normalized

        self.occupancy_pred = normalize_forecasts(
            "occupancy_pred", self.occupancy_pred, probability=True
        )
        self.flow_pred = normalize_forecasts("flow_pred", self.flow_pred)
        self.occupancy_uncertainty = normalize_forecasts(
            "occupancy_uncertainty", self.occupancy_uncertainty
        )
        self.flow_uncertainty = normalize_forecasts("flow_uncertainty", self.flow_uncertainty)
        for name, uncertainties, predictions in (
            ("occupancy_uncertainty", self.occupancy_uncertainty, self.occupancy_pred),
            ("flow_uncertainty", self.flow_uncertainty, self.flow_pred),
        ):
            for horizon in uncertainties.keys() & predictions.keys():
                if uncertainties[horizon].shape != predictions[horizon].shape:
                    raise ValueError(f"{name}[{horizon:g}] shape must match its prediction exactly")
        masks: dict[float, np.ndarray] = {}
        for raw_horizon, raw_mask in self.forecast_observable_mask.items():
            horizon = float(raw_horizon)
            if not np.isfinite(horizon) or horizon <= 0:
                raise ValueError("forecast_observable_mask horizons must be positive and finite")
            mask = np.asarray(raw_mask)
            if mask.dtype != np.bool_:
                raise ValueError(f"forecast_observable_mask[{horizon:g}] must be boolean")
            if mask.ndim == 0 or mask.shape[0] != count:
                raise ValueError(
                    f"forecast_observable_mask[{horizon:g}] must have N samples on its first axis"
                )
            allowed = []
            if horizon in self.occupancy_pred:
                allowed.append(self.occupancy_pred[horizon].shape)
            if horizon in self.flow_pred and self.flow_pred[horizon].ndim > 1:
                allowed.append(self.flow_pred[horizon].shape[:-1])
            if allowed and mask.shape not in allowed:
                raise ValueError(
                    f"forecast_observable_mask[{horizon:g}] shape {mask.shape} must be one of "
                    f"{allowed}"
                )
            masks[horizon] = mask
        self.forecast_observable_mask = masks

        if self.risk_pred is not None:
            risk = np.asarray(self.risk_pred, dtype=float)
            if risk.shape != (count,):
                raise ValueError("risk_pred must have shape [N]")
            if not np.all(np.isfinite(risk)) or np.any((risk < 0) | (risk > 1)):
                raise ValueError("risk_pred must contain finite probabilities in [0, 1]")
            self.risk_pred = risk
        if self.latency_ms is not None:
            latency = np.asarray(self.latency_ms, dtype=float)
            if latency.shape != (count,):
                raise ValueError("latency_ms must have shape [N]")
            if not np.all(np.isfinite(latency)) or np.any(latency < 0):
                raise ValueError("latency_ms must contain finite non-negative values")
            self.latency_ms = latency
        resources = {str(key): float(value) for key, value in self.resource.items()}
        if not all(np.isfinite(value) and value >= 0 for value in resources.values()):
            raise ValueError("resource values must be finite non-negative scalars")
        self.resource = resources
        if self.planned_trajectory is not None:
            trajectory = np.asarray(self.planned_trajectory, dtype=float)
            if (
                trajectory.ndim != 2
                or trajectory.shape[1] != 3
                or not np.all(np.isfinite(trajectory))
            ):
                raise ValueError("planned_trajectory must be a finite array with shape [M,3]")
            self.planned_trajectory = trajectory
        diagnostics = {
            str(key): float(value) for key, value in self.planner_cost_diagnostics.items()
        }
        if not all(np.isfinite(value) for value in diagnostics.values()):
            raise ValueError("planner cost diagnostics must be finite")
        self.planner_cost_diagnostics = diagnostics


@dataclass(slots=True)
class RunContext:
    output_dir: Path
    seed: int
    config: dict[str, Any]
