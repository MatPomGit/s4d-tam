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
    algorithm: str
    timestamps: np.ndarray
    estimated_positions: np.ndarray
    estimated_quaternions: np.ndarray | None = None
    pose_covariances: np.ndarray | None = None
    ood_scores: np.ndarray | None = None
    semantic_pred: np.ndarray | None = None
    occupancy_pred: dict[float, np.ndarray] = field(default_factory=dict)
    flow_pred: dict[float, np.ndarray] = field(default_factory=dict)
    risk_pred: np.ndarray | None = None
    latency_ms: np.ndarray | None = None
    resource: dict[str, float] = field(default_factory=dict)
    navigation: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        count = len(np.asarray(self.timestamps))
        if np.shape(self.estimated_positions) != (count, 3):
            raise ValueError("estimated_positions must have shape [N,3]")
        if self.pose_covariances is not None:
            covariance = np.asarray(self.pose_covariances, dtype=float)
            if covariance.shape != (count, 3, 3):
                raise ValueError("pose_covariances must have shape [N,3,3]")
            if not np.all(np.isfinite(covariance)) or np.any(
                np.linalg.eigvalsh(covariance) <= 0
            ):
                raise ValueError("pose_covariances must be finite and positive definite")
            self.pose_covariances = covariance
        if self.ood_scores is not None:
            scores = np.asarray(self.ood_scores, dtype=float)
            if scores.shape != (count,) or not np.all(np.isfinite(scores)):
                raise ValueError("ood_scores must be a finite vector with one value per estimate")
            self.ood_scores = scores


@dataclass(slots=True)
class RunContext:
    output_dir: Path
    seed: int
    config: dict[str, Any]
