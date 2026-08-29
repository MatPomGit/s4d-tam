from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from s4dtam_benchmark.contracts import AvailabilityState


@dataclass(frozen=True, slots=True)
class EncodedObservation:
    modality: str
    timestamp: float
    features: np.ndarray
    state: int = AvailabilityState.AVAILABLE
    confidence: float = 1.0

    def __post_init__(self) -> None:
        features = np.asarray(self.features, dtype=float)
        if features.ndim != 1 or not np.all(np.isfinite(features)):
            raise ValueError("encoded features must be a finite vector")
        if self.state not in range(4):
            raise ValueError("invalid availability state")
        object.__setattr__(self, "features", features)


class ModalityEncoder(ABC):
    modality: str

    def __init__(self, output_dim: int = 3, scale: float = 1.0):
        if output_dim < 1:
            raise ValueError("output_dim must be positive")
        self.output_dim = output_dim
        self.scale = float(scale)

    @abstractmethod
    def encode(self, sample: np.ndarray, timestamp: float) -> EncodedObservation:
        """Encode one synchronized sample."""

    def _encode_numeric(self, sample: np.ndarray, timestamp: float) -> EncodedObservation:
        values = np.asarray(sample, dtype=float).reshape(-1)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError(f"{self.modality} sample must contain finite values")
        # Fixed analytic projection: deterministic across processes and Python versions.
        indices = np.arange(1, values.size + 1, dtype=float)
        rows = np.arange(1, self.output_dim + 1, dtype=float)[:, None]
        projection = np.cos(rows * indices * 0.173) / np.sqrt(values.size)
        features = self.scale * (projection @ values)
        return EncodedObservation(self.modality, float(timestamp), features)
