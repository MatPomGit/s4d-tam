from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from s4dtam_benchmark.contracts import AvailabilityState


@dataclass(frozen=True, slots=True)
class EncodedObservation:
    modality: str
    timestamp: float
    features: np.ndarray
    state: AvailabilityState = AvailabilityState.AVAILABLE
    confidence: float = 1.0

    def __post_init__(self) -> None:
        features = np.array(self.features, dtype=np.float64, copy=True)
        if features.ndim != 1 or not np.all(np.isfinite(features)):
            raise ValueError("encoded features must be a finite vector")
        if not np.isfinite(self.timestamp):
            raise ValueError("timestamp must be finite")
        try:
            state = AvailabilityState(self.state)
        except ValueError as error:
            raise ValueError("invalid availability state") from error
        if not 0.0 <= self.confidence <= 1.0 or not np.isfinite(self.confidence):
            raise ValueError("confidence must be finite and in [0, 1]")
        if state == AvailabilityState.AVAILABLE and self.confidence == 0.0:
            raise ValueError("available observations must have positive confidence")
        features.flags.writeable = False
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "state", state)


class ModalityEncoder(ABC):
    """Stateless, deterministic interface implemented by every sensor encoder."""

    modality: ClassVar[str]

    def __init__(self, output_dim: int = 3, scale: float = 1.0):
        if output_dim < 1:
            raise ValueError("output_dim must be positive")
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("scale must be finite and positive")
        self.output_dim = int(output_dim)
        self.scale = float(scale)

    @abstractmethod
    def encode(self, sample: np.ndarray, timestamp: float) -> EncodedObservation:
        """Encode one synchronized sample."""

    def _validated_sample(self, sample: np.ndarray) -> np.ndarray:
        values = np.asarray(sample, dtype=np.float64)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError(f"{self.modality} sample must contain finite values")
        return values

    def _project(self, descriptor: np.ndarray, timestamp: float) -> EncodedObservation:
        """Project a compact sensor-specific descriptor without learned weights.

        This CPU reference deliberately uses an analytic projection. Its phase is
        modality-specific, so equal numeric inputs from unrelated sensors do not
        collapse to the same representation.
        """
        values = np.asarray(descriptor, dtype=np.float64).reshape(-1)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError(f"{self.modality} descriptor must be a finite vector")
        modality_phase = sum((index + 1) * ord(char) for index, char in enumerate(self.modality))
        columns = np.arange(1, values.size + 1, dtype=np.float64)[None, :]
        rows = np.arange(1, self.output_dim + 1, dtype=np.float64)[:, None]
        weights = np.sin(rows * columns * 0.173 + modality_phase * 0.001)
        weights /= np.linalg.norm(weights, axis=1, keepdims=True).clip(min=1e-12)
        features = self.scale * (weights @ values)
        return EncodedObservation(self.modality, float(timestamp), features)


def distribution_descriptor(values: np.ndarray) -> np.ndarray:
    """Return robust first-order statistics for a scalar sensor field."""
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    return np.array(
        [
            np.mean(flat),
            np.std(flat),
            np.min(flat),
            np.quantile(flat, 0.25),
            np.median(flat),
            np.quantile(flat, 0.75),
            np.max(flat),
        ]
    )
