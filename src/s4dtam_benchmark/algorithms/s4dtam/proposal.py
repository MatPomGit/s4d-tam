"""Creation of token candidates from encoder output."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class TokenCandidate:
    """An observation that may update an existing token or create a new one."""

    position: np.ndarray
    features: np.ndarray
    semantic_logits: np.ndarray
    covariance: np.ndarray
    timestamp: float
    sensory_descriptor: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    source_index: int | None = None
    confidence: float = 1.0

    @property
    def uncertainty(self) -> float:
        return float(np.trace(self.covariance))

    def __post_init__(self) -> None:
        self.position = _finite_vector(self.position, "position", length=3)
        self.features = _finite_vector(self.features, "features")
        self.semantic_logits = _finite_vector(self.semantic_logits, "semantic_logits")
        self.sensory_descriptor = _finite_vector(
            self.sensory_descriptor, "sensory_descriptor", allow_empty=True
        )
        self.covariance = np.asarray(self.covariance, dtype=float)
        if self.covariance.shape != (3, 3) or not np.all(np.isfinite(self.covariance)):
            raise ValueError("covariance must be a finite 3x3 matrix")
        if not np.allclose(self.covariance, self.covariance.T):
            raise ValueError("covariance must be symmetric")
        if np.min(np.linalg.eigvalsh(self.covariance)) < 0:
            raise ValueError("covariance must be positive semidefinite")
        if not np.isfinite(self.timestamp):
            raise ValueError("timestamp must be finite")
        if not np.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between zero and one")


def _finite_vector(
    value: np.ndarray, name: str, *, length: int | None = None, allow_empty: bool = False
) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or (length is not None and len(result) != length):
        expected = f" with length {length}" if length is not None else ""
        raise ValueError(f"{name} must be a vector{expected}")
    if not allow_empty and result.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result.copy()


class TokenProposalModule:
    """Convert encoded observations into validated, uncertainty-aware candidates."""

    def __init__(self, semantic_classes: int = 8, default_variance: float = 0.04):
        if semantic_classes <= 0:
            raise ValueError("semantic_classes must be positive")
        if not np.isfinite(default_variance) or default_variance <= 0:
            raise ValueError("default_variance must be finite and positive")
        self.semantic_classes = semantic_classes
        self.default_variance = default_variance

    def propose(
        self,
        encoded: np.ndarray,
        timestamp: float,
        *,
        positions: np.ndarray | None = None,
        semantic_logits: np.ndarray | None = None,
        uncertainty: np.ndarray | float | None = None,
        sensory_descriptors: np.ndarray | None = None,
        proposal_confidence: np.ndarray | float | None = None,
    ) -> list[TokenCandidate]:
        features = np.atleast_2d(np.asarray(encoded, dtype=float))
        locations = features[:, :3] if positions is None else np.atleast_2d(positions).astype(float)
        if len(locations) != len(features) or locations.shape[1] != 3:
            raise ValueError("positions must have shape (candidates, 3)")
        semantics = self._semantics(semantic_logits, len(features))
        descriptors = (
            features if sensory_descriptors is None else np.atleast_2d(sensory_descriptors)
        )
        if len(descriptors) != len(features):
            raise ValueError("sensory_descriptors must have one row per candidate")
        covariances = self._covariances(uncertainty, len(features))
        confidence = np.broadcast_to(
            1.0 if proposal_confidence is None else proposal_confidence, (len(features),)
        ).astype(float)
        return [
            TokenCandidate(
                position=locations[i].copy(),
                features=features[i].copy(),
                semantic_logits=semantics[i].copy(),
                covariance=covariances[i].copy(),
                timestamp=float(timestamp),
                sensory_descriptor=np.asarray(descriptors[i], dtype=float).copy(),
                source_index=i,
                confidence=float(confidence[i]),
            )
            for i in range(len(features))
        ]

    def _semantics(self, values: np.ndarray | None, count: int) -> np.ndarray:
        if values is None:
            return np.zeros((count, self.semantic_classes), dtype=float)
        raw = np.asarray(values)
        if raw.ndim == 1 and len(raw) == count and np.issubdtype(raw.dtype, np.integer):
            if np.any(raw < 0) or np.any(raw >= self.semantic_classes):
                raise ValueError("semantic class index is out of range")
            result = np.eye(self.semantic_classes)[raw.astype(int)]
        else:
            result = np.asarray(values, dtype=float)
        result = np.atleast_2d(result)
        if result.shape != (count, self.semantic_classes):
            raise ValueError("semantic_logits have an invalid shape")
        if not np.all(np.isfinite(result)):
            raise ValueError("semantic_logits must be finite")
        return result

    def _covariances(self, value: np.ndarray | float | None, count: int) -> np.ndarray:
        if value is None:
            return np.repeat((np.eye(3) * self.default_variance)[None, :, :], count, axis=0)
        array = np.asarray(value, dtype=float)
        if array.ndim == 0:
            if not np.isfinite(array) or array <= 0:
                raise ValueError("uncertainty variance must be finite and positive")
            return np.repeat((np.eye(3) * float(array))[None, :, :], count, axis=0)
        if array.shape == (3, 3):
            return np.repeat(array[None, :, :], count, axis=0)
        if array.shape != (count, 3, 3):
            raise ValueError(
                "uncertainty must be a variance, 3x3 matrix, or one matrix per candidate"
            )
        return array
