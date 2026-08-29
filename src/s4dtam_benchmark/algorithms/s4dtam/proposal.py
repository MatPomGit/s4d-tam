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


class TokenProposalModule:
    """Convert encoded observations into validated, uncertainty-aware candidates."""

    def __init__(self, semantic_classes: int = 8, default_variance: float = 0.04):
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
        result = np.asarray(values, dtype=float)
        if result.ndim == 1 and len(result) == count and np.issubdtype(result.dtype, np.integer):
            result = np.eye(self.semantic_classes)[result.astype(int)]
        result = np.atleast_2d(result)
        if result.shape != (count, self.semantic_classes):
            raise ValueError("semantic_logits have an invalid shape")
        return result

    def _covariances(self, value: np.ndarray | float | None, count: int) -> np.ndarray:
        if value is None:
            return np.repeat((np.eye(3) * self.default_variance)[None, :, :], count, axis=0)
        array = np.asarray(value, dtype=float)
        if array.ndim == 0:
            return np.repeat((np.eye(3) * float(array))[None, :, :], count, axis=0)
        if array.shape == (3, 3):
            return np.repeat(array[None, :, :], count, axis=0)
        if array.shape != (count, 3, 3):
            raise ValueError(
                "uncertainty must be a variance, 3x3 matrix, or one matrix per candidate"
            )
        return array
