"""Configurable token association with a transparent radial fallback."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from scipy.optimize import linear_sum_assignment

from .proposal import TokenCandidate
from .token import Token4D


@dataclass(frozen=True, slots=True)
class TokenMatch:
    token_index: int
    candidate_index: int
    confidence: float
    features: dict[str, float]


@dataclass(slots=True)
class AssociationResult:
    matches: list[TokenMatch]
    new_candidates: list[TokenCandidate]
    rejected_pairs: list[tuple[int, int, float]] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)
    discarded_candidates: list[TokenCandidate] = field(default_factory=list)


class TokenAssociator(Protocol):
    def associate(
        self, tokens: list[Token4D], candidates: list[TokenCandidate]
    ) -> AssociationResult: ...


def _similarity(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or left.shape != right.shape:
        return 0.0
    norm = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if norm == 0.0 else float(np.dot(left, right) / norm)


class FeatureAssociator:
    """One-to-one global assignment over spatial, temporal and sensory evidence."""

    def __init__(self, rejection_threshold: float = 0.35, max_mahalanobis: float = 6.0):
        if not 0 <= rejection_threshold <= 1:
            raise ValueError("rejection_threshold must be between zero and one")
        self.rejection_threshold = rejection_threshold
        self.max_mahalanobis = max_mahalanobis

    def associate(
        self, tokens: list[Token4D], candidates: list[TokenCandidate]
    ) -> AssociationResult:
        if not tokens or not candidates:
            return AssociationResult([], list(candidates), metadata=self._metadata([], []))
        confidence = np.zeros((len(tokens), len(candidates)))
        feature_grid: list[list[dict[str, float]]] = []
        for ti, token in enumerate(tokens):
            row = []
            for ci, candidate in enumerate(candidates):
                values = self._features(token, candidate)
                row.append(values)
                confidence[ti, ci] = self._confidence(values)
            feature_grid.append(row)

        token_indices, candidate_indices = linear_sum_assignment(1.0 - confidence)
        matches, rejected = [], []
        matched_candidates: set[int] = set()
        for ti, ci in zip(token_indices, candidate_indices, strict=True):
            score = float(confidence[ti, ci])
            if score >= self.rejection_threshold:
                matches.append(TokenMatch(int(ti), int(ci), score, feature_grid[ti][ci]))
                matched_candidates.add(int(ci))
            else:
                rejected.append((int(ti), int(ci), score))
        plausible = confidence >= self.rejection_threshold
        conflicts = {
            "many_to_one": int(np.sum(np.sum(plausible, axis=0) > 1)),
            "one_to_many": int(np.sum(np.sum(plausible, axis=1) > 1)),
        }
        return AssociationResult(
            matches,
            [candidate for i, candidate in enumerate(candidates) if i not in matched_candidates],
            rejected,
            self._metadata(conflicts, confidence),
        )

    def _features(self, token: Token4D, candidate: TokenCandidate) -> dict[str, float]:
        dt = max(candidate.timestamp - token.last_seen_s, 0.0)
        predicted = token.position + token.velocity * dt
        delta = candidate.position - predicted
        covariance = token.covariance + candidate.covariance + np.eye(3) * 1e-8
        mahalanobis = float(np.sqrt(delta @ np.linalg.pinv(covariance) @ delta))
        temporal = float(np.exp(-dt / 2.0))
        motion_error = float(np.linalg.norm(delta))
        semantic = _similarity(token.semantic_logits, candidate.semantic_logits)
        sensory = _similarity(token.sensory_descriptor, candidate.sensory_descriptor)
        return {
            "mahalanobis": mahalanobis,
            "temporal": temporal,
            "motion": motion_error,
            "semantic": semantic,
            "sensory": sensory,
        }

    def _confidence(self, values: dict[str, float]) -> float:
        if values["mahalanobis"] > self.max_mahalanobis:
            return 0.0
        spatial = np.exp(-0.5 * values["mahalanobis"] ** 2)
        motion = np.exp(-values["motion"])
        semantic = (values["semantic"] + 1.0) / 2.0
        sensory = (values["sensory"] + 1.0) / 2.0
        return float(
            0.15 * spatial
            + 0.1 * values["temporal"]
            + 0.15 * motion
            + 0.15 * semantic
            + 0.45 * sensory
        )

    def _metadata(self, conflicts: object, confidence: object) -> dict[str, object]:
        return {
            "associator": "feature_global",
            "radial_fallback_used": False,
            "rejection_threshold": self.rejection_threshold,
            "conflicts": conflicts or {"many_to_one": 0, "one_to_many": 0},
        }


class RadialAssociator:
    """Simple radius-gated global association baseline and fallback."""

    def __init__(self, radius_m: float = 0.35):
        self.radius_m = radius_m

    def associate(
        self, tokens: list[Token4D], candidates: list[TokenCandidate]
    ) -> AssociationResult:
        if not tokens or not candidates:
            return AssociationResult([], list(candidates), metadata=self._metadata())
        distances = np.array(
            [[np.linalg.norm(t.position - c.position) for c in candidates] for t in tokens]
        )
        rows, cols = linear_sum_assignment(distances)
        matches, rejected, matched = [], [], set()
        for ti, ci in zip(rows, cols, strict=True):
            distance = float(distances[ti, ci])
            confidence = max(0.0, 1.0 - distance / self.radius_m)
            if distance <= self.radius_m:
                matches.append(TokenMatch(int(ti), int(ci), confidence, {"distance": distance}))
                matched.add(int(ci))
            else:
                rejected.append((int(ti), int(ci), confidence))
        return AssociationResult(
            matches,
            [c for i, c in enumerate(candidates) if i not in matched],
            rejected,
            self._metadata(),
        )

    def _metadata(self) -> dict[str, object]:
        return {"associator": "radial", "radial_fallback_used": True, "radius_m": self.radius_m}
