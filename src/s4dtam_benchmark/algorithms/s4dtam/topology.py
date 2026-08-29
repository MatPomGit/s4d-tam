"""Topological place graph with separate retrieval and geometric verification."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .reference_map import ReferenceMap, ReferenceToken


@dataclass(frozen=True, slots=True)
class PlaceCandidate:
    """Descriptor-retrieval result; no geometric decision has been made yet."""

    token_id: int
    descriptor_similarity: float
    position: np.ndarray

    def __post_init__(self) -> None:
        position = np.asarray(self.position, dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("candidate position must be a finite XYZ vector")
        object.__setattr__(self, "position", position.copy())


@dataclass(frozen=True, slots=True)
class VerifiedMatch:
    """Map match accepted by independent descriptor and geometric gates."""

    token_id: int
    confidence: float
    residual_m: float
    correction: np.ndarray


@dataclass(frozen=True, slots=True)
class MatchRejection:
    """Auditable reason why a retrieved candidate was not accepted."""

    token_id: int
    reason: str
    descriptor_similarity: float
    residual_m: float | None = None

    def to_dict(self) -> dict[str, float | int | str]:
        result: dict[str, float | int | str] = {
            "token_id": self.token_id,
            "reason": self.reason,
            "similarity": self.descriptor_similarity,
        }
        if self.residual_m is not None:
            result["residual_m"] = self.residual_m
        return result


@dataclass(slots=True)
class TopologicalGraph:
    """Graph of places/transitions providing alias-resistant map matching."""

    reference_map: ReferenceMap
    transitions: set[tuple[int, int]] = field(default_factory=set)
    descriptor_threshold: float = 0.8
    ambiguity_margin: float = 0.05
    geometric_threshold_m: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.descriptor_threshold <= 1.0:
            raise ValueError("descriptor_threshold must be between zero and one")
        if not 0.0 <= self.ambiguity_margin <= 1.0:
            raise ValueError("ambiguity_margin must be between zero and one")
        if not np.isfinite(self.geometric_threshold_m) or self.geometric_threshold_m <= 0:
            raise ValueError("geometric_threshold_m must be finite and positive")
        for source, target in self.transitions:
            self._validate_transition(source, target)

    def add_place(self, place: ReferenceToken) -> None:
        """Add a place while preserving the map's stable identifier invariant."""
        if place.frame not in self.reference_map.coordinate_frames:
            raise ValueError(f"place uses unknown coordinate frame: {place.frame}")
        if any(item.token_id == place.token_id for item in self.reference_map.tokens):
            raise ValueError(f"duplicate place identifier: {place.token_id}")
        self.reference_map.tokens.append(place)

    def add_transition(self, source: int, target: int) -> None:
        self._validate_transition(source, target)
        self.transitions.add((source, target))

    def _validate_transition(self, source: int, target: int) -> None:
        ids = {token.token_id for token in self.reference_map.tokens}
        if source not in ids or target not in ids:
            raise KeyError("transition endpoints must be places in the reference map")

    def generate_candidates(self, descriptor: np.ndarray, limit: int = 5) -> list[PlaceCandidate]:
        """Retrieve descriptor candidates without applying any pose geometry."""
        query = np.asarray(descriptor, dtype=float)
        if query.ndim != 1 or query.size == 0 or not np.all(np.isfinite(query)):
            raise ValueError("query descriptor must be a non-empty finite vector")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("candidate limit must be a positive integer")
        candidates: list[PlaceCandidate] = []
        for token in self.reference_map.tokens:
            if query.shape != token.descriptor.shape:
                continue
            denominator = float(np.linalg.norm(query) * np.linalg.norm(token.descriptor))
            similarity = 0.0 if denominator == 0 else float(query @ token.descriptor / denominator)
            if similarity >= self.descriptor_threshold:
                position = self.reference_map.transform(token.position, token.frame)
                candidates.append(PlaceCandidate(token.token_id, similarity, position))
        return sorted(candidates, key=lambda item: (-item.descriptor_similarity, item.token_id))[:limit]

    def verify_candidates(
        self, candidates: list[PlaceCandidate], estimated_position: np.ndarray
    ) -> tuple[VerifiedMatch | None, list[dict[str, float | int | str]]]:
        """Geometrically gate retrieved places and reject perceptual aliases."""
        estimate = np.asarray(estimated_position, dtype=float)
        if estimate.shape != (3,) or not np.all(np.isfinite(estimate)):
            raise ValueError("estimated_position must be a finite XYZ vector")
        rejected: list[MatchRejection] = []
        geometrically_valid: list[tuple[float, float, PlaceCandidate]] = []
        for candidate in candidates:
            residual = float(np.linalg.norm(candidate.position - estimate))
            if residual > self.geometric_threshold_m:
                rejected.append(
                    MatchRejection(
                        candidate.token_id, "geometry", candidate.descriptor_similarity, residual
                    )
                )
                continue
            geometry = max(0.0, 1.0 - residual / self.geometric_threshold_m)
            confidence = float(candidate.descriptor_similarity * (0.5 + 0.5 * geometry))
            geometrically_valid.append((confidence, residual, candidate))

        geometrically_valid.sort(key=lambda item: (-item[0], item[1], item[2].token_id))
        if not geometrically_valid:
            return None, [item.to_dict() for item in rejected]
        best = geometrically_valid[0]
        if len(geometrically_valid) > 1 and best[0] - geometrically_valid[1][0] < self.ambiguity_margin:
            rejected.extend(
                MatchRejection(candidate.token_id, "perceptual_alias",
                               candidate.descriptor_similarity, residual)
                for _, residual, candidate in geometrically_valid
            )
            return None, [item.to_dict() for item in rejected]

        confidence, residual, candidate = best
        rejected.extend(
            MatchRejection(other.token_id, "lower_confidence", other.descriptor_similarity,
                           other_residual)
            for _, other_residual, other in geometrically_valid[1:]
        )
        return (
            VerifiedMatch(candidate.token_id, confidence, residual, candidate.position - estimate),
            [item.to_dict() for item in rejected],
        )

    def match(
        self, descriptor: np.ndarray, estimated_position: np.ndarray
    ) -> tuple[
        VerifiedMatch | None,
        list[PlaceCandidate],
        list[dict[str, float | int | str]],
    ]:
        candidates = self.generate_candidates(descriptor)
        match, rejected = self.verify_candidates(candidates, estimated_position)
        return match, candidates, rejected
