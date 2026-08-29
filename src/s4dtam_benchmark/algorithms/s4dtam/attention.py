"""Hierarchical, deterministic attention for persistent 4-D map tokens.

The reference implementation deliberately avoids learned parameters and random
operations.  This makes attention scores reproducible across runs and safe to
use as the primary key of the memory pruning policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .token import Token4D


@dataclass(frozen=True, slots=True)
class AttentionWeights:
    """Weights assigned to the three levels of hierarchical attention.

    Args:
        local: Contribution of spatial neighbourhood support.
        temporal: Contribution of observation recency.
        global_: Contribution of long-term and scene-level evidence.

    Raises:
        ValueError: If a weight is negative or all weights are zero.
    """

    local: float = 0.35
    temporal: float = 0.25
    global_: float = 0.40

    def __post_init__(self) -> None:
        values = (self.local, self.temporal, self.global_)
        if any(not np.isfinite(value) or value < 0 for value in values):
            raise ValueError("attention weights must be finite and non-negative")
        if sum(values) <= 0:
            raise ValueError("at least one attention weight must be positive")

    @property
    def normalized(self) -> np.ndarray:
        """Return weights normalized to sum to one."""
        values = np.asarray((self.local, self.temporal, self.global_), dtype=float)
        return values / values.sum()


@dataclass(frozen=True, slots=True)
class AttentionScores:
    """Per-token component scores and their weighted aggregate."""

    local: dict[int, float]
    temporal: dict[int, float]
    global_: dict[int, float]
    combined: dict[int, float]


class HierarchicalAttention:
    """Evaluate local, temporal, and global relationships between tokens.

    Args:
        local_radius_m: Radius used to count spatial neighbours, in metres.
        temporal_scale_s: Exponential recency decay constant, in seconds.
        weights: Optional component weights. Values are normalized internally.

    Raises:
        ValueError: If either scale is non-finite or not strictly positive.
    """

    def __init__(
        self,
        local_radius_m: float = 1.0,
        temporal_scale_s: float = 5.0,
        weights: AttentionWeights | None = None,
    ) -> None:
        if not np.isfinite(local_radius_m) or local_radius_m <= 0:
            raise ValueError("local_radius_m must be finite and positive")
        if not np.isfinite(temporal_scale_s) or temporal_scale_s <= 0:
            raise ValueError("temporal_scale_s must be finite and positive")
        self.local_radius_m = float(local_radius_m)
        self.temporal_scale_s = float(temporal_scale_s)
        self.weights = weights or AttentionWeights()

    def score_components(self, tokens: list[Token4D], now_s: float) -> AttentionScores:
        """Compute all attention levels for a map snapshot.

        Local attention measures neighbourhood support. Temporal attention uses
        time since the last observation. Global attention combines repeated
        observations, semantic confidence, and similarity to the scene embedding.

        Args:
            tokens: Tokens in the current memory snapshot.
            now_s: Current monotonic sequence timestamp, in seconds.

        Returns:
            Component and combined scores keyed by stable token identifier.

        Raises:
            ValueError: If ``now_s`` is not finite or token identifiers repeat.
        """
        if not np.isfinite(now_s):
            raise ValueError("now_s must be finite")
        identifiers = [token.token_id for token in tokens]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("token identifiers must be unique")
        if not tokens:
            return AttentionScores({}, {}, {}, {})

        local = self._local_scores(tokens)
        temporal = np.exp(
            -np.maximum(now_s - np.asarray([token.last_seen_s for token in tokens]), 0.0)
            / self.temporal_scale_s
        )
        global_scores = self._global_scores(tokens)
        weights = self.weights.normalized
        combined = weights[0] * local + weights[1] * temporal + weights[2] * global_scores
        mappings = [
            {identifier: float(value) for identifier, value in zip(identifiers, values)}
            for values in (local, temporal, global_scores, combined)
        ]
        return AttentionScores(*mappings)

    def score(self, tokens: list[Token4D], now_s: float) -> dict[int, float]:
        """Return combined attention scores.

        Args:
            tokens: Tokens in the current memory snapshot.
            now_s: Current monotonic sequence timestamp, in seconds.

        Returns:
            Combined score keyed by token identifier.
        """
        return self.score_components(tokens, now_s).combined

    def _local_scores(self, tokens: list[Token4D]) -> np.ndarray:
        """Return normalized neighbour counts for ``tokens``."""
        positions = np.asarray([token.position for token in tokens], dtype=float)
        distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
        neighbours = np.maximum((distances <= self.local_radius_m).sum(axis=1) - 1, 0)
        return neighbours / max(int(neighbours.max()), 1)

    @staticmethod
    def _global_scores(tokens: list[Token4D]) -> np.ndarray:
        """Return normalized persistence, semantics, and scene agreement evidence."""
        persistence = np.log1p([token.hit_count for token in tokens])
        persistence /= max(float(persistence.max()), 1.0)
        semantic = np.asarray(
            [float(np.max(token.semantic_logits, initial=0.0)) for token in tokens]
        )
        semantic /= max(float(semantic.max(initial=0.0)), 1.0)

        non_empty = [token.embedding for token in tokens if token.embedding.size]
        embedding_agreement: np.ndarray | None = None
        if non_empty and len({embedding.shape for embedding in non_empty}) == 1:
            centroid = np.mean(non_empty, axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm > 0:
                embedding_agreement = np.zeros(len(tokens), dtype=float)
                for index, token in enumerate(tokens):
                    norm = np.linalg.norm(token.embedding)
                    if token.embedding.shape == centroid.shape and norm > 0:
                        cosine = float(np.dot(token.embedding, centroid) / (norm * centroid_norm))
                        embedding_agreement[index] = np.clip((cosine + 1.0) / 2.0, 0.0, 1.0)

        # Missing modalities are excluded instead of being interpreted as zero evidence.
        optional_evidence = []
        if np.any(semantic):
            optional_evidence.append(semantic)
        if embedding_agreement is not None:
            optional_evidence.append(embedding_agreement)
        if not optional_evidence:
            return persistence
        # Repeated observations remain the dominant global importance signal;
        # optional modalities refine rather than replace persistence evidence.
        refinement = np.mean(optional_evidence, axis=0)
        return 0.7 * persistence + 0.3 * refinement
