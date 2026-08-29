from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class TokenState(str, Enum):
    """Lifecycle state of a persistent map token."""

    PENDING = "pending"
    ACTIVE = "active"
    SLEEPING = "sleeping"


@dataclass(slots=True)
class Token4D:
    """Persistent spatial token enriched with temporal and importance signals.

    Array fields own their buffers. ``history`` stores position snapshots and is
    bounded by :class:`ResourceBudgets` when managed by ``TokenMemory``.

    Args:
        token_id: Stable, monotonically increasing identifier.
        position: Current XYZ position in metres.
        covariance: Three-dimensional position covariance.
        velocity: Current XYZ velocity estimate in metres per second.
        semantic_logits: Accumulated semantic-class evidence.
        last_seen_s: Timestamp of the latest associated observation.
        observations: Number of observations merged into this token.
        risk: Optional downstream risk estimate.
        history: Retained position observations.
        sensory_descriptor: Most recent association descriptor.
        embedding: Feature vector used by global attention.
        activated_at_s: Timestamp of the latest activation.
        active_time_s: Accumulated time while the token was active.
        hit_count: Number of successful observations used for importance.
        attention_score: Most recently computed aggregate attention score.
        state: Current lifecycle state.
    """
    token_id: int
    position: np.ndarray
    covariance: np.ndarray
    velocity: np.ndarray
    semantic_logits: np.ndarray
    last_seen_s: float
    observations: int = 1
    risk: float = 0.0
    history: list[np.ndarray] = field(default_factory=list)
    sensory_descriptor: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    embedding: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    activated_at_s: float = 0.0
    active_time_s: float = 0.0
    hit_count: int = 1
    attention_score: float = 0.0
    state: TokenState = TokenState.ACTIVE

    @property
    def uncertainty(self) -> float:
        """Return total positional uncertainty as covariance trace."""
        return float(np.trace(self.covariance))
