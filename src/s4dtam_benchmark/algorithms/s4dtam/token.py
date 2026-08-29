from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class Token4D:
    token_id: int
    position: np.ndarray
    covariance: np.ndarray
    velocity: np.ndarray
    semantic_logits: np.ndarray
    last_seen_s: float
    observations: int = 1
    risk: float = 0.0
    history: list[np.ndarray] = field(default_factory=list)

    @property
    def uncertainty(self) -> float:
        return float(np.trace(self.covariance))
