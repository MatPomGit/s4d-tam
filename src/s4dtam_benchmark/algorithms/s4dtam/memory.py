from __future__ import annotations

import numpy as np

from .token import Token4D


class TokenMemory:
    """Minimal numerical token memory; replace encoders without changing evaluation APIs."""

    def __init__(self, association_radius_m: float = 0.35, process_noise: float = 0.01):
        self.association_radius_m = association_radius_m
        self.process_noise = process_noise
        self.tokens: list[Token4D] = []

    def update(self, position: np.ndarray, timestamp: float, semantic_class: int | None = None) -> Token4D:
        token = self._nearest(position)
        if token is None or np.linalg.norm(token.position - position) > self.association_radius_m:
            logits = np.zeros(8, dtype=float)
            if semantic_class is not None and semantic_class < len(logits):
                logits[semantic_class] = 1.0
            token = Token4D(
                token_id=len(self.tokens),
                position=position.copy(),
                covariance=np.eye(3) * 0.05,
                velocity=np.zeros(3),
                semantic_logits=logits,
                last_seen_s=timestamp,
                history=[position.copy()],
            )
            self.tokens.append(token)
            return token

        dt = max(timestamp - token.last_seen_s, 1e-6)
        previous = token.position.copy()
        measurement_noise = np.eye(3) * 0.04
        prior_cov = token.covariance + np.eye(3) * self.process_noise * dt
        gain = prior_cov @ np.linalg.inv(prior_cov + measurement_noise)
        token.position = token.position + gain @ (position - token.position)
        token.covariance = (np.eye(3) - gain) @ prior_cov
        token.velocity = 0.7 * token.velocity + 0.3 * (token.position - previous) / dt
        token.last_seen_s = timestamp
        token.observations += 1
        token.history.append(position.copy())
        if semantic_class is not None and semantic_class < len(token.semantic_logits):
            token.semantic_logits *= 0.95
            token.semantic_logits[semantic_class] += 1.0
        return token

    def _nearest(self, position: np.ndarray) -> Token4D | None:
        if not self.tokens:
            return None
        return min(self.tokens, key=lambda token: float(np.linalg.norm(token.position - position)))
