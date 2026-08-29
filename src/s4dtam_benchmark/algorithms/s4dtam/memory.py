from __future__ import annotations

import numpy as np

from .association import AssociationResult, FeatureAssociator, RadialAssociator, TokenAssociator
from .proposal import TokenCandidate, TokenProposalModule
from .token import Token4D


class TokenMemory:
    """Token lifecycle driven by a configurable association interface."""

    def __init__(
        self,
        association_radius_m: float = 0.35,
        process_noise: float = 0.01,
        associator: TokenAssociator | None = None,
        *,
        association_mode: str = "feature",
        rejection_threshold: float = 0.35,
        new_token_threshold: float = 0.5,
    ):
        self.association_radius_m = association_radius_m
        self.process_noise = process_noise
        self.new_token_threshold = new_token_threshold
        if associator is not None:
            self.associator = associator
        elif association_mode == "feature":
            self.associator = FeatureAssociator(rejection_threshold=rejection_threshold)
        elif association_mode == "radial":
            self.associator = RadialAssociator(association_radius_m)
        else:
            raise ValueError("association_mode must be 'feature' or 'radial'")
        self.tokens: list[Token4D] = []
        self._next_token_id = 0
        self.last_association: AssociationResult | None = None
        self.proposals = TokenProposalModule()

    def update(
        self, position: np.ndarray, timestamp: float, semantic_class: int | None = None
    ) -> Token4D:
        logits = np.zeros(8, dtype=float)
        if semantic_class is not None and 0 <= semantic_class < len(logits):
            logits[semantic_class] = 1.0
        candidate = self.proposals.propose(
            np.asarray(position), timestamp, semantic_logits=logits[None, :]
        )[0]
        return self.update_candidates([candidate])[0]

    def update_candidates(self, candidates: list[TokenCandidate]) -> list[Token4D]:
        """Associate a frame jointly, resolving both directions of assignment conflict."""
        if not candidates:
            self.last_association = self.associator.associate(self.tokens, [])
            return []
        result = self.associator.associate(self.tokens, candidates)
        self.last_association = result
        output: dict[int, Token4D] = {}
        for match in result.matches:
            token = self.tokens[match.token_index]
            self._merge(token, candidates[match.candidate_index])
            output[match.candidate_index] = token
        new_ids = {id(candidate) for candidate in result.new_candidates}
        for index, candidate in enumerate(candidates):
            if id(candidate) in new_ids:
                if candidate.confidence >= self.new_token_threshold:
                    output[index] = self._create(candidate)
                else:
                    result.discarded_candidates.append(candidate)
        result.metadata["discarded_proposals"] = len(result.discarded_candidates)
        return [output[index] for index in sorted(output)]

    def _create(self, candidate: TokenCandidate) -> Token4D:
        token = Token4D(
            token_id=self._next_token_id,
            position=candidate.position.copy(),
            covariance=candidate.covariance.copy(),
            velocity=np.zeros(3),
            semantic_logits=candidate.semantic_logits.copy(),
            last_seen_s=candidate.timestamp,
            history=[candidate.position.copy()],
            sensory_descriptor=candidate.sensory_descriptor.copy(),
        )
        self._next_token_id += 1
        self.tokens.append(token)
        return token

    def _merge(self, token: Token4D, candidate: TokenCandidate) -> None:
        dt = max(candidate.timestamp - token.last_seen_s, 1e-6)
        previous = token.position.copy()
        prior_cov = token.covariance + np.eye(3) * self.process_noise * dt
        gain = prior_cov @ np.linalg.pinv(prior_cov + candidate.covariance)
        token.position = token.position + gain @ (candidate.position - token.position)
        token.covariance = (np.eye(3) - gain) @ prior_cov
        token.velocity = 0.7 * token.velocity + 0.3 * (token.position - previous) / dt
        token.last_seen_s = candidate.timestamp
        token.observations += 1
        token.history.append(candidate.position.copy())
        token.semantic_logits = 0.95 * token.semantic_logits + candidate.semantic_logits
        if candidate.sensory_descriptor.size:
            token.sensory_descriptor = candidate.sensory_descriptor.copy()

    def _nearest(self, position: np.ndarray) -> Token4D | None:
        """Deprecated compatibility helper; association never calls this method."""
        if not self.tokens:
            return None
        return min(self.tokens, key=lambda token: float(np.linalg.norm(token.position - position)))
