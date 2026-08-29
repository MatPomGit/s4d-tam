from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .association import (
    AssociationResult,
    FallbackAssociator,
    FeatureAssociator,
    RadialAssociator,
    TokenAssociator,
)
from .attention import HierarchicalAttention
from .proposal import TokenCandidate, TokenProposalModule
from .token import Token4D, TokenState


@dataclass(frozen=True, slots=True)
class LifecycleRules:
    """Thresholds controlling state transitions in token memory.

    Args:
        activation_hits: Observations required to promote a pending token.
        sleep_after_s: Inactivity duration before an active token sleeps.
        reactivate_on_match: Whether matching a sleeping token reactivates it.
        merge_distance_m: Distance below which duplicate tokens are merged. Zero
            disables automatic merging.
        remove_after_s: Inactivity duration after which a token is removed.
    """

    activation_hits: int = 1
    sleep_after_s: float = 10.0
    reactivate_on_match: bool = True
    merge_distance_m: float = 0.0
    remove_after_s: float = 60.0

    def __post_init__(self) -> None:
        if self.activation_hits < 1:
            raise ValueError("activation_hits must be at least one")
        if not np.isfinite(self.merge_distance_m) or self.merge_distance_m < 0:
            raise ValueError("merge_distance_m must be finite and non-negative")
        if not np.isfinite(self.sleep_after_s) or self.sleep_after_s < 0:
            raise ValueError("sleep_after_s must be finite and non-negative")
        if not np.isfinite(self.remove_after_s) or self.remove_after_s <= self.sleep_after_s:
            raise ValueError("remove_after_s must be finite and greater than sleep_after_s")


@dataclass(frozen=True, slots=True)
class ResourceBudgets:
    """Hard capacity limits and an update-time service-level objective.

    Token, byte and history limits are enforced synchronously. The time limit is
    measured and reported rather than used as a pruning key: wall-clock timing is
    intentionally excluded from pruning to preserve deterministic map contents.

    Args:
        max_tokens: Maximum number of resident tokens, or ``None`` for unlimited.
        max_memory_bytes: Maximum NumPy payload bytes, or ``None`` for unlimited.
        max_update_time_ms: Update latency objective in milliseconds.
        max_history_entries: Maximum position snapshots retained per token.
    """

    max_tokens: int | None = None
    max_memory_bytes: int | None = None
    max_update_time_ms: float | None = None
    max_history_entries: int | None = 64

    def __post_init__(self) -> None:
        integer_limits = (self.max_tokens, self.max_memory_bytes, self.max_history_entries)
        if any(value is not None and value < 1 for value in integer_limits):
            raise ValueError("capacity budgets must be positive when specified")
        if self.max_update_time_ms is not None and (
            not np.isfinite(self.max_update_time_ms) or self.max_update_time_ms <= 0
        ):
            raise ValueError("max_update_time_ms must be finite and positive")


class TokenMemory:
    """Manage association, lifecycle transitions, attention and bounded storage.

    Args:
        association_radius_m: Radius for radial association and default local attention.
        process_noise: Position process-noise variance per second.
        associator: Optional custom associator. It takes precedence over ``association_mode``.
        association_mode: Built-in association strategy, ``"feature"`` or ``"radial"``.
        rejection_threshold: Feature-association rejection threshold.
        new_token_threshold: Minimum proposal confidence for token creation.
        lifecycle: Lifecycle policy. Defaults to :class:`LifecycleRules`.
        budgets: Grouped resource limits. Cannot be combined with individual limits.
        attention: Custom deterministic attention evaluator.
        max_tokens: Convenience alias for ``ResourceBudgets.max_tokens``.
        max_memory_bytes: Convenience alias for ``ResourceBudgets.max_memory_bytes``.
        max_update_time_ms: Convenience alias for the update latency objective.

    Raises:
        ValueError: If configuration values are invalid or conflicting.
    """

    def __init__(
        self,
        association_radius_m: float = 0.35,
        process_noise: float = 0.01,
        associator: TokenAssociator | None = None,
        *,
        association_mode: str = "feature",
        rejection_threshold: float = 0.35,
        new_token_threshold: float = 0.5,
        lifecycle: LifecycleRules | None = None,
        budgets: ResourceBudgets | None = None,
        attention: HierarchicalAttention | None = None,
        max_tokens: int | None = None,
        max_memory_bytes: int | None = None,
        max_update_time_ms: float | None = None,
    ) -> None:
        if not np.isfinite(association_radius_m) or association_radius_m <= 0:
            raise ValueError("association_radius_m must be finite and positive")
        if not np.isfinite(process_noise) or process_noise < 0:
            raise ValueError("process_noise must be finite and non-negative")
        if not 0 <= new_token_threshold <= 1:
            raise ValueError("new_token_threshold must be between zero and one")
        self.association_radius_m = association_radius_m
        self.process_noise = process_noise
        self.new_token_threshold = new_token_threshold
        self.lifecycle = lifecycle or LifecycleRules()
        if budgets is not None and any(
            value is not None for value in (max_tokens, max_memory_bytes, max_update_time_ms)
        ):
            raise ValueError("pass either budgets or individual budget arguments, not both")
        self.budgets = budgets or ResourceBudgets(
            max_tokens=max_tokens,
            max_memory_bytes=max_memory_bytes,
            max_update_time_ms=max_update_time_ms,
        )
        self.attention = attention or HierarchicalAttention(local_radius_m=association_radius_m)
        self.last_update_ms = 0.0
        self.time_budget_exceeded = False
        if associator is not None:
            self.associator = associator
        elif association_mode == "feature":
            self.associator = FallbackAssociator(
                FeatureAssociator(rejection_threshold=rejection_threshold),
                RadialAssociator(association_radius_m),
            )
        elif association_mode == "radial":
            self.associator = RadialAssociator(association_radius_m)
        else:
            raise ValueError("association_mode must be 'feature' or 'radial'")
        self.tokens: list[Token4D] = []
        self._next_token_id = 0
        self._current_timestamp_s: float | None = None
        self.last_association: AssociationResult | None = None
        self.association_summary: dict[str, object] = {
            "frames": 0,
            "matches": 0,
            "new_tokens": 0,
            "rejected_pairs": 0,
            "suppressed_conflict_candidates": 0,
            "discarded_proposals": 0,
            "radial_fallback_used": False,
        }
        self.proposals = TokenProposalModule()

    def update(
        self, position: np.ndarray, timestamp: float, semantic_class: int | None = None
    ) -> Token4D:
        """Create or update one token from a legacy position observation.

        Args:
            position: XYZ observation accepted by ``TokenProposalModule``.
            timestamp: Monotonic sequence timestamp in seconds.
            semantic_class: Optional class identifier in the eight-class legacy space.

        Returns:
            Token associated with the observation, even if it is subsequently
            evicted to satisfy a zero-slack capacity budget.

        Raises:
            ValueError: If the semantic class is outside the supported range.
        """
        logits = np.zeros(8, dtype=float)
        if semantic_class is not None:
            if not 0 <= semantic_class < len(logits):
                raise ValueError("semantic_class must be between 0 and 7")
            logits[semantic_class] = 1.0
        candidate = self.proposals.propose(
            np.asarray(position), timestamp, semantic_logits=logits[None, :]
        )[0]
        return self.update_candidates([candidate])[0]

    def update_candidates(self, candidates: list[TokenCandidate]) -> list[Token4D]:
        """Associate a frame jointly and apply lifecycle and capacity policies.

        Args:
            candidates: Proposals from one timestamp. Candidate order is preserved
                in the returned list; rejected candidates are omitted.

        Returns:
            Associated or newly created tokens in candidate-index order.

        Raises:
            ValueError: If candidates contain non-finite or different timestamps.
        """
        started = perf_counter()
        if not candidates:
            self.last_association = self.associator.associate(self.tokens, [])
            self._record_association(self.last_association, new_tokens=0)
            self._finish_timing(started)
            return []
        timestamps = np.asarray([candidate.timestamp for candidate in candidates], dtype=float)
        if not np.all(np.isfinite(timestamps)) or not np.allclose(timestamps, timestamps[0]):
            raise ValueError("all candidates in a frame must share one finite timestamp")
        now_s = float(timestamps[0])
        self._validate_timestamp(now_s)
        self._advance_lifecycle(now_s)
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
        self._record_association(result, new_tokens=len(output) - len(result.matches))
        merged_into = self._merge_duplicates()
        output = {
            index: merged_into.get(token.token_id, token) for index, token in output.items()
        }
        self._apply_budgets(now_s)
        self._finish_timing(started)
        return [output[index] for index in sorted(output)]

    def advance(self, timestamp: float) -> None:
        """Advance lifecycle time when a frame contains no observations.

        Args:
            timestamp: Current finite sequence timestamp in seconds.
        """
        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        now_s = float(timestamp)
        self._validate_timestamp(now_s)
        self._advance_lifecycle(now_s)
        self._apply_budgets(now_s)

    def _validate_timestamp(self, timestamp: float) -> None:
        """Require non-decreasing sequence time and remember the latest timestamp."""
        if self._current_timestamp_s is not None and timestamp < self._current_timestamp_s:
            raise ValueError("timestamps must be monotonically non-decreasing")
        self._current_timestamp_s = timestamp

    def _finish_timing(self, started: float) -> None:
        """Store update latency and evaluate the non-deterministic latency SLO."""
        self.last_update_ms = (perf_counter() - started) * 1000.0
        limit = self.budgets.max_update_time_ms
        self.time_budget_exceeded = limit is not None and self.last_update_ms > limit

    def _record_association(self, result: AssociationResult, *, new_tokens: int) -> None:
        """Accumulate run-level diagnostics so an earlier fallback is never hidden."""
        increments = {
            "frames": 1,
            "matches": len(result.matches),
            "new_tokens": new_tokens,
            "rejected_pairs": len(result.rejected_pairs),
            "suppressed_conflict_candidates": len(result.suppressed_candidates),
            "discarded_proposals": len(result.discarded_candidates),
        }
        for key, value in increments.items():
            self.association_summary[key] = int(self.association_summary[key]) + value
        self.association_summary["radial_fallback_used"] = bool(
            self.association_summary["radial_fallback_used"]
            or result.metadata.get("radial_fallback_used", False)
        )
        self.association_summary["last_frame"] = dict(result.metadata)

    def _create(self, candidate: TokenCandidate) -> Token4D:
        """Create and register an owning token from ``candidate``."""
        token = Token4D(
            token_id=self._next_token_id,
            position=candidate.position.copy(),
            covariance=candidate.covariance.copy(),
            velocity=np.zeros(3),
            semantic_logits=candidate.semantic_logits.copy(),
            last_seen_s=candidate.timestamp,
            history=[candidate.position.copy()],
            sensory_descriptor=candidate.sensory_descriptor.copy(),
            embedding=candidate.sensory_descriptor.copy(),
            activated_at_s=candidate.timestamp,
            state=(TokenState.ACTIVE if self.lifecycle.activation_hits <= 1 else TokenState.PENDING),
        )
        self._next_token_id += 1
        self.tokens.append(token)
        return token

    def _merge(self, token: Token4D, candidate: TokenCandidate) -> None:
        """Fuse a matched observation into ``token`` using a Kalman-style update."""
        dt = max(candidate.timestamp - token.last_seen_s, 1e-6)
        was_active = token.state == TokenState.ACTIVE
        previous = token.position.copy()
        prior_cov = token.covariance + np.eye(3) * self.process_noise * dt
        gain = prior_cov @ np.linalg.pinv(prior_cov + candidate.covariance)
        token.position = token.position + gain @ (candidate.position - token.position)
        token.covariance = (np.eye(3) - gain) @ prior_cov
        token.velocity = 0.7 * token.velocity + 0.3 * (token.position - previous) / dt
        token.last_seen_s = candidate.timestamp
        token.observations += 1
        token.hit_count += 1
        if was_active:
            token.active_time_s += dt
        if token.state == TokenState.SLEEPING and self.lifecycle.reactivate_on_match:
            token.state = TokenState.ACTIVE
            token.activated_at_s = candidate.timestamp
        token.history.append(candidate.position.copy())
        self._trim_history(token)
        token.semantic_logits = 0.95 * token.semantic_logits + candidate.semantic_logits
        if candidate.sensory_descriptor.size:
            token.sensory_descriptor = candidate.sensory_descriptor.copy()
            token.embedding = candidate.sensory_descriptor.copy()
        if token.hit_count >= self.lifecycle.activation_hits:
            token.state = TokenState.ACTIVE

    def _advance_lifecycle(self, now_s: float) -> None:
        """Sleep or remove inactive tokens at ``now_s``."""
        retained: list[Token4D] = []
        for token in self.tokens:
            age = now_s - token.last_seen_s
            if age >= self.lifecycle.remove_after_s:
                continue
            if age >= self.lifecycle.sleep_after_s:
                token.state = TokenState.SLEEPING
            retained.append(token)
        self.tokens = retained

    def _merge_duplicates(self) -> dict[int, Token4D]:
        """Merge duplicates and map removed identifiers to their survivor."""
        radius = self.lifecycle.merge_distance_m
        if radius <= 0:
            return {}
        merged_into: dict[int, Token4D] = {}
        # Oldest identifier is the deterministic survivor.
        for survivor in sorted(self.tokens, key=lambda token: token.token_id):
            if all(survivor is not resident for resident in self.tokens):
                continue
            duplicates = [
                token
                for token in self.tokens
                if token.token_id > survivor.token_id
                and np.linalg.norm(token.position - survivor.position) <= radius
            ]
            for duplicate in duplicates:
                total_hits = survivor.hit_count + duplicate.hit_count
                survivor.position = (
                    survivor.position * survivor.hit_count + duplicate.position * duplicate.hit_count
                ) / total_hits
                survivor.semantic_logits += duplicate.semantic_logits
                survivor.hit_count = total_hits
                survivor.observations += duplicate.observations
                survivor.history.extend(duplicate.history)
                self._trim_history(survivor)
                merged_into[duplicate.token_id] = survivor
                self.tokens.remove(duplicate)
        return merged_into

    def _trim_history(self, token: Token4D) -> None:
        """Discard the oldest history entries beyond the configured limit."""
        limit = self.budgets.max_history_entries
        if limit is not None and len(token.history) > limit:
            del token.history[:-limit]

    @staticmethod
    def token_bytes(token: Token4D) -> int:
        """Return bytes owned by all NumPy payload buffers of ``token``.

        Args:
            token: Token whose position, uncertainty, motion, semantic, descriptor,
                embedding and history buffers are measured.

        Returns:
            Sum of NumPy ``nbytes`` values. Python container overhead is excluded.
        """
        arrays = [
            token.position,
            token.covariance,
            token.velocity,
            token.semantic_logits,
            token.sensory_descriptor,
            token.embedding,
            *token.history,
        ]
        return int(sum(array.nbytes for array in arrays))

    @property
    def map_bytes(self) -> int:
        """Return current bytes in all token-owned NumPy payload buffers."""
        return sum(self.token_bytes(token) for token in self.tokens)

    def _apply_budgets(self, now_s: float) -> None:
        """Update importance and prune until every hard capacity limit is met."""
        scores = self.attention.score(self.tokens, now_s)
        for token in self.tokens:
            token.attention_score = scores[token.token_id]
        def over_budget() -> bool:
            """Return whether a hard capacity limit is currently exceeded."""
            return (
                self.budgets.max_tokens is not None
                and len(self.tokens) > self.budgets.max_tokens
            ) or (
                self.budgets.max_memory_bytes is not None
                and self.map_bytes > self.budgets.max_memory_bytes
            )
        # score then token id gives deterministic resolution of equal scores.
        while self.tokens and over_budget():
            victim = min(self.tokens, key=lambda token: (token.attention_score, token.token_id))
            self.tokens.remove(victim)

    def _nearest(self, position: np.ndarray) -> Token4D | None:
        """Deprecated compatibility helper; association never calls this method."""
        if not self.tokens:
            return None
        return min(self.tokens, key=lambda token: float(np.linalg.norm(token.position - position)))
