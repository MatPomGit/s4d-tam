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
from .telemetry import EventSink
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


@dataclass(frozen=True, slots=True)
class ModalityNoiseModel:
    """Configurable diagonal observation and process-noise model.

    Args:
        modality_variances: Observation variance for each sensor modality.
        default_variance: Observation variance for unlisted and fused inputs.
        process_variance_per_s: State-transition variance accumulated per second.
        quality_power: Exponent controlling the low-quality variance penalty.
        minimum_quality: Lower bound used when scaling rejected/missing inputs.

    Unknown and fused modalities use ``default_variance``. All returned matrices
    are strictly positive definite, including zero-noise configurations.
    """

    modality_variances: dict[str, float] | None = None
    default_variance: float = 0.05
    process_variance_per_s: float = 0.01
    quality_power: float = 2.0
    minimum_quality: float = 0.05

    def __post_init__(self) -> None:
        values = list((self.modality_variances or {}).values()) + [
            self.default_variance,
            self.process_variance_per_s,
        ]
        if any(not np.isfinite(value) or value < 0 for value in values):
            raise ValueError("noise variances must be finite and non-negative")
        if not np.isfinite(self.quality_power) or self.quality_power < 0:
            raise ValueError("quality_power must be finite and non-negative")
        if not 0 < self.minimum_quality <= 1:
            raise ValueError("minimum_quality must be in (0, 1]")

    def covariance(self, modality: str, quality: float, dt: float) -> np.ndarray:
        """Return measurement covariance for a sensor sample.

        Args:
            modality: Sensor modality name or ``"fused"``.
            quality: Normalized measurement quality in the closed interval [0, 1].
            dt: Time since the preceding frame, in seconds.

        Returns:
            A positive-definite 3-by-3 position covariance matrix.

        Raises:
            ValueError: If quality or elapsed time is outside its valid domain.
        """
        if not np.isfinite(quality) or not 0 <= quality <= 1:
            raise ValueError("measurement quality must be finite and in [0, 1]")
        if not np.isfinite(dt) or dt < 0:
            raise ValueError("time interval must be finite and non-negative")
        base = (self.modality_variances or {}).get(modality, self.default_variance)
        scale = max(quality, self.minimum_quality) ** (-self.quality_power)
        variance = base * scale + self.process_variance_per_s * dt
        return np.eye(3) * max(float(variance), np.finfo(float).eps)

    def process_covariance(self, dt: float) -> np.ndarray:
        """Return state-transition covariance accumulated over ``dt`` seconds."""
        if not np.isfinite(dt) or dt < 0:
            raise ValueError("time interval must be finite and non-negative")
        variance = max(self.process_variance_per_s * dt, np.finfo(float).eps)
        return np.eye(3) * variance


class TokenMemory:
    """Manage association, lifecycle transitions, attention and bounded storage.

    Args:
        association_radius_m: Radius for radial association and default local attention.
        process_noise: Backwards-compatible default process-noise variance per second.
        noise_model: Modality-, quality-, and time-aware noise configuration. When
            supplied, it supersedes ``process_noise``.
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
        event_sink: Optional structured event collector.
        log_attention_components: Include attention levels in pruning events.

    Raises:
        ValueError: If configuration values are invalid or conflicting.
    """

    def __init__(
        self,
        association_radius_m: float = 0.35,
        process_noise: float = 0.01,
        noise_model: ModalityNoiseModel | None = None,
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
        event_sink: EventSink | None = None,
        log_attention_components: bool = True,
    ) -> None:
        if not np.isfinite(association_radius_m) or association_radius_m <= 0:
            raise ValueError("association_radius_m must be finite and positive")
        if not np.isfinite(process_noise) or process_noise < 0:
            raise ValueError("process_noise must be finite and non-negative")
        if not 0 <= new_token_threshold <= 1:
            raise ValueError("new_token_threshold must be between zero and one")
        self.association_radius_m = association_radius_m
        self.noise_model = noise_model or ModalityNoiseModel(process_variance_per_s=process_noise)
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
        self.event_sink = event_sink
        self.log_attention_components = log_attention_components
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
        self._emit(
            "memory_initialized",
            None,
            association_mode=association_mode,
            max_tokens=self.budgets.max_tokens,
            max_memory_bytes=self.budgets.max_memory_bytes,
            max_history_entries=self.budgets.max_history_entries,
        )

    def _emit(self, event: str, sequence_time_s: float | None, **fields: object) -> None:
        """Forward one structured event when collection is enabled."""
        if self.event_sink is not None:
            self.event_sink.emit(event, sequence_time_s, **fields)

    def update(
        self,
        position: np.ndarray,
        timestamp: float,
        semantic_class: int | None = None,
        *,
        modality: str = "legacy",
        quality: float = 1.0,
    ) -> Token4D:
        """Create or update one token from a position observation.

        Args:
            position: XYZ observation accepted by ``TokenProposalModule``.
            timestamp: Monotonic sequence timestamp in seconds.
            semantic_class: Optional class identifier in the eight-class legacy space.
            modality: Source modality used to select measurement noise.
            quality: Normalized measurement quality in [0, 1].

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
        dt = (
            0.0
            if self._current_timestamp_s is None
            else max(timestamp - self._current_timestamp_s, 0.0)
        )
        candidate = self.proposals.propose(
            np.asarray(position),
            timestamp,
            semantic_logits=logits[None, :],
            uncertainty=self.noise_model.covariance(modality, quality, dt),
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
        self._emit(
            "frame_started",
            now_s,
            candidate_count=len(candidates),
            resident_token_count=len(self.tokens),
        )
        self._advance_lifecycle(now_s)
        result = self.associator.associate(self.tokens, candidates)
        self.last_association = result
        output: dict[int, Token4D] = {}
        for match in result.matches:
            token = self.tokens[match.token_index]
            self._merge(token, candidates[match.candidate_index])
            self._emit(
                "token_matched",
                now_s,
                token_id=token.token_id,
                candidate_index=match.candidate_index,
                confidence=float(match.confidence),
                evidence={key: float(value) for key, value in match.features.items()},
            )
            output[match.candidate_index] = token
        new_ids = {id(candidate) for candidate in result.new_candidates}
        for index, candidate in enumerate(candidates):
            if id(candidate) in new_ids:
                if candidate.confidence >= self.new_token_threshold:
                    output[index] = self._create(candidate)
                else:
                    result.discarded_candidates.append(candidate)
                    self._emit(
                        "proposal_discarded",
                        now_s,
                        candidate_index=index,
                        confidence=float(candidate.confidence),
                        threshold=float(self.new_token_threshold),
                    )
        result.metadata["discarded_proposals"] = len(result.discarded_candidates)
        self._record_association(result, new_tokens=len(output) - len(result.matches))
        merged_into = self._merge_duplicates()
        output = {index: merged_into.get(token.token_id, token) for index, token in output.items()}
        self._apply_budgets(now_s)
        self._finish_timing(started)
        self._emit(
            "frame_completed",
            now_s,
            matched_count=len(result.matches),
            output_count=len(output),
            resident_token_count=len(self.tokens),
            map_bytes=self.map_bytes,
            update_time_ms=self.last_update_ms,
            time_budget_exceeded=self.time_budget_exceeded,
        )
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
        if self.time_budget_exceeded:
            self._emit(
                "time_budget_exceeded",
                self._current_timestamp_s,
                measured_ms=self.last_update_ms,
                budget_ms=limit,
            )

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
            state=(
                TokenState.ACTIVE if self.lifecycle.activation_hits <= 1 else TokenState.PENDING
            ),
        )
        self._next_token_id += 1
        self.tokens.append(token)
        self._emit(
            "token_created",
            candidate.timestamp,
            token_id=token.token_id,
            state=token.state.value,
            confidence=float(candidate.confidence),
        )
        return token

    def _merge(self, token: Token4D, candidate: TokenCandidate) -> None:
        """Fuse a matched observation into ``token`` using a Kalman-style update."""
        dt = max(candidate.timestamp - token.last_seen_s, 1e-6)
        was_active = token.state == TokenState.ACTIVE
        previous = token.position.copy()
        prior_cov = token.covariance + self.noise_model.process_covariance(dt)
        innovation_cov = prior_cov + candidate.covariance
        gain = np.linalg.solve(innovation_cov.T, prior_cov.T).T
        token.position = token.position + gain @ (candidate.position - token.position)
        identity = np.eye(3)
        residual_gain = identity - gain
        # Joseph form preserves symmetry and positive definiteness under rounding.
        token.covariance = (
            residual_gain @ prior_cov @ residual_gain.T + gain @ candidate.covariance @ gain.T
        )
        token.covariance = 0.5 * (token.covariance + token.covariance.T)
        token.velocity = 0.7 * token.velocity + 0.3 * (token.position - previous) / dt
        token.last_seen_s = candidate.timestamp
        token.observations += 1
        token.hit_count += 1
        if was_active:
            token.active_time_s += dt
        if token.state == TokenState.SLEEPING and self.lifecycle.reactivate_on_match:
            token.state = TokenState.ACTIVE
            token.activated_at_s = candidate.timestamp
            self._emit(
                "token_state_changed",
                candidate.timestamp,
                token_id=token.token_id,
                previous_state=TokenState.SLEEPING.value,
                new_state=TokenState.ACTIVE.value,
                reason="matched_observation",
            )
        token.history.append(candidate.position.copy())
        self._trim_history(token)
        token.semantic_logits = 0.95 * token.semantic_logits + candidate.semantic_logits
        if candidate.sensory_descriptor.size:
            token.sensory_descriptor = candidate.sensory_descriptor.copy()
            token.embedding = candidate.sensory_descriptor.copy()
        if token.hit_count >= self.lifecycle.activation_hits:
            previous_state = token.state
            token.state = TokenState.ACTIVE
            if previous_state != token.state:
                self._emit(
                    "token_state_changed",
                    candidate.timestamp,
                    token_id=token.token_id,
                    previous_state=previous_state.value,
                    new_state=token.state.value,
                    reason="activation_hit_threshold",
                )

    def _advance_lifecycle(self, now_s: float) -> None:
        """Sleep or remove inactive tokens at ``now_s``."""
        retained: list[Token4D] = []
        for token in self.tokens:
            age = now_s - token.last_seen_s
            if age >= self.lifecycle.remove_after_s:
                self._emit(
                    "token_removed",
                    now_s,
                    token_id=token.token_id,
                    previous_state=token.state.value,
                    reason="inactivity_timeout",
                    inactive_for_s=float(age),
                )
                continue
            if age >= self.lifecycle.sleep_after_s and token.state != TokenState.SLEEPING:
                previous_state = token.state
                token.state = TokenState.SLEEPING
                self._emit(
                    "token_state_changed",
                    now_s,
                    token_id=token.token_id,
                    previous_state=previous_state.value,
                    new_state=token.state.value,
                    reason="inactivity_timeout",
                )
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
                    survivor.position * survivor.hit_count
                    + duplicate.position * duplicate.hit_count
                ) / total_hits
                survivor.semantic_logits += duplicate.semantic_logits
                survivor.hit_count = total_hits
                survivor.observations += duplicate.observations
                survivor.history.extend(duplicate.history)
                self._trim_history(survivor)
                merged_into[duplicate.token_id] = survivor
                self.tokens.remove(duplicate)
                self._emit(
                    "tokens_merged",
                    self._current_timestamp_s,
                    survivor_token_id=survivor.token_id,
                    removed_token_id=duplicate.token_id,
                    survivor_hit_count=survivor.hit_count,
                )
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
        score_set = self.attention.score_components(self.tokens, now_s)
        scores = score_set.combined
        for token in self.tokens:
            token.attention_score = scores[token.token_id]

        def over_budget() -> bool:
            """Return whether a hard capacity limit is currently exceeded."""
            return (
                self.budgets.max_tokens is not None and len(self.tokens) > self.budgets.max_tokens
            ) or (
                self.budgets.max_memory_bytes is not None
                and self.map_bytes > self.budgets.max_memory_bytes
            )

        # score then token id gives deterministic resolution of equal scores.
        while self.tokens and over_budget():
            victim = min(self.tokens, key=lambda token: (token.attention_score, token.token_id))
            fields: dict[str, object] = {
                "token_id": victim.token_id,
                "attention_score": victim.attention_score,
                "resident_token_count_before": len(self.tokens),
                "map_bytes_before": self.map_bytes,
                "reason": "capacity_budget",
            }
            if self.log_attention_components:
                fields.update(
                    local_attention=score_set.local[victim.token_id],
                    temporal_attention=score_set.temporal[victim.token_id],
                    global_attention=score_set.global_[victim.token_id],
                )
            self._emit("token_pruned", now_s, **fields)
            self.tokens.remove(victim)

    def _nearest(self, position: np.ndarray) -> Token4D | None:
        """Deprecated compatibility helper; association never calls this method."""
        if not self.tokens:
            return None
        return min(self.tokens, key=lambda token: float(np.linalg.norm(token.position - position)))
