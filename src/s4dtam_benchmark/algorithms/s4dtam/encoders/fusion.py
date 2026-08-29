from __future__ import annotations

import numpy as np

from s4dtam_benchmark.contracts import MODALITIES, AvailabilityState

from .base import EncodedObservation


class MaskedFusion:
    """Mean fusion which preserves why every input did or did not participate."""

    def __init__(self, output_dim: int = 3, modality_weights: dict[str, float] | None = None):
        if output_dim < 1:
            raise ValueError("output_dim must be positive")
        supplied_weights = modality_weights or {}
        unknown = set(supplied_weights) - set(MODALITIES)
        if unknown:
            raise ValueError(f"unknown modality weights: {sorted(unknown)}")
        if any(not np.isfinite(weight) or weight <= 0 for weight in supplied_weights.values()):
            raise ValueError("modality weights must be finite and positive")
        self.output_dim = int(output_dim)
        self.modality_weights = {
            name: float(supplied_weights.get(name, 1.0)) for name in MODALITIES
        }

    def fuse(
        self, observations: list[EncodedObservation], states: dict[str, int], timestamp: float
    ) -> EncodedObservation:
        normalized_states = self._validate_inputs(observations, states)
        available = [
            item
            for item in observations
            if normalized_states[item.modality] == AvailabilityState.AVAILABLE
        ]
        weights = np.array(
            [item.confidence * self.modality_weights[item.modality] for item in available]
        )
        features = (
            np.average(np.stack([item.features for item in available]), axis=0, weights=weights)
            if available
            else np.zeros(self.output_dim)
        )
        confidence = float(1.0 - np.prod(1.0 - np.array([item.confidence for item in available])))
        state = (
            AvailabilityState.AVAILABLE
            if available
            else self._aggregate_unavailable(normalized_states)
        )
        return EncodedObservation("fused", timestamp, features, state, confidence)

    def _validate_inputs(
        self, observations: list[EncodedObservation], states: dict[str, int]
    ) -> dict[str, AvailabilityState]:
        unknown = set(states) - set(MODALITIES)
        if unknown:
            raise ValueError(f"unknown modality states: {sorted(unknown)}")
        normalized = {name: AvailabilityState(value) for name, value in states.items()}
        seen: set[str] = set()
        for observation in observations:
            if observation.modality not in MODALITIES:
                raise ValueError(f"unknown encoded modality: {observation.modality!r}")
            if observation.modality in seen:
                raise ValueError(f"duplicate encoded modality: {observation.modality!r}")
            if observation.modality not in normalized:
                raise ValueError(f"missing availability state for {observation.modality!r}")
            if observation.features.shape != (self.output_dim,):
                raise ValueError(
                    f"{observation.modality} features must have shape ({self.output_dim},)"
                )
            seen.add(observation.modality)
        return normalized

    @staticmethod
    def _aggregate_unavailable(states: dict[str, int]) -> int:
        # More local failures take precedence over a globally absent stream.
        for state in (AvailabilityState.QUALITY_REJECTED, AvailabilityState.SAMPLE_MISSING):
            if state in states.values():
                return state
        return AvailabilityState.STREAM_ABSENT
