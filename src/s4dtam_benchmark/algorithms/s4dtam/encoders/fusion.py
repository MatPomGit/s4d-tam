from __future__ import annotations

import numpy as np

from s4dtam_benchmark.contracts import AvailabilityState

from .base import EncodedObservation


class MaskedFusion:
    """Mean fusion which preserves why every input did or did not participate."""

    def __init__(self, output_dim: int = 3):
        self.output_dim = output_dim

    def fuse(
        self, observations: list[EncodedObservation], states: dict[str, int], timestamp: float
    ) -> EncodedObservation:
        usable = [
            item.features * item.confidence
            for item in observations
            if states[item.modality] == AvailabilityState.AVAILABLE
        ]
        confidence = sum(
            item.confidence
            for item in observations
            if states[item.modality] == AvailabilityState.AVAILABLE
        )
        features = np.sum(usable, axis=0) / confidence if confidence else np.zeros(self.output_dim)
        state = AvailabilityState.AVAILABLE if usable else self._aggregate_unavailable(states)
        return EncodedObservation("fused", timestamp, features, state, min(1.0, confidence))

    @staticmethod
    def _aggregate_unavailable(states: dict[str, int]) -> int:
        # More local failures take precedence over a globally absent stream.
        for state in (AvailabilityState.QUALITY_REJECTED, AvailabilityState.SAMPLE_MISSING):
            if state in states.values():
                return state
        return AvailabilityState.STREAM_ABSENT
