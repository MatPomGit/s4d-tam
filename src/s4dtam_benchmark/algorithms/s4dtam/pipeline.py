from __future__ import annotations

from time import perf_counter

import numpy as np

from s4dtam_benchmark.algorithms.base import AlgorithmAdapter
from s4dtam_benchmark.contracts import AlgorithmResult, RunContext, SequenceData

from .memory import TokenMemory


class S4DTAMReference(AlgorithmAdapter):
    """Transparent CPU reference for the proposed token lifecycle and interfaces."""

    name = "s4d_tam_reference"

    def __init__(self, association_radius_m: float = 0.35):
        self.association_radius_m = association_radius_m

    def run(self, sequence: SequenceData, context: RunContext) -> AlgorithmResult:
        if sequence.observations is None:
            raise ValueError("S4D-TAM reference requires normalized 3-D observations")
        memory = TokenMemory(self.association_radius_m)
        estimates, semantics, latency = [], [], []
        for index, (timestamp, observation) in enumerate(
            zip(sequence.timestamps, sequence.observations, strict=True)
        ):
            start = perf_counter()
            semantic_hint = (
                int(sequence.semantic_observations[index])
                if sequence.semantic_observations is not None
                else None
            )
            token = memory.update(observation, float(timestamp), semantic_hint)
            estimates.append(token.position.copy())
            semantics.append(int(np.argmax(token.semantic_logits)))
            latency.append((perf_counter() - start) * 1000.0)

        occupancy_pred = {}
        if sequence.occupancy_observations is not None:
            for horizon in sequence.occupancy_gt:
                steps = max(1, int(round(horizon / np.median(np.diff(sequence.timestamps)))))
                velocity_proxy = np.roll(sequence.occupancy_observations, steps, axis=0)
                velocity_proxy[:steps] = sequence.occupancy_observations[:steps]
                occupancy_pred[horizon] = np.clip(
                    0.65 * sequence.occupancy_observations + 0.25 * velocity_proxy + 0.05,
                    0.0,
                    1.0,
                )
        return AlgorithmResult(
            algorithm=self.name,
            timestamps=sequence.timestamps,
            estimated_positions=np.asarray(estimates),
            semantic_pred=np.asarray(semantics),
            occupancy_pred=occupancy_pred,
            latency_ms=np.asarray(latency),
            resource={
                "token_count": float(len(memory.tokens)),
                "map_bytes": float(
                    sum(
                        token.position.nbytes
                        + token.covariance.nbytes
                        + token.velocity.nbytes
                        + token.semantic_logits.nbytes
                        for token in memory.tokens
                    )
                ),
            },
            metadata={"implementation": "reference_cpu", "not_flight_certified": True},
        )
