from __future__ import annotations

from time import perf_counter

import numpy as np

from s4dtam_benchmark.algorithms.base import AlgorithmAdapter
from s4dtam_benchmark.contracts import AlgorithmResult, RunContext, SequenceData


class DeadReckoning(AlgorithmAdapter):
    name = "dead_reckoning"

    def __init__(self, drift_per_step: float = 0.002):
        self.drift_per_step = drift_per_step

    def run(self, sequence: SequenceData, context: RunContext) -> AlgorithmResult:
        start = perf_counter()
        source = sequence.observations if sequence.observations is not None else sequence.gt_positions
        increments = np.diff(source, axis=0, prepend=source[[0]])
        drift = np.arange(len(source))[:, None] * self.drift_per_step
        positions = source[[0]] + np.cumsum(increments, axis=0) + np.hstack(
            (drift, np.zeros((len(source), 2)))
        )
        total_ms = (perf_counter() - start) * 1000.0
        return AlgorithmResult(
            algorithm=self.name,
            timestamps=sequence.timestamps,
            estimated_positions=positions,
            latency_ms=np.full(len(source), total_ms / max(len(source), 1)),
        )
