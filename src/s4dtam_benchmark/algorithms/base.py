from __future__ import annotations

from abc import ABC, abstractmethod

from s4dtam_benchmark.contracts import AlgorithmResult, RunContext, SequenceData


class AlgorithmAdapter(ABC):
    name: str

    @abstractmethod
    def run(self, sequence: SequenceData, context: RunContext) -> AlgorithmResult:
        """Run an algorithm or load its outputs for one sequence."""
