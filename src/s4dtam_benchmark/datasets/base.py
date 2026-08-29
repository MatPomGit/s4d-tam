from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from s4dtam_benchmark.contracts import SequenceData


class DatasetAdapter(ABC):
    @abstractmethod
    def sequences(self) -> Iterator[SequenceData]:
        """Yield normalized benchmark sequences."""
