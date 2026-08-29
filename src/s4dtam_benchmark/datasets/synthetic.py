from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from s4dtam_benchmark.contracts import SequenceData
from s4dtam_benchmark.datasets.base import DatasetAdapter


class SyntheticDataset(DatasetAdapter):
    """Small deterministic dataset for CI; never use it as scientific evidence."""

    def __init__(self, seed: int = 7, length: int = 240):
        self.seed = seed
        self.length = length

    def sequences(self) -> Iterator[SequenceData]:
        rng = np.random.default_rng(self.seed)
        t = np.arange(self.length, dtype=float) * 0.05
        gt = np.column_stack((0.6 * t, np.sin(0.35 * t), 0.2 * np.cos(0.2 * t)))
        observations = gt + rng.normal(0.0, 0.06, gt.shape)
        semantic_gt = ((np.arange(self.length) // 30) % 4).astype(int)
        semantic_observations = semantic_gt.copy()
        corrupted = rng.random(self.length) < 0.08
        semantic_observations[corrupted] = rng.integers(0, 4, int(np.sum(corrupted)))
        phase = np.arange(self.length)[:, None] + np.arange(24)[None, :] * 3
        occupancy_observations = ((phase // 17) % 5 == 0).astype(float)
        occupancy_future = np.roll(occupancy_observations, -20, axis=0)
        yield SequenceData(
            dataset="synthetic",
            sequence_id="ci_curve_001",
            timestamps=t,
            gt_positions=gt,
            observations=observations,
            semantic_observations=semantic_observations,
            semantic_gt=semantic_gt,
            occupancy_observations=occupancy_observations,
            occupancy_gt={1.0: occupancy_future},
            navigation_gt={"shortest_path_m": float(np.linalg.norm(np.diff(gt, axis=0), axis=1).sum())},
            metadata={"purpose": "software smoke test only"},
        )
