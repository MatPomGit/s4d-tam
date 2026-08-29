from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from s4dtam_benchmark.contracts import SequenceData
from s4dtam_benchmark.datasets.base import DatasetAdapter


class ManifestDataset(DatasetAdapter):
    """Loads the repository's dataset-neutral NPZ interchange format."""

    def __init__(self, name: str, root: str | Path, manifest: str | Path | None = None):
        self.name = name
        self.root = Path(root)
        self.manifest = Path(manifest) if manifest else self.root / "manifest.json"

    def sequences(self) -> Iterator[SequenceData]:
        if not self.manifest.exists():
            raise FileNotFoundError(
                f"Missing manifest: {self.manifest}. See docs/datasets.md for conversion."
            )
        spec = json.loads(self.manifest.read_text(encoding="utf-8"))
        for item in spec["sequences"]:
            path = self.root / item["file"]
            with np.load(path, allow_pickle=False) as data:
                horizons = item.get("occupancy_horizons_s", [])
                yield SequenceData(
                    dataset=self.name,
                    sequence_id=item["id"],
                    timestamps=data["timestamps"],
                    gt_positions=data["gt_positions"],
                    gt_quaternions=data["gt_quaternions"] if "gt_quaternions" in data else None,
                    observations=data["observations"] if "observations" in data else None,
                    semantic_observations=(
                        data["semantic_observations"] if "semantic_observations" in data else None
                    ),
                    semantic_gt=data["semantic_gt"] if "semantic_gt" in data else None,
                    occupancy_observations=(
                        data["occupancy_observations"]
                        if "occupancy_observations" in data
                        else None
                    ),
                    occupancy_gt={float(h): data[f"occupancy_gt_{h}"] for h in horizons},
                    risk_gt=data["risk_gt"] if "risk_gt" in data else None,
                    metadata=item.get("metadata", {}),
                )
