"""Deterministic MARSIM sample exporter and normalized adapter."""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from s4dtam_benchmark.datasets.manifest import ManifestDataset


class MARSIMDataset(ManifestDataset):
    def __init__(self, root: str | Path, manifest: str | Path | None = None):
        super().__init__("marsim", root, manifest)


class MARSIMExporter:
    def __init__(self, output_root: str | Path, *, seed: int, axis_convention: str = "enu"):
        self.output_root = Path(output_root)
        self.seed = int(seed)
        self.axis_convention = axis_convention

    def export(self, samples: Iterable[dict[str, Any]], *, simulator_version: str) -> Path:
        """Export samples sorted by timestamp; seed controls stable tie ordering and names."""
        records = list(samples)
        rng = np.random.default_rng(self.seed)
        tie_breakers = rng.random(len(records))
        ordered = [record for _, _, record in sorted(
            zip((float(r["timestamp"]) for r in records), tie_breakers, records),
            key=lambda item: (item[0], item[1]))]
        timestamps = np.asarray([record["timestamp"] for record in ordered], float)
        if not len(timestamps) or np.any(~np.isfinite(timestamps)) or np.any(np.diff(timestamps) < 0):
            raise ValueError("MARSIM timestamps must be finite")
        positions = np.asarray([record["position_m"] for record in ordered], float)
        if positions.shape != (len(ordered), 3):
            raise ValueError("MARSIM position_m must contain three coordinates")
        self.output_root.mkdir(parents=True, exist_ok=True)
        filename = f"sequence_seed_{self.seed:010d}_000000.npz"
        np.savez(self.output_root / filename, timestamps=timestamps, gt_positions=positions,
                 observations=np.asarray([record.get("observation", []) for record in ordered]))
        manifest = {"dataset": "marsim", "dataset_version": simulator_version,
                    "random_seed": self.seed, "axis_convention": self.axis_convention,
                    "timestamp_unit": "s", "position_unit": "m",
                    "sequences": [{"id": f"seed_{self.seed:010d}_000000", "file": filename,
                                   "metadata": {"random_seed": self.seed}}]}
        path = self.output_root / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path
