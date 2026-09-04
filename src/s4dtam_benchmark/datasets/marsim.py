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
        """Export samples in deterministic timestamp order."""
        records = list(samples)
        if not records:
            raise ValueError("MARSIM samples must not be empty")
        timestamps = np.asarray([record["timestamp"] for record in records], dtype=float)
        if np.any(~np.isfinite(timestamps)):
            raise ValueError("MARSIM timestamps must be finite")
        order = np.argsort(timestamps, kind="stable")
        ordered = [records[index] for index in order]
        timestamps = timestamps[order]
        if np.any(np.diff(timestamps) <= 0):
            raise ValueError("MARSIM timestamps must not contain duplicates")
        positions = np.asarray([record["position_m"] for record in ordered], dtype=float)
        if positions.shape != (len(ordered), 3):
            raise ValueError("MARSIM position_m must contain three coordinates")
        if np.any(~np.isfinite(positions)):
            raise ValueError("MARSIM position_m coordinates must be finite")
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
