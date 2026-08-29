from __future__ import annotations

from pathlib import Path

import numpy as np

from s4dtam_benchmark.algorithms.base import AlgorithmAdapter
from s4dtam_benchmark.contracts import AlgorithmResult, RunContext, SequenceData


class ExternalArtifactAlgorithm(AlgorithmAdapter):
    """Loads normalized output created by an external ROS/C++ baseline wrapper."""

    def __init__(self, name: str, result_root: str | Path):
        self.name = name
        self.result_root = Path(result_root)

    def run(self, sequence: SequenceData, context: RunContext) -> AlgorithmResult:
        path = self.result_root / sequence.dataset / f"{sequence.sequence_id}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"No {self.name} artifact for {sequence.dataset}/{sequence.sequence_id}: {path}"
            )
        with np.load(path, allow_pickle=False) as data:
            occupancy = {
                float(key.removeprefix("occupancy_pred_")): data[key]
                for key in data.files
                if key.startswith("occupancy_pred_")
            }
            flow = {
                float(key.removeprefix("flow_pred_")): data[key]
                for key in data.files
                if key.startswith("flow_pred_")
            }
            return AlgorithmResult(
                algorithm=self.name,
                timestamps=data["timestamps"],
                estimated_positions=data["estimated_positions"],
                estimated_quaternions=(
                    data["estimated_quaternions"] if "estimated_quaternions" in data else None
                ),
                pose_covariances=(data["pose_covariances"] if "pose_covariances" in data else None),
                semantic_pred=data["semantic_pred"] if "semantic_pred" in data else None,
                occupancy_pred=occupancy,
                flow_pred=flow,
                risk_pred=data["risk_pred"] if "risk_pred" in data else None,
                latency_ms=data["latency_ms"] if "latency_ms" in data else None,
                resource={
                    key.removeprefix("resource_"): float(data[key])
                    for key in data.files
                    if key.startswith("resource_")
                },
                navigation={
                    key.removeprefix("navigation_"): float(data[key])
                    for key in data.files
                    if key.startswith("navigation_")
                },
            )
