"""Strict adapters for immutable, containerised external SLAM baselines.

The ROS/C++ launchers write a small, method-independent NPZ file.  Keeping the
parser here (rather than in the evaluators) makes it impossible for a baseline
to receive method-specific trajectory post-processing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from s4dtam_benchmark.algorithms.base import AlgorithmAdapter
from s4dtam_benchmark.contracts import AlgorithmResult, RunContext, SequenceData


REQUIRED_ARRAYS = ("timestamps", "estimated_positions", "estimated_quaternions", "latency_ms")
REQUIRED_RESOURCE_KEYS = ("resource_peak_rss_mb", "resource_cpu_time_s")


@dataclass(frozen=True, slots=True)
class ExternalWrapper:
    """Reproducible launch contract consumed by the out-of-process runner."""

    name: str
    upstream: str
    commit: str
    container: str
    inputs: dict[str, str]
    calibration: tuple[str, ...]
    output_format: str = "s4dtam-algorithm-result-npz/v1"


WRAPPERS = {
    "orb_slam3": ExternalWrapper(
        "orb_slam3", "https://github.com/UZ-SLAMLab/ORB_SLAM3",
        "4452a3c4d6d9d7333b30f1e6f2e67a4311a7c991",
        "ghcr.io/s4d-tam/orb-slam3@sha256:5c02f44e8738891a64056c44fe26fb90d1cf48f44c0f9c340fad22c80f4f78fb",
        {"camera": "/camera/image_raw", "imu": "/imu"},
        ("camera_intrinsics", "camera_distortion", "T_camera_imu", "imu_noise"),
    ),
    "vins_mono": ExternalWrapper(
        "vins_mono", "https://github.com/HKUST-Aerial-Robotics/VINS-Mono",
        "90dabb5d09c326d23f83a1c2aa0e81f6f3f5ed12",
        "ghcr.io/s4d-tam/vins-mono@sha256:894f4600e68a7bc4a7821078b5e6758a421982daf99894d505ef474017efd648",
        {"camera": "/cam0/image_raw", "imu": "/imu0"},
        ("camera_intrinsics", "camera_distortion", "T_camera_imu", "imu_noise"),
    ),
    "fast_lio2": ExternalWrapper(
        "fast_lio2", "https://github.com/hku-mars/FAST_LIO",
        "7a7cf9b0df52c25f69e4c4f8e92d3552cbe59c29",
        "ghcr.io/s4d-tam/fast-lio2@sha256:29187acbe2282620ff673b241b3d2b5169279d53275aa000e791656c4fc95f10",
        {"lidar": "/points_raw", "imu": "/imu"},
        ("T_lidar_imu", "imu_noise", "lidar_model", "scan_period"),
    ),
    "lio_sam": ExternalWrapper(
        "lio_sam", "https://github.com/TixiaoShan/LIO-SAM",
        "a4f2af6c7b6f61d8c4b5176d81e9cafe22b72cc8",
        "ghcr.io/s4d-tam/lio-sam@sha256:21d018ffd37a6480b427ec8267fd8d42219298b059536991a927adcc42ebc9a7",
        {"lidar": "/points_raw", "imu": "/imu", "gps": "/gps/odom"},
        ("T_lidar_imu", "T_gps_imu", "imu_noise", "lidar_model"),
    ),
}


def _missing(data: np.lib.npyio.NpzFile, field: str) -> None:
    if field not in data.files:
        raise ValueError(f"Incomplete external artifact: missing field '{field}'")


def parse_external_artifact(path: str | Path, name: str) -> AlgorithmResult:
    """Parse and fully validate normalized NPZ output before evaluation."""
    with np.load(Path(path), allow_pickle=False) as data:
        for field in (*REQUIRED_ARRAYS, *REQUIRED_RESOURCE_KEYS):
            _missing(data, field)
        timestamps = np.asarray(data["timestamps"], dtype=np.float64)
        positions = np.asarray(data["estimated_positions"], dtype=np.float64)
        quaternions = np.asarray(data["estimated_quaternions"], dtype=np.float64)
        latency = np.asarray(data["latency_ms"], dtype=np.float64)
        if timestamps.ndim != 1:
            raise ValueError("Invalid field 'timestamps': expected shape [N]")
        n = timestamps.size
        if n == 0 or not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0):
            raise ValueError("Invalid field 'timestamps': values must be finite and strictly increasing")
        expected = {"estimated_positions": (n, 3), "estimated_quaternions": (n, 4),
                    "latency_ms": (n,)}
        for field, value in (("estimated_positions", positions),
                             ("estimated_quaternions", quaternions), ("latency_ms", latency)):
            if value.shape != expected[field]:
                raise ValueError(f"Invalid field '{field}': expected shape {expected[field]}, got {value.shape}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"Invalid field '{field}': values must be finite")
        norms = np.linalg.norm(quaternions, axis=1)
        if np.any(norms == 0):
            raise ValueError("Invalid field 'estimated_quaternions': zero-norm quaternion")
        quaternions = quaternions / norms[:, None]
        if np.any(latency < 0):
            raise ValueError("Invalid field 'latency_ms': values must be non-negative")
        resource = {key.removeprefix("resource_"): float(np.asarray(data[key]))
                    for key in data.files if key.startswith("resource_")}
        for field, value in resource.items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"Invalid field 'resource_{field}': expected non-negative scalar")
        return AlgorithmResult(
            algorithm=name, timestamps=timestamps, estimated_positions=positions,
            estimated_quaternions=quaternions, latency_ms=latency, resource=resource,
            metadata={"artifact_format": "s4dtam-algorithm-result-npz/v1"},
        )


class ExternalArtifactAlgorithm(AlgorithmAdapter):
    """Load a normalized output created by one of the pinned baseline wrappers."""

    def __init__(self, name: str, result_root: str | Path, config: dict[str, Any] | None = None):
        if name not in WRAPPERS:
            raise ValueError(f"No external wrapper definition for '{name}'")
        self.name = name
        self.result_root = Path(result_root)
        self.wrapper = WRAPPERS[name]
        self.config = config or {}

    def run(self, sequence: SequenceData, context: RunContext) -> AlgorithmResult:
        path = self.result_root / sequence.dataset / f"{sequence.sequence_id}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"No {self.name} artifact for {sequence.dataset}/{sequence.sequence_id}: {path}"
            )
        result = parse_external_artifact(path, self.name)
        result.metadata["wrapper"] = {
            "upstream": self.wrapper.upstream, "commit": self.wrapper.commit,
            "container": self.wrapper.container, "inputs": self.wrapper.inputs,
            "calibration": self.wrapper.calibration,
            "loop_closure": self.config.get("loop_closure"),
            "hardware": self.config.get("hardware"), "warm_up": self.config.get("warm_up"),
        }
        return result
