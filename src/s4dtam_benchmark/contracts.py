from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class SequenceData:
    dataset: str
    sequence_id: str
    timestamps: np.ndarray
    gt_positions: np.ndarray
    gt_quaternions: np.ndarray | None = None
    observations: np.ndarray | None = None
    semantic_observations: np.ndarray | None = None
    semantic_gt: np.ndarray | None = None
    occupancy_observations: np.ndarray | None = None
    occupancy_gt: dict[float, np.ndarray] = field(default_factory=dict)
    flow_gt: dict[float, np.ndarray] = field(default_factory=dict)
    risk_gt: np.ndarray | None = None
    navigation_gt: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AlgorithmResult:
    algorithm: str
    timestamps: np.ndarray
    estimated_positions: np.ndarray
    estimated_quaternions: np.ndarray | None = None
    pose_covariances: np.ndarray | None = None
    semantic_pred: np.ndarray | None = None
    occupancy_pred: dict[float, np.ndarray] = field(default_factory=dict)
    flow_pred: dict[float, np.ndarray] = field(default_factory=dict)
    risk_pred: np.ndarray | None = None
    latency_ms: np.ndarray | None = None
    resource: dict[str, float] = field(default_factory=dict)
    navigation: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RunContext:
    output_dir: Path
    seed: int
    config: dict[str, Any]
