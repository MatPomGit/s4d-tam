"""Blackbird ROS bag adapter with explicit, unambiguous synchronization."""
from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from s4dtam_benchmark.contracts import SequenceData
from s4dtam_benchmark.datasets.base import DatasetAdapter

BagReader = Callable[[Path], Mapping[str, Sequence[Mapping[str, Any]]]]


class BlackbirdDataset(DatasetAdapter):
    """Synchronize camera, IMU, and ground truth streams to camera timestamps."""

    def __init__(self, root: str | Path, *, topics: Mapping[str, str],
                 sync_tolerance_s: float, bag_reader: BagReader | None = None,
                 axis_convention: str = "enu"):
        self.root = Path(root)
        self.topics = dict(topics)
        self.sync_tolerance_s = float(sync_tolerance_s)
        self.bag_reader = bag_reader or self._read_json_bag
        self.axis_convention = axis_convention
        if set(self.topics) != {"camera", "imu", "ground_truth"}:
            raise ValueError("Blackbird topics must configure camera, imu, and ground_truth")
        if self.sync_tolerance_s < 0:
            raise ValueError("Blackbird sync_tolerance_s must be non-negative")

    @staticmethod
    def _read_json_bag(path: Path) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """Read a dependency-free JSON extraction of a ROS bag (use a reader callback for .bag)."""
        if path.suffix != ".json":
            raise RuntimeError("Native ROS bag reading requires a configured bag_reader")
        return json.loads(path.read_text(encoding="utf-8"))["topics"]

    def sequences(self) -> Iterator[SequenceData]:
        bags = sorted((*self.root.glob("*.bag"), *self.root.glob("*.bag.json")))
        if not bags:
            raise FileNotFoundError(f"No Blackbird ROS bag found below {self.root}")
        for bag in bags:
            streams = self.bag_reader(bag)
            selected: dict[str, Sequence[Mapping[str, Any]]] = {}
            for role, topic in self.topics.items():
                if topic not in streams:
                    raise ValueError(f"Blackbird bag {bag.name} is missing required topic {topic}")
                selected[role] = streams[topic]
                times = np.asarray([message["timestamp"] for message in streams[topic]], float)
                if not len(times) or np.any(~np.isfinite(times)) or np.any(np.diff(times) <= 0):
                    raise ValueError(f"Blackbird topic {topic} is empty or non-monotonic")
            camera = selected["camera"]
            timestamps = np.asarray([m["timestamp"] for m in camera], float)
            matched = {role: self._match(timestamps, messages, role)
                       for role, messages in selected.items() if role != "camera"}
            positions = np.asarray([m["position_m"] for m in matched["ground_truth"]], float)
            if positions.shape != (len(camera), 3):
                raise ValueError("Blackbird ground-truth position_m must contain three coordinates")
            observations = np.asarray([[*np.asarray(cam.get("data", []), float).ravel(),
                                        *np.asarray(imu.get("data", []), float).ravel()]
                                       for cam, imu in zip(camera, matched["imu"], strict=True)])
            yield SequenceData(dataset="blackbird", sequence_id=bag.name.removesuffix(".bag.json"),
                               timestamps=timestamps, gt_positions=positions,
                               observations=observations,
                               metadata={"topics": self.topics, "sync_tolerance_s": self.sync_tolerance_s,
                                         "axis_convention": self.axis_convention})

    def _match(self, targets: np.ndarray, messages: Sequence[Mapping[str, Any]], role: str):
        source = np.asarray([m["timestamp"] for m in messages], float)
        result = []
        used: set[int] = set()
        for target in targets:
            distance = np.abs(source - target)
            candidates = np.flatnonzero(distance <= self.sync_tolerance_s)
            if len(candidates) != 1:
                raise ValueError(f"Blackbird {role} synchronization is ambiguous or missing at {target}")
            index = int(candidates[0])
            if index in used:
                raise ValueError(f"Blackbird {role} synchronization would reuse a measurement")
            used.add(index)
            result.append(messages[index])
        return result
