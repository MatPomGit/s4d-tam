"""Strict converter/adapter for a downloaded TartanAir sequence."""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from s4dtam_benchmark.contracts import SequenceData
from s4dtam_benchmark.datasets.base import DatasetAdapter

_REQUIRED_CALIBRATION = {"camera_matrix", "image_size", "camera_frame", "body_frame"}
_AXIS_TRANSFORMS = {
    "tartanair_ned_to_enu": np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=float),
    "identity": np.eye(3),
}


class TartanAirDataset(DatasetAdapter):
    """Read raw sequences described by ``sequence.json`` without silently repairing data."""

    def __init__(self, root: str | Path, *, axis_convention: str = "tartanair_ned_to_enu"):
        self.root = Path(root)
        if axis_convention not in _AXIS_TRANSFORMS:
            raise ValueError(f"Unsupported TartanAir axis convention: {axis_convention}")
        self.axis_convention = axis_convention

    def sequences(self) -> Iterator[SequenceData]:
        descriptions = sorted(self.root.glob("*/sequence.json"))
        if not descriptions and (self.root / "sequence.json").exists():
            descriptions = [self.root / "sequence.json"]
        if not descriptions:
            raise FileNotFoundError(f"No TartanAir sequence.json found below {self.root}")
        for description in descriptions:
            yield self._read(description)

    def _read(self, description: Path) -> SequenceData:
        spec: dict[str, Any] = json.loads(description.read_text(encoding="utf-8"))
        base = description.parent
        frames = spec.get("frames", [])
        if "frame_count" in spec and int(spec["frame_count"]) != len(frames):
            raise ValueError("TartanAir declared frame_count does not match the frame list")
        indices = [int(frame["index"]) for frame in frames]
        if indices != list(range(len(frames))):
            raise ValueError("TartanAir frames are missing or out of order")
        for frame in frames:
            path = base / frame["file"]
            if not path.is_file():
                raise ValueError(f"Missing TartanAir frame: {path}")
        timestamps = np.asarray([frame["timestamp"] for frame in frames], dtype=float)
        if len(timestamps) == 0 or np.any(~np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0):
            raise ValueError("TartanAir timestamps must be finite and strictly increasing")
        calibration = spec.get("calibration", {})
        missing = sorted(_REQUIRED_CALIBRATION - calibration.keys())
        if missing:
            raise ValueError(f"Incomplete TartanAir calibration; missing: {', '.join(missing)}")
        if spec.get("timestamp_unit") != "s" or spec.get("position_unit") != "m":
            raise ValueError("TartanAir timestamp_unit='s' and position_unit='m' are required")
        positions = np.asarray(spec.get("positions"), dtype=float)
        if positions.shape != (len(frames), 3):
            raise ValueError("TartanAir positions must have shape [frame_count, 3]")
        positions = positions @ _AXIS_TRANSFORMS[self.axis_convention].T
        quaternions = spec.get("quaternions_xyzw")
        if quaternions is not None:
            quaternions = np.asarray(quaternions, dtype=float)
            if quaternions.shape != (len(frames), 4):
                raise ValueError("TartanAir quaternions_xyzw must have shape [frame_count, 4]")
        return SequenceData(
            dataset="tartanair", sequence_id=str(spec.get("id", base.name)),
            timestamps=timestamps, gt_positions=positions, gt_quaternions=quaternions,
            observations=np.asarray([str(base / f["file"]) for f in frames]),
            metadata={"calibration": calibration, "axis_transform": self.axis_convention,
                      "timestamp_unit": "s", "position_unit": "m"},
        )
