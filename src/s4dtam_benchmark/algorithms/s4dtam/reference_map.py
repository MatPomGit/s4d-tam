"""Portable, versioned reference-map representation and coordinate transforms."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Mapping
from typing import Any

import numpy as np


MAP_SCHEMA = "s4dtam-reference-map/v1"


class ReferenceMapFormatError(ValueError):
    """Raised when a map artifact does not conform to the supported schema."""


def _json_value(value: Any) -> Any:
    """Convert common numeric containers without silently stringifying them."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"map metadata contains a non-JSON value: {type(value).__name__}")


def _vector(value: object, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector of length {size}")
    return result


@dataclass(frozen=True, slots=True)
class CoordinateFrame:
    """A named frame whose homogeneous transform maps points into the map frame."""

    name: str
    to_map: np.ndarray = field(default_factory=lambda: np.eye(4))

    def __post_init__(self) -> None:
        transform = np.asarray(self.to_map, dtype=float)
        if not self.name or transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError("coordinate frame needs a name and a finite 4x4 transform")
        if not np.allclose(transform[3], [0, 0, 0, 1]):
            raise ValueError("coordinate transform must be homogeneous")
        if abs(float(np.linalg.det(transform[:3, :3]))) < 1e-12:
            raise ValueError("coordinate transform must be invertible")
        object.__setattr__(self, "to_map", transform.copy())


@dataclass(frozen=True, slots=True)
class ReferenceToken:
    """Persistent map token with a place descriptor in a declared frame."""

    token_id: int
    position: np.ndarray
    descriptor: np.ndarray
    frame: str = "map"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.token_id, bool) or not isinstance(self.token_id, (int, np.integer)):
            raise TypeError("token_id must be an integer")
        if int(self.token_id) < 0:
            raise ValueError("token_id must be non-negative")
        object.__setattr__(self, "token_id", int(self.token_id))
        if not self.frame:
            raise ValueError("token frame must be non-empty")
        object.__setattr__(self, "position", _vector(self.position, 3, "position").copy())
        descriptor = np.asarray(self.descriptor, dtype=float)
        if descriptor.ndim != 1 or descriptor.size == 0 or not np.all(np.isfinite(descriptor)):
            raise ValueError("descriptor must be a non-empty finite vector")
        object.__setattr__(self, "descriptor", descriptor.copy())


@dataclass(slots=True)
class ReferenceMap:
    """Self-describing map artifact suitable for deterministic interchange."""

    tokens: list[ReferenceToken] = field(default_factory=list)
    coordinate_frames: dict[str, CoordinateFrame] = field(
        default_factory=lambda: {"map": CoordinateFrame("map")}
    )
    calibration: dict[str, Any] = field(default_factory=dict)
    origin: dict[str, Any] = field(default_factory=dict)
    build_metadata: dict[str, Any] = field(default_factory=dict)
    schema: str = MAP_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != MAP_SCHEMA:
            raise ReferenceMapFormatError(f"unsupported reference map schema: {self.schema}")
        if "map" not in self.coordinate_frames:
            raise ReferenceMapFormatError(
                "coordinate_frames must contain the canonical 'map' frame"
            )
        if not np.allclose(self.coordinate_frames["map"].to_map, np.eye(4), atol=1e-12):
            raise ReferenceMapFormatError("the canonical 'map' frame transform must be identity")
        if len({token.token_id for token in self.tokens}) != len(self.tokens):
            raise ValueError("reference token identifiers must be unique")
        unknown = {token.frame for token in self.tokens} - set(self.coordinate_frames)
        if unknown:
            raise ValueError(f"tokens use unknown coordinate frames: {sorted(unknown)}")

    def transform(self, points: np.ndarray, source: str, target: str = "map") -> np.ndarray:
        """Transform one XYZ point or an array of points between declared frames."""
        if source not in self.coordinate_frames or target not in self.coordinate_frames:
            raise KeyError(f"unknown coordinate frame: {source!r} or {target!r}")
        values = np.asarray(points, dtype=float)
        if values.shape[-1:] != (3,) or not np.all(np.isfinite(values)):
            raise ValueError("points must be finite with a final dimension of three")
        source_to_map = self.coordinate_frames[source].to_map
        map_to_target = np.linalg.inv(self.coordinate_frames[target].to_map)
        transform = map_to_target @ source_to_map
        flat = values.reshape(-1, 3)
        homogeneous = np.column_stack((flat, np.ones(len(flat))))
        return (homogeneous @ transform.T)[:, :3].reshape(values.shape)

    def token_positions(self, frame: str = "map") -> np.ndarray:
        return np.asarray(
            [self.transform(token.position, token.frame, frame) for token in self.tokens],
            dtype=float,
        ).reshape((-1, 3))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "tokens": [
                {"id": token.token_id, "position": token.position.tolist(),
                 "descriptor": token.descriptor.tolist(), "frame": token.frame,
                 "metadata": _json_value(token.metadata)}
                for token in self.tokens
            ],
            "coordinate_frames": {
                name: {"to_map": frame.to_map.tolist()}
                for name, frame in sorted(self.coordinate_frames.items())
            },
            "calibration": _json_value(self.calibration),
            "origin": _json_value(self.origin),
            "build_metadata": _json_value(self.build_metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReferenceMap":
        if not isinstance(payload, Mapping):
            raise ReferenceMapFormatError("reference map root must be an object")
        schema = payload.get("schema")
        if schema != MAP_SCHEMA:
            raise ReferenceMapFormatError(f"unsupported reference map schema: {schema!r}")
        raw_frames = payload.get("coordinate_frames")
        raw_tokens = payload.get("tokens")
        if not isinstance(raw_frames, Mapping) or not isinstance(raw_tokens, list):
            raise ReferenceMapFormatError(
                "coordinate_frames must be an object and tokens must be an array"
            )
        frames = {
            name: CoordinateFrame(name, value["to_map"])
            for name, value in raw_frames.items()
        }
        tokens = [
            ReferenceToken(item["id"], item["position"], item["descriptor"],
                           item.get("frame", "map"), item.get("metadata", {}))
            for item in raw_tokens
        ]
        return cls(tokens, frames, payload.get("calibration", {}), payload.get("origin", {}),
                   payload.get("build_metadata", {}), schema)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(destination)

    @classmethod
    def load(cls, path: str | Path) -> "ReferenceMap":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
