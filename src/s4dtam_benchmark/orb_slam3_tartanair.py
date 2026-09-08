"""Prepare deterministic ORB-SLAM3 monocular inputs from converted TartanAir data."""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ORBSLAM3InputSummary:
    """Summary of a prepared TartanAir-to-ORB-SLAM3 input cohort."""

    sequences: int
    frames: int
    output_root: Path


def prepare_orb_slam3_tartanair_inputs(
    converted_root: str | Path,
    output_root: str | Path,
    *,
    overwrite: bool = False,
) -> ORBSLAM3InputSummary:
    """Create TUM-style monocular input lists and calibrated ORB-SLAM3 settings.

    The official ORB-SLAM3 ``mono_tum`` example consumes a sequence directory
    containing ``rgb.txt`` plus a separate OpenCV YAML settings file. This
    adapter derives both files exclusively from the frozen ``sequence.json``
    contract produced by the S4D-TAM TartanAir converter. No camera parameters
    are guessed or copied from TUM/EuRoC examples.
    """
    source_root = Path(converted_root)
    destination_root = Path(output_root)
    descriptors = sorted(source_root.glob("*/sequence.json"))
    if not descriptors:
        raise FileNotFoundError(f"No converted TartanAir sequence.json below {source_root}")

    destination_root.mkdir(parents=True, exist_ok=True)
    total_frames = 0
    for descriptor in descriptors:
        spec: dict[str, Any] = json.loads(descriptor.read_text(encoding="utf-8"))
        sequence_id = str(spec.get("id", descriptor.parent.name))
        frames = spec.get("frames")
        if not isinstance(frames, list) or not frames:
            raise ValueError(f"TartanAir sequence {sequence_id} has no frames")

        destination = destination_root / sequence_id
        if destination.exists():
            if not overwrite:
                raise FileExistsError(
                    f"ORB-SLAM3 input directory already exists: {destination}; use --overwrite"
                )
            shutil.rmtree(destination)
        destination.mkdir(parents=True)

        calibration = _validated_calibration(spec, sequence_id)
        fps = _declared_fps(spec, sequence_id)
        rgb_lines = [
            "# TartanAir RGB stream prepared for ORB-SLAM3 mono_tum\n",
            f"# sequence: {sequence_id}\n",
            "# timestamp rgb_file\n",
        ]
        for frame in frames:
            timestamp = float(frame["timestamp"])
            if not math.isfinite(timestamp):
                raise ValueError(f"Non-finite timestamp in TartanAir sequence {sequence_id}")
            frame_path = descriptor.parent / str(frame["file"])
            if not frame_path.is_file():
                raise ValueError(f"Missing TartanAir frame for ORB-SLAM3: {frame_path}")
            relative = os.path.relpath(frame_path, start=destination).replace(os.sep, "/")
            rgb_lines.append(f"{timestamp:.9f} {relative}\n")

        rgb_path = destination / "rgb.txt"
        settings_path = destination / "settings.yaml"
        rgb_path.write_text("".join(rgb_lines), encoding="utf-8")
        settings_path.write_text(
            _render_settings(calibration, fps),
            encoding="utf-8",
        )

        manifest = {
            "schema": "s4dtam-orb-slam3-tartanair-input/v1",
            "dataset": "tartanair",
            "sequence_id": sequence_id,
            "source_sequence_descriptor": os.path.relpath(descriptor, start=destination).replace(
                os.sep, "/"
            ),
            "source_sequence_descriptor_sha256": _sha256_file(descriptor),
            "frame_count": len(frames),
            "fps": fps,
            "orb_slam3_mode": "monocular",
            "orb_slam3_example": "Examples/Monocular/mono_tum",
            "rgb_list": rgb_path.name,
            "settings": settings_path.name,
            "rgb_list_sha256": _sha256_file(rgb_path),
            "settings_sha256": _sha256_file(settings_path),
        }
        (destination / "input-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        total_frames += len(frames)

    return ORBSLAM3InputSummary(
        sequences=len(descriptors),
        frames=total_frames,
        output_root=destination_root,
    )


def _validated_calibration(spec: dict[str, Any], sequence_id: str) -> dict[str, Any]:
    calibration = spec.get("calibration")
    if not isinstance(calibration, dict):
        raise ValueError(f"Missing TartanAir calibration for {sequence_id}")
    matrix = calibration.get("camera_matrix")
    size = calibration.get("image_size")
    distortion = calibration.get("distortion_coefficients", [0.0, 0.0, 0.0, 0.0])
    if (
        not isinstance(matrix, list)
        or len(matrix) != 3
        or any(not isinstance(row, list) or len(row) != 3 for row in matrix)
    ):
        raise ValueError(f"Invalid camera_matrix for TartanAir sequence {sequence_id}")
    if not isinstance(size, list) or len(size) != 2:
        raise ValueError(f"Invalid image_size for TartanAir sequence {sequence_id}")
    if not isinstance(distortion, list) or len(distortion) < 4:
        raise ValueError(f"Invalid distortion coefficients for TartanAir sequence {sequence_id}")
    numeric_values = [
        float(matrix[0][0]),
        float(matrix[1][1]),
        float(matrix[0][2]),
        float(matrix[1][2]),
        float(size[0]),
        float(size[1]),
        *[float(value) for value in distortion[:4]],
    ]
    if not all(math.isfinite(value) for value in numeric_values):
        raise ValueError(f"Non-finite camera calibration for TartanAir sequence {sequence_id}")
    return calibration


def _declared_fps(spec: dict[str, Any], sequence_id: str) -> float:
    provenance = spec.get("provenance")
    if not isinstance(provenance, dict) or "declared_fps" not in provenance:
        raise ValueError(f"Missing declared_fps for TartanAir sequence {sequence_id}")
    fps = float(provenance["declared_fps"])
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid declared_fps for TartanAir sequence {sequence_id}")
    return fps


def _render_settings(calibration: dict[str, Any], fps: float) -> str:
    matrix = calibration["camera_matrix"]
    distortion = calibration.get("distortion_coefficients", [0.0, 0.0, 0.0, 0.0])
    width, height = calibration["image_size"]
    return f'''%YAML:1.0
File.version: "1.0"
Camera.type: "PinHole"
Camera1.fx: {float(matrix[0][0]):.9f}
Camera1.fy: {float(matrix[1][1]):.9f}
Camera1.cx: {float(matrix[0][2]):.9f}
Camera1.cy: {float(matrix[1][2]):.9f}
Camera1.k1: {float(distortion[0]):.9f}
Camera1.k2: {float(distortion[1]):.9f}
Camera1.p1: {float(distortion[2]):.9f}
Camera1.p2: {float(distortion[3]):.9f}
Camera.fps: {fps:.9f}
Camera.RGB: 1
Camera.width: {int(width)}
Camera.height: {int(height)}
ORBextractor.nFeatures: 1000
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0
'''


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
