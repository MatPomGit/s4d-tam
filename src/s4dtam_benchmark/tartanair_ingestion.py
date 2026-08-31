"""Reproducible ingestion and freeze utilities for TartanAir V1-style trajectories."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from s4dtam_benchmark.datasets import TartanAirDataset

_IMAGE_RE = re.compile(r"^(\d+)_left\.png$")

# TartanAir V1 pinhole calibration published by the dataset authors.
_TARTANAIR_V1_CALIBRATION = {
    "camera_matrix": [[320.0, 0.0, 320.0], [0.0, 320.0, 240.0], [0.0, 0.0, 1.0]],
    "image_size": [640, 480],
    "camera_frame": "camera_left",
    "body_frame": "body",
    "distortion_coefficients": [0.0, 0.0, 0.0, 0.0],
    "source": "TartanAir V1 published pinhole calibration",
}


@dataclass(frozen=True)
class TartanAirConversionSummary:
    sequences: int
    frames: int
    output_root: Path


@dataclass(frozen=True)
class TartanAirFreezeSummary:
    sequences: int
    frames: int
    files: int
    output_dir: Path
    manifest_sha256: str
    sequence_list_sha256: str


def discover_tartanair_v1_trajectories(raw_root: str | Path) -> list[Path]:
    """Find trajectory directories containing the canonical V1 pose and left RGB streams."""
    root = Path(raw_root)
    if not root.is_dir():
        raise FileNotFoundError(f"TartanAir raw root does not exist: {root}")
    trajectories = [
        pose.parent
        for pose in sorted(root.rglob("pose_left.txt"))
        if (pose.parent / "image_left").is_dir()
    ]
    if not trajectories:
        raise FileNotFoundError(
            f"No TartanAir V1 trajectories with pose_left.txt and image_left found below {root}"
        )
    return trajectories


def convert_tartanair_v1(
    raw_root: str | Path,
    output_root: str | Path,
    *,
    fps: float = 10.0,
    link_mode: str = "symlink",
    overwrite: bool = False,
) -> TartanAirConversionSummary:
    """Convert V1-style trajectories into the strict one-level ``sequence.json`` contract.

    Raw TartanAir V1 trajectories do not carry per-frame timestamps in the canonical pose file.
    Timestamps are therefore generated deterministically from the frame index and the explicitly
    recorded sampling rate. The default is 10 Hz, but callers can override it and the assumption is
    persisted in every descriptor.
    """
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("TartanAir fps must be finite and positive")
    if link_mode not in {"symlink", "hardlink", "copy"}:
        raise ValueError("link_mode must be one of: symlink, hardlink, copy")

    raw = Path(raw_root)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)

    total_frames = 0
    trajectories = discover_tartanair_v1_trajectories(raw)
    for trajectory in trajectories:
        relative = trajectory.relative_to(raw)
        sequence_id = "__".join(relative.parts)
        destination = output / sequence_id
        if destination.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Converted TartanAir sequence already exists: {destination}; use --overwrite"
                )
            shutil.rmtree(destination)
        frames_dir = destination / "frames"
        frames_dir.mkdir(parents=True)

        images = sorted((trajectory / "image_left").glob("*_left.png"))
        if not images:
            raise ValueError(f"No left RGB PNG frames found in {trajectory / 'image_left'}")
        indices = []
        for image in images:
            match = _IMAGE_RE.match(image.name)
            if match is None:
                raise ValueError(f"Unexpected TartanAir left-image filename: {image.name}")
            indices.append(int(match.group(1)))
        if indices != list(range(len(images))):
            raise ValueError(f"TartanAir frame indices are missing or out of order in {trajectory}")

        poses = np.loadtxt(trajectory / "pose_left.txt", dtype=float)
        if poses.ndim == 1:
            poses = poses.reshape(1, -1)
        if poses.shape != (len(images), 7):
            raise ValueError(
                "TartanAir pose/image count mismatch or invalid pose format in "
                f"{trajectory}: poses={poses.shape}, images={len(images)}"
            )
        if np.any(~np.isfinite(poses)):
            raise ValueError(f"TartanAir pose file contains non-finite values: {trajectory}")
        quaternion_norms = np.linalg.norm(poses[:, 3:7], axis=1)
        if np.any(np.abs(quaternion_norms - 1.0) > 1e-3):
            raise ValueError(f"TartanAir pose file contains non-unit quaternions: {trajectory}")

        frame_entries = []
        for index, source in enumerate(images):
            target = frames_dir / source.name
            _materialize_frame(source, target, link_mode)
            frame_entries.append(
                {
                    "index": index,
                    "timestamp": index / fps,
                    "file": f"frames/{source.name}",
                }
            )

        descriptor = {
            "schema": "s4dtam-tartanair-sequence/v1",
            "id": sequence_id,
            "frame_count": len(images),
            "timestamp_unit": "s",
            "position_unit": "m",
            "frames": frame_entries,
            "positions": poses[:, :3].tolist(),
            "quaternions_xyzw": poses[:, 3:7].tolist(),
            "calibration": dict(_TARTANAIR_V1_CALIBRATION),
            "provenance": {
                "source_layout": "tartanair_v1",
                "source_relative_trajectory": relative.as_posix(),
                "pose_file": "pose_left.txt",
                "image_stream": "image_left",
                "timestamp_source": "frame_index_divided_by_declared_fps",
                "declared_fps": float(fps),
                "materialization": link_mode,
                "source_coordinate_frame": "NED",
                "adapter_coordinate_transform": "tartanair_ned_to_enu",
            },
        }
        (destination / "sequence.json").write_text(
            json.dumps(descriptor, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        total_frames += len(images)

    # Validate the complete generated cohort with the same adapter used by the benchmark.
    list(TartanAirDataset(output).sequences())
    return TartanAirConversionSummary(
        sequences=len(trajectories), frames=total_frames, output_root=output
    )


def freeze_tartanair_cohort(
    converted_root: str | Path,
    output_dir: str | Path,
) -> TartanAirFreezeSummary:
    """Validate and hash a converted TartanAir cohort into immutable evidence files."""
    root = Path(converted_root)
    destination = Path(output_dir)
    sequences = list(TartanAirDataset(root).sequences())
    descriptors = sorted(root.glob("*/sequence.json"))
    if len(descriptors) != len(sequences):
        raise ValueError("TartanAir descriptor discovery does not match validated sequence count")

    destination.mkdir(parents=True, exist_ok=True)
    sequence_ids = [sequence.sequence_id for sequence in sequences]
    sequence_list_text = "".join(f"{sequence_id}\n" for sequence_id in sequence_ids)
    sequence_list_path = destination / "sequence-list.txt"
    sequence_list_path.write_text(sequence_list_text, encoding="utf-8")
    sequence_list_hash = _sha256_bytes(sequence_list_text.encode("utf-8"))

    files: dict[str, Path] = {}
    frame_count = 0
    for descriptor in descriptors:
        spec = json.loads(descriptor.read_text(encoding="utf-8"))
        rel_descriptor = descriptor.relative_to(root).as_posix()
        files[rel_descriptor] = descriptor
        for frame in spec["frames"]:
            path = descriptor.parent / str(frame["file"])
            if not path.is_file():
                raise ValueError(f"Missing TartanAir frame while freezing cohort: {path}")
            rel = path.relative_to(root).as_posix()
            files[rel] = path
            frame_count += 1

    manifest_lines = [f"{_sha256_file(path)}  {relative}\n" for relative, path in sorted(files.items())]
    manifest_text = "".join(manifest_lines)
    manifest_path = destination / "files.sha256"
    manifest_path.write_text(manifest_text, encoding="utf-8")
    manifest_hash = _sha256_bytes(manifest_text.encode("utf-8"))

    freeze = {
        "schema": "s4dtam-tartanair-freeze/v1",
        "dataset": "tartanair",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "converted_root": str(root),
        "sequence_count": len(sequences),
        "frame_count": frame_count,
        "file_count": len(files),
        "sequence_list": sequence_list_path.name,
        "sequence_list_sha256": sequence_list_hash,
        "file_manifest": manifest_path.name,
        "file_manifest_sha256": manifest_hash,
    }
    (destination / "freeze.json").write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return TartanAirFreezeSummary(
        sequences=len(sequences),
        frames=frame_count,
        files=len(files),
        output_dir=destination,
        manifest_sha256=manifest_hash,
        sequence_list_sha256=sequence_list_hash,
    )


def _materialize_frame(source: Path, target: Path, mode: str) -> None:
    if mode == "copy":
        shutil.copy2(source, target)
    elif mode == "hardlink":
        os.link(source, target)
    else:
        relative = os.path.relpath(source, start=target.parent)
        target.symlink_to(relative)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
