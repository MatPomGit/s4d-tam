from __future__ import annotations

import json
from pathlib import Path

import pytest

from s4dtam_benchmark.orb_slam3_tartanair import prepare_orb_slam3_tartanair_inputs


def _write_sequence(root: Path, *, declared_fps: float | None = 10.0) -> Path:
    sequence = root / "env__Easy__P000"
    frames = sequence / "frames"
    frames.mkdir(parents=True)
    (frames / "000000_left.png").write_bytes(b"frame-0")
    (frames / "000001_left.png").write_bytes(b"frame-1")

    provenance: dict[str, float] = {}
    if declared_fps is not None:
        provenance["declared_fps"] = declared_fps

    descriptor = {
        "schema": "s4dtam-tartanair-sequence/v1",
        "id": "env__Easy__P000",
        "frame_count": 2,
        "timestamp_unit": "s",
        "position_unit": "m",
        "frames": [
            {"index": 0, "timestamp": 0.0, "file": "frames/000000_left.png"},
            {"index": 1, "timestamp": 0.1, "file": "frames/000001_left.png"},
        ],
        "positions": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        "quaternions_xyzw": [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
        "calibration": {
            "camera_matrix": [
                [320.0, 0.0, 320.0],
                [0.0, 320.0, 240.0],
                [0.0, 0.0, 1.0],
            ],
            "image_size": [640, 480],
            "camera_frame": "camera_left",
            "body_frame": "body",
            "distortion_coefficients": [0.0, 0.0, 0.0, 0.0],
        },
        "provenance": provenance,
    }
    (sequence / "sequence.json").write_text(
        json.dumps(descriptor),
        encoding="utf-8",
    )
    return sequence


def test_prepare_orb_slam3_tartanair_inputs(tmp_path: Path) -> None:
    converted = tmp_path / "converted"
    output = tmp_path / "orb-inputs"
    _write_sequence(converted)

    summary = prepare_orb_slam3_tartanair_inputs(converted, output)

    assert summary.sequences == 1
    assert summary.frames == 2
    sequence_output = output / "env__Easy__P000"

    rgb_lines = (sequence_output / "rgb.txt").read_text(encoding="utf-8").splitlines()
    assert len(rgb_lines) == 5
    assert rgb_lines[3].startswith("0.000000000 ")
    assert rgb_lines[3].endswith("frames/000000_left.png")
    assert rgb_lines[4].startswith("0.100000000 ")
    assert rgb_lines[4].endswith("frames/000001_left.png")

    settings = (sequence_output / "settings.yaml").read_text(encoding="utf-8")
    assert 'File.version: "1.0"' in settings
    assert 'Camera.type: "PinHole"' in settings
    assert "Camera1.fx: 320.000000000" in settings
    assert "Camera1.fy: 320.000000000" in settings
    assert "Camera1.cx: 320.000000000" in settings
    assert "Camera1.cy: 240.000000000" in settings
    assert "Camera.fps: 10.000000000" in settings
    assert "Camera.width: 640" in settings
    assert "Camera.height: 480" in settings

    manifest = json.loads(
        (sequence_output / "input-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema"] == "s4dtam-orb-slam3-tartanair-input/v1"
    assert manifest["frame_count"] == 2
    assert manifest["fps"] == 10.0
    assert manifest["orb_slam3_mode"] == "monocular"
    assert len(manifest["source_sequence_descriptor_sha256"]) == 64
    assert len(manifest["rgb_list_sha256"]) == 64
    assert len(manifest["settings_sha256"]) == 64


def test_prepare_rejects_existing_destination_without_overwrite(tmp_path: Path) -> None:
    converted = tmp_path / "converted"
    output = tmp_path / "orb-inputs"
    _write_sequence(converted)
    prepare_orb_slam3_tartanair_inputs(converted, output)

    with pytest.raises(FileExistsError, match="use --overwrite"):
        prepare_orb_slam3_tartanair_inputs(converted, output)


def test_prepare_requires_declared_fps(tmp_path: Path) -> None:
    converted = tmp_path / "converted"
    output = tmp_path / "orb-inputs"
    _write_sequence(converted, declared_fps=None)

    with pytest.raises(ValueError, match="declared_fps"):
        prepare_orb_slam3_tartanair_inputs(converted, output)
