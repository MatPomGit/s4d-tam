from __future__ import annotations

import json
from pathlib import Path

import pytest

from s4dtam_benchmark.config import load_yaml
from s4dtam_benchmark.datasets import TartanAirDataset
from s4dtam_benchmark.readiness import validate_readiness_matrix


def test_repository_readiness_matrix_is_valid() -> None:
    config = load_yaml(Path("configs/readiness/dataset_baseline_matrix.yaml"))
    summary = validate_readiness_matrix(config)
    assert summary.supported_pairs == 0
    assert summary.blocked_pairs == 6
    assert summary.not_applicable_pairs == 10
    assert summary.publication_ready is False


def test_supported_pair_requires_declared_sensors() -> None:
    config = load_yaml(Path("configs/readiness/dataset_baseline_matrix.yaml"))
    config["matrix"]["tartanair"]["vins_mono"] = {"state": "supported"}
    with pytest.raises(ValueError, match="missing sensors: imu"):
        validate_readiness_matrix(config)


def test_tartanair_preflight_accepts_strict_minimal_sequence(tmp_path: Path) -> None:
    sequence = tmp_path / "office_easy_p000"
    sequence.mkdir()
    for index in range(3):
        (sequence / f"rgb_{index:06d}.png").write_bytes(b"fixture")
    descriptor = {
        "id": "office_easy_p000",
        "frame_count": 3,
        "timestamp_unit": "s",
        "position_unit": "m",
        "frames": [
            {"index": 0, "timestamp": 0.0, "file": "rgb_000000.png"},
            {"index": 1, "timestamp": 0.1, "file": "rgb_000001.png"},
            {"index": 2, "timestamp": 0.2, "file": "rgb_000002.png"},
        ],
        "positions": [[1.0, 2.0, -3.0], [2.0, 3.0, -4.0], [3.0, 4.0, -5.0]],
        "quaternions_xyzw": [[0.0, 0.0, 0.0, 1.0]] * 3,
        "calibration": {
            "camera_matrix": [[320.0, 0.0, 160.0], [0.0, 320.0, 120.0], [0.0, 0.0, 1.0]],
            "image_size": [320, 240],
            "camera_frame": "camera_left",
            "body_frame": "body",
        },
    }
    (sequence / "sequence.json").write_text(json.dumps(descriptor), encoding="utf-8")

    loaded = list(TartanAirDataset(tmp_path).sequences())
    assert len(loaded) == 1
    assert loaded[0].sequence_id == "office_easy_p000"
    assert loaded[0].gt_positions.tolist()[0] == [2.0, 1.0, 3.0]
    assert loaded[0].metadata["axis_transform"] == "tartanair_ned_to_enu"


def test_tartanair_preflight_rejects_missing_frame(tmp_path: Path) -> None:
    sequence = tmp_path / "broken"
    sequence.mkdir()
    descriptor = {
        "id": "broken",
        "timestamp_unit": "s",
        "position_unit": "m",
        "frames": [{"index": 0, "timestamp": 0.0, "file": "missing.png"}],
        "positions": [[0.0, 0.0, 0.0]],
        "calibration": {
            "camera_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "image_size": [1, 1],
            "camera_frame": "camera",
            "body_frame": "body",
        },
    }
    (sequence / "sequence.json").write_text(json.dumps(descriptor), encoding="utf-8")

    with pytest.raises(ValueError, match="Missing TartanAir frame"):
        list(TartanAirDataset(tmp_path).sequences())
