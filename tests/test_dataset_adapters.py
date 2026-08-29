from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from s4dtam_benchmark.datasets import (
    AeroVerseDataset, BlackbirdDataset, MARSIMDataset, MARSIMExporter, TartanAirDataset,
)
from s4dtam_benchmark.experiment import _dataset


def tartan_fixture(tmp_path: Path) -> tuple[Path, dict]:
    sequence = tmp_path / "easy"
    sequence.mkdir()
    for name in ("000.png", "001.png"):
        (sequence / name).write_bytes(b"png")
    spec = {"id": "easy", "timestamp_unit": "s", "position_unit": "m",
            "frames": [{"index": 0, "timestamp": 0.0, "file": "000.png"},
                       {"index": 1, "timestamp": 0.1, "file": "001.png"}],
            "positions": [[1, 2, 3], [4, 5, 6]],
            "calibration": {"camera_matrix": [1] * 9, "image_size": [1, 1],
                            "camera_frame": "camera", "body_frame": "body"}}
    (sequence / "sequence.json").write_text(json.dumps(spec), encoding="utf-8")
    return sequence / "sequence.json", spec


def test_tartanair_converts_and_transforms_axes(tmp_path):
    tartan_fixture(tmp_path)
    result = next(TartanAirDataset(tmp_path).sequences())
    np.testing.assert_array_equal(result.gt_positions[0], [2, 1, -3])
    assert result.sequence_id == "easy"


@pytest.mark.parametrize("mutation, message", [
    (lambda s: s["frames"].__setitem__(1, {**s["frames"][1], "index": 2}), "missing or out of order"),
    (lambda s: s["calibration"].pop("camera_matrix"), "Incomplete"),
    (lambda s: s["frames"][1].__setitem__("timestamp", 0.0), "strictly increasing"),
])
def test_tartanair_rejects_invalid_input(tmp_path, mutation, message):
    path, spec = tartan_fixture(tmp_path)
    mutation(spec)
    path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        next(TartanAirDataset(tmp_path).sequences())


def blackbird_topics():
    return {"/cam": [{"timestamp": 1.0, "data": [1]}, {"timestamp": 2.0, "data": [2]}],
            "/imu": [{"timestamp": 1.001, "data": [3]}, {"timestamp": 2.001, "data": [4]}],
            "/gt": [{"timestamp": 1.002, "position_m": [1, 2, 3]},
                    {"timestamp": 2.002, "position_m": [4, 5, 6]}]}


def test_blackbird_synchronizes_explicit_topics(tmp_path):
    bag = tmp_path / "flight.bag"
    bag.touch()
    result = next(BlackbirdDataset(tmp_path, topics={"camera": "/cam", "imu": "/imu",
        "ground_truth": "/gt"}, sync_tolerance_s=.005,
        bag_reader=lambda _: blackbird_topics()).sequences())
    np.testing.assert_array_equal(result.gt_positions[1], [4, 5, 6])


def test_blackbird_rejects_missing_topic_and_bad_time(tmp_path):
    (tmp_path / "flight.bag").touch()
    streams = blackbird_topics()
    streams.pop("/imu")
    adapter = BlackbirdDataset(tmp_path, topics={"camera": "/cam", "imu": "/imu",
        "ground_truth": "/gt"}, sync_tolerance_s=.005, bag_reader=lambda _: streams)
    with pytest.raises(ValueError, match="missing required topic"):
        next(adapter.sequences())
    streams = blackbird_topics()
    streams["/imu"][1]["timestamp"] = 1.0
    adapter.bag_reader = lambda _: streams
    with pytest.raises(ValueError, match="non-monotonic"):
        next(adapter.sequences())


def test_marsim_export_is_seeded_and_deterministic(tmp_path):
    samples = [{"timestamp": 2, "position_m": [2, 0, 0]},
               {"timestamp": 1, "position_m": [1, 0, 0]}]
    manifest = MARSIMExporter(tmp_path, seed=42).export(samples, simulator_version="abc")
    spec = json.loads(manifest.read_text())
    assert spec["random_seed"] == 42
    assert spec["sequences"][0]["file"] == "sequence_seed_0000000042_000000.npz"
    np.testing.assert_array_equal(next(MARSIMDataset(tmp_path).sequences()).timestamps, [1, 2])


def test_aeroverse_gates_license_version_and_completeness(tmp_path):
    np.savez(tmp_path / "seq.npz", timestamps=[0.0], gt_positions=[[0, 0, 0]])
    manifest = {"dataset_version": "v1", "license": {"id": "terms", "accepted": False},
                "sequences": [{"id": "seq", "file": "seq.npz"}]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    adapter = AeroVerseDataset(tmp_path, required_version="v1", accepted_license="terms")
    with pytest.raises(PermissionError, match="explicitly accepted"):
        next(adapter.sequences())
    manifest["license"]["accepted"] = True
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert next(adapter.sequences()).sequence_id == "seq"
    with pytest.raises(ValueError, match="release mismatch"):
        next(AeroVerseDataset(tmp_path, required_version="v2", accepted_license="terms").sequences())


def test_experiment_factory_registers_dataset_types(tmp_path):
    tartan_fixture(tmp_path)
    adapter = _dataset({"type": "tartanair", "root": str(tmp_path),
                        "axis_convention": "identity"}, seed=7)
    assert isinstance(adapter, TartanAirDataset)
