from pathlib import Path

import numpy as np
import pytest

from s4dtam_benchmark.algorithms.external import WRAPPERS, parse_external_artifact


def artifact(path: Path, **updates):
    values = {
        "timestamps": np.array([1.0, 2.0]),
        "estimated_positions": np.zeros((2, 3)),
        "estimated_quaternions": np.array([[0, 0, 0, 2], [0, 0, 0, 1.0]]),
        "latency_ms": np.array([3.0, 4.0]),
        "resource_peak_rss_mb": np.array(128.0),
        "resource_cpu_time_s": np.array(2.5),
    }
    values.update(updates)
    np.savez(path, **values)


@pytest.mark.parametrize("name", sorted(WRAPPERS))
def test_minimal_artifact_parser_normalizes_every_wrapper(tmp_path, name):
    path = tmp_path / "result.npz"
    artifact(path)
    result = parse_external_artifact(path, name)
    assert result.algorithm == name
    assert result.estimated_positions.shape == (2, 3)
    np.testing.assert_allclose(np.linalg.norm(result.estimated_quaternions, axis=1), 1)
    np.testing.assert_array_equal(result.latency_ms, [3, 4])
    assert result.resource == {"peak_rss_mb": 128.0, "cpu_time_s": 2.5}


def test_parser_names_missing_required_field(tmp_path):
    path = tmp_path / "result.npz"
    artifact(path)
    with np.load(path) as source:
        np.savez(path, **{key: source[key] for key in source.files if key != "latency_ms"})
    with pytest.raises(ValueError, match="missing field 'latency_ms'"):
        parse_external_artifact(path, "orb_slam3")


@pytest.mark.parametrize("timestamps", [np.array([1.0, 1.0]), np.array([2.0, 1.0]), np.array([1.0, np.nan])])
def test_parser_rejects_incompatible_timestamps(tmp_path, timestamps):
    path = tmp_path / "result.npz"
    artifact(path, timestamps=timestamps)
    with pytest.raises(ValueError, match="timestamps"):
        parse_external_artifact(path, "lio_sam")


@pytest.mark.parametrize("field,value", [
    ("estimated_positions", np.zeros((2, 2))),
    ("estimated_quaternions", np.zeros((2, 3))),
    ("latency_ms", np.zeros((2, 1))),
])
def test_parser_rejects_incompatible_shapes(tmp_path, field, value):
    path = tmp_path / "result.npz"
    artifact(path, **{field: value})
    with pytest.raises(ValueError, match=f"Invalid field '{field}'"):
        parse_external_artifact(path, "fast_lio2")
