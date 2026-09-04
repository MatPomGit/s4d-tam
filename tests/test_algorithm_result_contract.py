import numpy as np
import pytest

from s4dtam_benchmark.contracts import AlgorithmResult, SequenceData
from s4dtam_benchmark.evaluation.runner import evaluate_result
from s4dtam_benchmark.evaluation.semantic import semantic_metrics


def make_result(**fields) -> AlgorithmResult:
    defaults = {
        "algorithm": "contract-test",
        "timestamps": [0.0, 1.0],
        "estimated_positions": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    }
    return AlgorithmResult(**(defaults | fields))


def test_valid_optional_fields_are_normalized() -> None:
    result = make_result(
        estimated_quaternions=[[0, 0, 0, 2], [0, 0, 1, 0]],
        semantic_pred=[1, 2],
        occupancy_pred={1: [[[0.2]], [[0.8]]]},
        flow_pred={1: [[[[0, 0]]], [[[1, 0]]]]},
        occupancy_uncertainty={1: [[[0.1]], [[0.1]]]},
        flow_uncertainty={1: [[[[0.1, 0.1]]], [[[0.2, 0.2]]]]},
        forecast_observable_mask={1: [[[True]], [[False]]]},
        risk_pred=[0.1, 0.9],
        latency_ms=[1, 2],
        resource={"rss_mb": 4},
    )
    np.testing.assert_allclose(np.linalg.norm(result.estimated_quaternions, axis=1), 1.0)
    assert result.latency_ms.dtype == float
    assert result.forecast_observable_mask[1.0].dtype == np.bool_


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("estimated_quaternions", np.ones((2, 1)), "shape"),
        ("estimated_quaternions", [[0, 0, 0, 0], [0, 0, 0, 1]], "zero-norm"),
        ("estimated_quaternions", [[0, 0, 0, np.nan], [0, 0, 0, 1]], "finite"),
        ("semantic_pred", [1], "first axis"),
        ("semantic_pred", [1, np.inf], "finite"),
        ("latency_ms", [[1], [2]], "shape"),
        ("latency_ms", [1], "shape"),
        ("latency_ms", [1, np.nan], "finite"),
        ("latency_ms", [1, np.inf], "finite"),
        ("latency_ms", [1, -1], "non-negative"),
        ("risk_pred", [[0.1], [0.2]], "shape"),
        ("risk_pred", [0.1, 1.1], r"\[0, 1\]"),
        ("occupancy_pred", {1: [[[0.1]], [[-0.1]]]}, r"\[0, 1\]"),
        ("occupancy_pred", {1: [[[0.1]]]}, "first axis"),
        ("flow_pred", {1: [[[0.0]], [[np.inf]]]}, "finite"),
        ("resource", {"rss": np.nan}, "finite"),
    ],
)
def test_invalid_optional_fields_are_rejected(field, value, message) -> None:
    with pytest.raises(ValueError, match=message):
        make_result(**{field: value})


def test_forecast_mask_shape_cannot_broadcast() -> None:
    with pytest.raises(ValueError, match="shape"):
        make_result(
            occupancy_pred={1: np.ones((2, 3, 4)) * 0.5},
            forecast_observable_mask={1: np.ones((2, 3, 1), dtype=bool)},
        )


def test_semantic_column_vector_mismatch_is_a_clear_error() -> None:
    with pytest.raises(ValueError, match="shapes must match exactly"):
        semantic_metrics(np.array([0, 1]), np.array([[0], [1]]))

    sequence = SequenceData(
        dataset="test",
        sequence_id="semantic",
        timestamps=np.array([0.0, 1.0]),
        gt_positions=np.array([[0, 0, 0], [1, 0, 0]]),
        semantic_gt=np.array([0, 1]),
    )
    with pytest.raises(
        ValueError, match="semantic target and prediction shapes must match exactly"
    ):
        evaluate_result(sequence, make_result(semantic_pred=np.array([[0], [1]])))
