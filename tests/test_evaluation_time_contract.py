import numpy as np
import pytest

from s4dtam_benchmark.contracts import AlgorithmResult, SequenceData
from s4dtam_benchmark.evaluation.runner import evaluate_result, validate_time_contract


def sequence() -> SequenceData:
    return SequenceData(
        dataset="clock-test",
        sequence_id="sequence-7",
        timestamps=np.array([0.0, 1.0, 2.0, 3.0]),
        gt_positions=np.column_stack((np.arange(4), np.zeros((4, 2)))),
    )


def result(timestamps=None) -> AlgorithmResult:
    times = np.array([0.0, 1.0, 2.0, 3.0]) if timestamps is None else timestamps
    count = len(times)
    return AlgorithmResult(
        algorithm="clock-algorithm",
        timestamps=times,
        estimated_positions=np.column_stack((np.arange(count), np.zeros((count, 2)))),
    )


def test_matching_time_axes_are_accepted() -> None:
    validate_time_contract(sequence(), result())


def test_shifted_time_axis_reports_context_and_maximum_difference() -> None:
    shifted = result(np.array([0.1, 1.1, 2.1, 3.1]))
    with pytest.raises(ValueError) as error:
        validate_time_contract(sequence(), shifted)

    message = str(error.value)
    assert "dataset='clock-test'" in message
    assert "sequence_id='sequence-7'" in message
    assert "algorithm='clock-algorithm'" in message
    assert "dataset_samples=4" in message
    assert "result_samples=4" in message
    assert "max_time_difference_s=0.1" in message


def test_different_sample_counts_are_rejected() -> None:
    with pytest.raises(ValueError, match=r"dataset_samples=4, result_samples=3"):
        validate_time_contract(sequence(), result(np.array([0.0, 1.0, 2.0])))


@pytest.mark.parametrize(
    "timestamps",
    [np.array([0.0, 1.0, 1.0]), np.array([0.0, 2.0, 1.0])],
    ids=["duplicate", "decreasing"],
)
def test_result_requires_strictly_increasing_timestamps(timestamps) -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        result(timestamps)


def test_result_rejects_an_empty_time_axis() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        result(np.array([]))


def test_shifted_identical_trajectory_is_rejected_before_zero_ate() -> None:
    shifted = result(np.array([0.1, 1.1, 2.1, 3.1]))
    np.testing.assert_array_equal(shifted.estimated_positions, sequence().gt_positions)

    with pytest.raises(ValueError, match="timestamp contract mismatch"):
        evaluate_result(sequence(), shifted)
