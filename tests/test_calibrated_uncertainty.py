from pathlib import Path

import numpy as np

from s4dtam_benchmark.algorithms.s4dtam import ModalityNoiseModel, S4DTAMReference
from s4dtam_benchmark.contracts import RunContext, SequenceData
from s4dtam_benchmark.evaluation.uncertainty import ood_metrics


def _sequence(observations: np.ndarray) -> SequenceData:
    count = len(observations)
    return SequenceData(
        dataset="synthetic",
        sequence_id="uncertainty",
        timestamps=np.arange(count, dtype=float),
        gt_positions=observations.copy(),
        observations=observations,
    )


def test_pipeline_covariances_are_dimensionally_valid_and_positive_definite(tmp_path: Path):
    sequence = _sequence(np.zeros((4, 3)))
    result = S4DTAMReference(event_logging=None).run(
        sequence, RunContext(tmp_path, 7, {})
    )

    assert result.pose_covariances.shape == (4, 3, 3)
    assert np.all(np.linalg.eigvalsh(result.pose_covariances) > 0)
    assert result.ood_scores.shape == (4,)


def test_sensor_degradation_increases_measurement_uncertainty():
    model = ModalityNoiseModel(modality_variances={"lidar": 0.02}, quality_power=2)

    nominal = model.covariance("lidar", quality=1.0, dt=0.1)
    degraded = model.covariance("lidar", quality=0.25, dt=0.1)

    assert np.trace(degraded) > np.trace(nominal)


def test_energy_score_detects_synthetic_distribution_shift():
    rng = np.random.default_rng(4)
    in_distribution = rng.normal(0, 1, size=(100, 3))
    shifted = rng.normal(5, 1, size=(100, 3))
    scores = np.mean(np.vstack((in_distribution, shifted)) ** 2, axis=1)
    labels = np.r_[np.zeros(100, dtype=int), np.ones(100, dtype=int)]

    metrics = ood_metrics(labels, scores)

    assert metrics["ood/auroc"] > 0.95
    assert metrics["ood/auprc"] > 0.95
