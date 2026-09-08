from __future__ import annotations

import numpy as np

from s4dtam_benchmark.evaluation.trajectory import align_sim3, trajectory_metrics


def test_sim3_recovers_known_scale_rotation_and_translation() -> None:
    estimate = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 1.0]]
    )
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    scale = 2.5
    translation = np.array([4.0, -3.0, 2.0])
    reference = scale * (rotation @ estimate.T).T + translation

    aligned, recovered_scale = align_sim3(reference, estimate)

    np.testing.assert_allclose(aligned, reference, atol=1e-10)
    assert abs(recovered_scale - scale) < 1e-10
    metrics = trajectory_metrics(reference, estimate, alignment_mode="sim3")
    assert metrics["trajectory/ate_rmse_m"] < 1e-10
    assert abs(metrics["trajectory/alignment_scale"] - scale) < 1e-10


def test_se3_does_not_hide_monocular_scale_error() -> None:
    reference = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    estimate = reference / 2.0
    se3 = trajectory_metrics(reference, estimate, alignment_mode="se3")
    sim3 = trajectory_metrics(reference, estimate, alignment_mode="sim3")
    assert se3["trajectory/ate_rmse_m"] > 0.1
    assert sim3["trajectory/ate_rmse_m"] < 1e-10
