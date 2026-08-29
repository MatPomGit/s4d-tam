import unittest

import numpy as np

from s4dtam_benchmark.evaluation.trajectory import align_se3, trajectory_metrics


class TrajectoryMetricsTest(unittest.TestCase):
    def test_rigid_alignment_removes_rotation_and_translation(self):
        reference = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=float)
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        estimate = (rotation @ reference.T).T + np.array([3, -2, 1])
        aligned = align_se3(reference, estimate)
        np.testing.assert_allclose(aligned, reference, atol=1e-10)
        self.assertLess(trajectory_metrics(reference, estimate)["trajectory/ate_rmse_m"], 1e-10)

    def test_scale_error_is_not_hidden(self):
        reference = np.column_stack((np.arange(8), np.arange(8) ** 2, np.zeros(8))).astype(float)
        estimate = reference * 1.3
        self.assertGreater(trajectory_metrics(reference, estimate)["trajectory/ate_rmse_m"], 0.1)


if __name__ == "__main__":
    unittest.main()
