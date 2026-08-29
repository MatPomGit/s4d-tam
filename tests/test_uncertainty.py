import unittest

import numpy as np

from s4dtam_benchmark.evaluation.uncertainty import (
    binary_risk_metrics,
    pose_uncertainty_metrics,
)


class UncertaintyMetricsTest(unittest.TestCase):
    def test_pose_coverage_for_zero_error(self):
        target = np.zeros((5, 3))
        covariance = np.repeat(np.eye(3)[None, :, :], 5, axis=0)
        metrics = pose_uncertainty_metrics(target, target, covariance)
        self.assertAlmostEqual(metrics["uncertainty/pose_nees_mean"], 0.0)
        self.assertAlmostEqual(metrics["uncertainty/pose_95pct_coverage"], 1.0)

    def test_risk_ranking(self):
        target = np.array([0, 0, 1, 1])
        score = np.array([0.1, 0.2, 0.8, 0.9])
        metrics = binary_risk_metrics(target, score)
        self.assertAlmostEqual(metrics["risk/auroc"], 1.0)
        self.assertAlmostEqual(metrics["risk/false_alarm_rate"], 0.0)
        self.assertAlmostEqual(metrics["risk/miss_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
