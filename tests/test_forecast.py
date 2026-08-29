import unittest

import numpy as np

from s4dtam_benchmark.evaluation.forecast import occupancy_metrics


class ForecastMetricsTest(unittest.TestCase):
    def test_perfect_probabilities(self):
        target = np.array([[0, 1, 1, 0]], dtype=float)
        metrics = occupancy_metrics(target, target)
        self.assertAlmostEqual(metrics["iou"], 1.0, places=10)
        self.assertAlmostEqual(metrics["brier"], 0.0)
        self.assertLess(metrics["nll"], 1e-5)

    def test_calibration_penalizes_overconfidence(self):
        target = np.array([0, 1, 0, 1], dtype=float)
        calibrated = occupancy_metrics(target, np.array([0.1, 0.9, 0.1, 0.9]))
        wrong = occupancy_metrics(target, np.array([0.9, 0.1, 0.9, 0.1]))
        self.assertLess(calibrated["brier"], wrong["brier"])
        self.assertLess(calibrated["nll"], wrong["nll"])


if __name__ == "__main__":
    unittest.main()
