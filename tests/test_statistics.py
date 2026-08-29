import unittest

import numpy as np

from s4dtam_benchmark.evaluation.statistics import holm_adjust, paired_bootstrap


class StatisticsTest(unittest.TestCase):
    def test_paired_bootstrap_direction(self):
        candidate = np.array([1.0, 2.0, 3.0, 4.0])
        baseline = candidate + 1.0
        result = paired_bootstrap(candidate, baseline, resamples=1000)
        self.assertAlmostEqual(result["mean_difference"], -1.0)
        self.assertLess(result["ci95_high"], 0.0)

    def test_holm_adjustment_is_bounded(self):
        adjusted = holm_adjust([0.01, 0.03, 0.8])
        self.assertTrue(all(0 <= value <= 1 for value in adjusted))
        self.assertGreaterEqual(adjusted[0], 0.01)


if __name__ == "__main__":
    unittest.main()
