import unittest

import numpy as np

from s4dtam_benchmark.evaluation.semantic import semantic_metrics


class SemanticMetricsTest(unittest.TestCase):
    def test_known_confusion(self):
        target = np.array([0, 0, 1, 1])
        prediction = np.array([0, 1, 1, 1])
        metrics = semantic_metrics(target, prediction)
        self.assertAlmostEqual(metrics["semantic/accuracy"], 0.75)
        self.assertAlmostEqual(metrics["semantic/iou_class_0"], 0.5)
        self.assertAlmostEqual(metrics["semantic/iou_class_1"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
