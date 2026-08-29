import unittest

import numpy as np

from s4dtam_benchmark.evaluation.forecast import occupancy_metrics
from s4dtam_benchmark.algorithms.s4dtam.forecasting import (
    CausalForecaster,
    target_indices,
)


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


class CausalForecastTest(unittest.TestCase):
    """Regression tests for causality, time alignment, and forecast distributions."""

    def test_future_frames_cannot_change_earlier_predictions(self):
        prefix = [np.eye(5)[1], np.eye(5)[2]]
        first = CausalForecaster([1.0])
        second = CausalForecaster([1.0])
        for index, frame in enumerate(prefix):
            first.update(float(index), frame)
            second.update(float(index), frame)
        first.update(2.0, np.eye(5)[3])
        second.update(2.0, np.eye(5)[0])
        np.testing.assert_array_equal(
            first.result().occupancy_probability[1.0][:2],
            second.result().occupancy_probability[1.0][:2],
        )

    def test_irregular_timestamps_use_physical_target_time(self):
        times = np.array([0.0, 0.1, 0.9, 1.05, 2.8])
        np.testing.assert_array_equal(target_indices(times, 1.0), [3, 4, 4, 4, -1])

        forecaster = CausalForecaster([1.0])
        for timestamp in times:
            forecaster.update(timestamp, np.zeros((4, 4)))
        result = forecaster.result()
        np.testing.assert_array_equal(result.target_indices[1.0], [3, 4, 4, 4, -1])
        self.assertFalse(np.any(result.observable_mask[1.0][-1:]))

    def test_static_and_moving_objects_at_multiple_horizons(self):
        forecaster = CausalForecaster([1.0, 2.0])
        for timestamp, column in enumerate([1, 2, 3]):
            frame = np.zeros((7, 7))
            frame[1, 1] = 1.0  # static object
            frame[4, column] = 1.0  # object moving one cell per second
            forecaster.update(float(timestamp), frame)
        result = forecaster.result()
        for horizon, expected_column in ((1.0, 4), (2.0, 5)):
            probability = result.occupancy_probability[horizon][-1]
            self.assertGreater(probability[4, expected_column], 0.5)
            self.assertGreater(probability[1, 1], 0.5)
            self.assertEqual(result.flow_mean[horizon].shape, (3, 7, 7, 2))
            self.assertEqual(result.occupancy_uncertainty[horizon].shape, (3, 7, 7))
            self.assertTrue(np.all(result.flow_mean[horizon][-1, 1, 1] == 0.0))
            probability = result.occupancy_probability[horizon]
            np.testing.assert_allclose(
                result.occupancy_uncertainty[horizon], probability * (1.0 - probability)
            )

    def test_invalid_configuration_and_observations_are_rejected(self):
        with self.assertRaises(ValueError):
            CausalForecaster([])
        with self.assertRaises(ValueError):
            CausalForecaster([float("nan")])
        with self.assertRaises(ValueError):
            target_indices(np.array([0.0, 0.0]), 1.0)

        forecaster = CausalForecaster([1.0])
        with self.assertRaises(RuntimeError):
            forecaster.result()
        with self.assertRaises(ValueError):
            forecaster.update(0.0, np.array([1.2]))


if __name__ == "__main__":
    unittest.main()
