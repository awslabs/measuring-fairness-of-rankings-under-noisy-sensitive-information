import unittest
import numpy as np

from typing import Tuple
from inclusive_search_fairness_measurement.exposure.exposure_metric import ndcg_dropoff
from inclusive_search_fairness_measurement.exposure.wrapper import ExposureMetricFirstWorldview, \
    ExposureMetricSecondWorldview


def get_sample_ranking() -> Tuple[np.array, np.array]:
    """
    Prepare example dataset for metric computation
    """

    positions = np.array([0, 1, 2, 3, 4, 5])
    is_sensitive = np.array([0, 1, 0, 1, 0, 1])

    return positions, is_sensitive


class TestExposure(unittest.TestCase):
    def test_ndcg_dropoff(self):
        res = ndcg_dropoff(3)
        target = [0.469, 0.296, 0.235]
        np.testing.assert_almost_equal(res, target, decimal=3)
        self.assertAlmostEqual(np.sum(res), 1, delta=1e-5)

    def test_invalid_input(self):
        metric = ExposureMetricFirstWorldview()
        with self.assertRaises(ValueError) as context:
            positions = np.array([0, 1, 2, 3, 4, 5])
            is_sensitive = np.array([0, 1, 0, 1, 0])
            metric.compute(positions, is_sensitive)
        self.assertTrue(
            "Length of input arrays are different, failed to continue" in str(context.exception)
        )

    def test_simple_computation(self):
        metric = ExposureMetricFirstWorldview()
        positions, is_sensitive = get_sample_ranking()
        res = metric.compute(positions, is_sensitive)
        self.assertAlmostEqual(res, -0.1419, delta=1e-4)

        positions = np.array([0, 2, 1, 5, 3, 4])
        is_sensitive = np.array([0, 0, 1, 1, 1, 0])
        res = metric.compute(positions, is_sensitive)
        self.assertAlmostEqual(res, -0.1419, delta=1e-4)

    def test_simple_correction_1(self):
        """
        Tests exposure correction under Assumption I
        """
        metric = ExposureMetricFirstWorldview()
        positions, is_sensitive = get_sample_ranking()
        res = metric.compute(positions, is_sensitive)
        corr = metric.correct(res, 0.1, 0.1)
        self.assertAlmostEqual(corr, -0.1774, delta=1e-4)

    def test_simple_correction_2(self):
        """
        Tests exposure correction under Assumption II
        """
        metric = ExposureMetricSecondWorldview()
        positions, is_sensitive = get_sample_ranking()
        res = metric.compute(positions, is_sensitive)
        corr = metric.correct(res, 0.1, 0.1, 0.5)
        self.assertAlmostEqual(corr, -0.1135, delta=1e-4)


if __name__ == "__main__":
    unittest.main()