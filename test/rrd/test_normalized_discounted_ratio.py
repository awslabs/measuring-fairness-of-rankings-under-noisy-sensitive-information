import unittest
import numpy as np

from typing import Tuple
from inclusive_search_fairness_measurement.rrd.normalized_discounted_ratio_metric import normalized_discounted_ratio


def get_sample_ranking() -> Tuple[np.array, np.array]:
    """
    Prepare example dataset for reranking
    """

    positions = np.array([0, 1, 2, 3, 4, 5])
    is_sensitive = np.array([0, 1, 0, 1, 0, 1])

    return positions, is_sensitive


class TestrRD(unittest.TestCase):
    def test_invalid_input(self):
        with self.assertRaises(ValueError) as context:
            positions = np.array([0, 1, 2, 3, 4, 5])
            is_sensitive = np.array([0, 1, 0, 1, 0])
            normalized_discounted_ratio(positions, is_sensitive, n=6, start=2, step=2)
        self.assertTrue(
            "Length of input arrays are different, failed to continue" in str(context.exception)
        )

    def test_simple_case(self):
        positions, is_sensitive = get_sample_ranking()
        res = normalized_discounted_ratio(positions, is_sensitive, n=6, start=2, step=2)
        #print(res)
        self.assertEqual(res, 0.0)

    def test_failure(self):
        with self.assertRaises(ValueError) as context:
            positions = np.array([0, 1, 2, 3, 4, 5])
            is_sensitive = np.array([0, 1, 0, 1, 0, 1])
            normalized_discounted_ratio(positions, is_sensitive, n=1, start=2, step=2)
        self.assertTrue(
            "Invalid value for start or n, failed to continue" in str(context.exception)
        )
        with self.assertRaises(ValueError) as context:
            positions = np.array([0, 1, 2, 3, 4, 8])
            is_sensitive = np.array([1, 1, 1, 1, 1, 1])
            normalized_discounted_ratio(positions, is_sensitive, n=6, start=2, step=2)
        self.assertTrue(
            "All attribute values are the same, failed to continue" in str(context.exception)
        )
        with self.assertRaises(ValueError) as context:
            positions = np.array([0, 1, 2, 3, 4, 8])
            is_sensitive = np.array([0, 1, 0, 1, 0, 1])
            normalized_discounted_ratio(positions, is_sensitive, n=6, start=7, step=2)
        self.assertTrue(
            "Start value is larger than the largest rank, failed to continue" in str(context.exception)
        )
        with self.assertRaises(ValueError) as context:
            positions = np.array([0, 1, 2, 3, 4, 8])
            is_sensitive = np.array([0, 5, 0, 1, 0, 1])
            normalized_discounted_ratio(positions, is_sensitive, n=6, start=2, step=2)
        self.assertTrue(
            "is_sensitive variable must contain only 0 and 1, failed to continue" in str(context.exception)
        )
        with self.assertRaises(ValueError) as context:
            positions = np.array([0, 1, 2, 3, 4, 8])
            is_sensitive = np.array([1, 1, 0, 1, 0, 1])
            normalized_discounted_ratio(positions, is_sensitive, n=6, start=2, step=2)
        self.assertTrue(
            "Division by zero error, failed to compute the metric" in str(context.exception)
        )


if __name__ == "__main__":
    unittest.main()