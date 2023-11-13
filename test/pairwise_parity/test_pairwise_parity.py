import numpy as np
import unittest

from typing import Tuple
from inclusive_search_fairness_measurement.pairwise_parity.pairwise_parity_metric import pairwise_parity


def get_sample_ranking() -> Tuple[np.array, np.array]:
    """
    Prepare example dataset for metric computation
    """

    positions = np.array([0, 1, 2, 3, 4, 5])
    is_sensitive = np.array([0, 1, 0, 1, 0, 1])

    return positions, is_sensitive


class TestParity(unittest.TestCase):
    def test_invalid_input(self):
        with self.assertRaises(ValueError) as context:
            positions = np.array([0, 1, 2, 3, 4, 5])
            is_sensitive = np.array([0, 1, 0, 1, 0])
            pairwise_parity(positions, is_sensitive)
        self.assertTrue(
            "Length of input arrays are different, failed to continue" in str(context.exception)
        )

    def test_simple_cases(self):
        positions, is_sensitive = get_sample_ranking()
        res = pairwise_parity(positions, is_sensitive)
        print(res)
        self.assertEqual(res, 0.3333333333333333)

        positions = np.array([0, 1, 2, 3, 4, 8])
        is_sensitive = np.array([1, 1, 1, 1, 1, 1])
        res = pairwise_parity(positions, is_sensitive)
        self.assertEqual(res, 1.0)


if __name__ == "__main__":
    unittest.main()