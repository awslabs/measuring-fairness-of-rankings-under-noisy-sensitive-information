import unittest
import numpy as np

from typing import Tuple
from inclusive_search_fairness_measurement.visibility.visibility_score_metric import visibility_score

def get_sample_ranking() -> Tuple[np.array, np.array]:
    """
    Prepare example dataset for reranking
    """

    positions = np.array([0, 1, 2, 3, 4, 5])
    is_sensitive = np.array([1, 1, 0, 1, 0, 1])

    return positions, is_sensitive


class TestVisibilty(unittest.TestCase):
    def test_invalid_input(self):
        with self.assertRaises(ValueError) as context:
            positions = np.array([0, 1, 2, 3, 4, 5])
            is_sensitive = np.array([1, 1, 0, 1])
            visibility_score(positions, is_sensitive, n=2)
        self.assertTrue(
            "Length of input arrays are different, failed to continue" in str(context.exception)
        )

    def test_simple_case(self):
        positions, is_sensitive = get_sample_ranking()
        res = visibility_score(positions, is_sensitive, n=2)
        self.assertEqual(res, 1)


if __name__ == "__main__":
    unittest.main()