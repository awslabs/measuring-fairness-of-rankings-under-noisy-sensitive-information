import unittest
import numpy as np

from fairness_measurement_uncertainty.data_read.data_simulator import linear_increasing_exposure


class DataSimTest(unittest.TestCase):
    def test_exposure(self):
        n = 10
        s = 0.2
        a = linear_increasing_exposure(n, s)
        print(a)
        self.assertAlmostEqual(np.sum(a), n * s)


if __name__ == '__main__':
    unittest.main()
