import unittest

from inclusive_search_fairness_measurement.rnd.wrapper import rNDMetric, rnd_factory


class TestrNDCorrection(unittest.TestCase):
    def test_simple_case_1(self):
        """
        Test correction according to Assumption I
        """
        rnd_obj = rnd_factory(0.1, 1)
        estimated_parity = 0.1
        q = p = 0.3
        s = 0.5
        corrected_value = rnd_obj.correct(estimated_parity, q, p, s)
        self.assertAlmostEqual(corrected_value, 0.25)

    def test_simple_case_2(self):
        """
        Test correction according to Assumption II
        """
        rnd_obj = rnd_factory(0.1, 2)
        estimated_parity = 0.1
        q = p = 0.3
        s = 0.5
        corrected_value = rnd_obj.correct(estimated_parity, q, p, s)
        self.assertAlmostEqual(corrected_value, 0.04)


if __name__ == "__main__":
    unittest.main()