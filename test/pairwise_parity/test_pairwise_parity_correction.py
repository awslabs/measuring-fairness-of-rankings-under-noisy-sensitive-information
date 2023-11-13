import unittest

from inclusive_search_fairness_measurement.pairwise_parity.wrapper import dp_factory


class TestParityCorrection(unittest.TestCase):
    def test_simple_case_1(self):
        """
        Tests correction under Assumption I
        """
        dp_obj = dp_factory(1)
        estimated_parity = 0.1
        q = p = 0.3
        s = 0.5
        corrected_value = dp_obj.correct(estimated_parity, q, p, s)
        self.assertAlmostEqual(corrected_value, 0.25)

    def test_simple_case_2(self):
        """
        Tests correction under Assumption II
        """
        dp_obj = dp_factory(2)
        estimated_parity = 0.1
        q = p = 0.3
        s = 0.5
        corrected_value = dp_obj.correct(estimated_parity, q, p, s)
        self.assertAlmostEqual(corrected_value, 0.04)


if __name__ == "__main__":
    unittest.main()