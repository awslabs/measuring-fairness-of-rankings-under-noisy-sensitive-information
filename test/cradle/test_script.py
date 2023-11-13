import unittest

from src.InclusiveSearchFairnessMeasurement.cradle.fairness_metric_computation import Script
from test_pyspark import spark_session


class PySparkTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = spark_session()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.spark.stop()

    def test_script(self):
        """
        Tests the script for metric computation
        """
        output = Script.execute(self.spark, None, None)
        output.show()
        # self.assertEqual(output.count(), 10)


if __name__ == "__main__":
    unittest.main()
