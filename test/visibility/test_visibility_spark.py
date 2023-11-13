import unittest

from pyspark.sql import functions as F
from inclusive_search_fairness_measurement.rkl.wrapper import compute_rKL, compute_rKL_udf
from inclusive_search_fairness_measurement.visibility.wrapper import compute_visibility, compute_visibility_udf
from resources import sample_data_path
from test_pyspark import spark_session


search_identifiers = []




class PySparkTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = spark_session()

    @classmethod
    def tearDown(cls) -> None:
        cls.spark.stop()

    def test_individual_value(self):
        """
        Test visibility metric computation for one ranking
        """
        dataframe = self.spark.read.csv(sample_data_path, header=True)
        dataframe = dataframe.withColumn('is_sensitive', F.when(F.col("attribute_values") == "Women", 1).otherwise(0))
        dataframe = dataframe.withColumn(
            "asin_data",
            F.struct(F.col("position"), F.col("is_sensitive"), F.col("asin"))
        )
        dataframe = dataframe.groupby("keywords").agg(F.collect_list(F.col("asin_data")).alias("asin_data"))
        rows = dataframe.collect()
        res = compute_visibility(rows[1]["asin_data"])
        self.assertEqual(res, 1)

    def test_udf(self):
        """
        Test UDF function for visibility computation
        """
        dataframe = self.spark.read.csv(sample_data_path, header=True)
        dataframe = dataframe.withColumn('is_sensitive', F.when(F.col("attribute_values") == "Women", 1).otherwise(0))
        dataframe = dataframe.withColumn(
            "asin_data",
            F.struct(F.col("position"), F.col("is_sensitive"), F.col("asin"))
        )
        dataframe = dataframe.groupby("keywords").agg(F.collect_list(F.col("asin_data")).alias("asin_data"))
        dataframe = dataframe.withColumn("rKL", compute_visibility_udf("asin_data"))
        dataframe.show()
        rows = dataframe.collect()
        res = rows[1]["rKL"]
        self.assertEqual(res, 1)


if __name__ == "__main__":
    unittest.main()
