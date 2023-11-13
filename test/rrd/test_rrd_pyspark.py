import unittest

from pyspark.sql import functions as F
from inclusive_search_fairness_measurement.rrd.wrapper import compute_rRD, compute_rRD_udf
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
        Testing the rRD metric computation for only one ranking
        """
        dataframe = self.spark.read.csv(sample_data_path, header=True)
        dataframe = dataframe.withColumn('is_sensitive', F.when(F.col("attribute_values") == "Women", 1).otherwise(0))
        dataframe = dataframe.withColumn(
            "asin_data",
            F.struct(F.col("position"), F.col("is_sensitive"), F.col("asin"))
        )
        dataframe = dataframe.groupby("keywords").agg(F.collect_list(F.col("asin_data")).alias("asin_data"))
        rows = dataframe.collect()
        res = compute_rRD(rows[1]["asin_data"])
        print(res)
        self.assertAlmostEqual(res, 0.116, places=3)

    def test_udf(self):
        """
        Testing the UDF function for computing rRD
        """
        dataframe = self.spark.read.csv(sample_data_path, header=True)
        dataframe = dataframe.withColumn('is_sensitive', F.when(F.col("attribute_values") == "Women", 1).otherwise(0))
        dataframe = dataframe.withColumn(
            "asin_data",
            F.struct(F.col("position"), F.col("is_sensitive"), F.col("asin"))
        )
        dataframe = dataframe.groupby("keywords").agg(F.collect_list(F.col("asin_data")).alias("asin_data"))
        dataframe = dataframe.withColumn("rRD", compute_rRD_udf("asin_data"))
        dataframe.show()
        rows = dataframe.collect()
        res = rows[1]["rRD"]
        self.assertAlmostEqual(res, 0.116, places=3)


if __name__ == "__main__":
    unittest.main()
