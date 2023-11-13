import os
import subprocess
import sys
import unittest

from pyspark.sql import SparkSession

from resources import sample_data_path


def spark_session():
    """
    Fixture that enables running PySpark unit tests, by initializing a SparkSession and configuring it to run locally.
    """
    # Setup java home
    os.environ["JAVA_HOME"] = subprocess.run(["/usr/libexec/java_home", "-v", "1.8"],
                                             capture_output=True).stdout.decode().strip()
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

    # this fixes the pyspark python interpreter to the one used for running unit tests
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    spark = (
        SparkSession.builder.master("local")
            .config("spark.sql.shuffle.partitions", "1")
            .appName("TestSession")
            .getOrCreate()
    )
    return spark


class PySparkTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = spark_session()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.spark.stop()

    def test_dataread(self):
        dataframe = self.spark.read.csv(sample_data_path, header=True)
        self.assertEqual(dataframe.count(), 58)


if __name__ == "__main__":
    unittest.main()
