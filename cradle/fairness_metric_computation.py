# This script must contain a Krait class named {scriptObjectName}, which contains a static
# method 'execute' that receives three inputs:
# - "spark", which is the pyspark SparkSession in which the code is running
# - "input", a krait Dict of NodeID (str) -> pyspark DataFrame holding all the input Datasets
# - "execParams", a krait Dict of str -> str that contains the profile variables
# 'execute' must return a pyspark DataFrame
from pyspark.sql import functions as F
from inclusive_search_fairness_measurement.cradleutils import cradlelog, datasetlog
from inclusive_search_fairness_measurement.rnd.wrapper import compute_rND_udf
from inclusive_search_fairness_measurement.rkl.wrapper import compute_rKL_udf
from inclusive_search_fairness_measurement.rrd.wrapper import compute_rRD_udf


class Script:
    """
    Main script for computing metrics on fairness measurement dataset (Tommy+Browse+Attribute)
    """

    @staticmethod
    def execute(spark, input, execParams):
        # profile variables are accessible via the execParams argument and can be read with execParams["myVariable"]
        # new profile variables can be declared by including a comment in your script with the format below
        # for more information about using profile variables see https://w.amazon.com/index.php/Cradle/ScalaProfileVariables
        cradlelog("Execution params")
        cradlelog(execParams)

        table_name = "Fairness_Measurement"
        tommyAsin = spark.read.table(table_name)

        # tommyAsin = spark.read.csv(sample_data_path, header=True)

        # Filter by data
        tommyAsin = tommyAsin.filter(F.col("page") == 1)

        departments = ["Men", "Women"]
        # Adding a boolean column showing if the attribute is sensitive
        for department in departments:
            tommyAsin = tommyAsin.withColumn(f"is_{department}",
                        F.when(F.col("concat_ws(,, attribute_values)").contains(department), 1).otherwise(0))

            # Adding asin_data column
            tommyAsin = tommyAsin.withColumn(f"asin_data_{department}",
                                             F.struct(F.col("position"), F.col(f"is_{department}"), F.col("asin")))

        # Grouping by search identifiers
        key_columns = ["keywords", "marketplace_id", "search_date", "session", "query_group_id", "request_id", "qid"]
        aggregates = [F.collect_list(F.col(f"asin_data_{department}")).alias(f"asin_data_{department}") for department
                      in departments]
        tommyAsin = tommyAsin.groupby(key_columns).agg(*aggregates)

        for department in departments:
            # Compute rND metric
            tommyAsin = tommyAsin.withColumn(f"rND_{department}", compute_rND_udf(f"asin_data_{department}"))

            # Compute rKL
            tommyAsin = tommyAsin.withColumn(f"rKL_{department}", compute_rKL_udf(f"asin_data_{department}"))

            # Compute rRD
            tommyAsin = tommyAsin.withColumn(f"rRD_{department}", compute_rRD_udf(f"asin_data_{department}"))

        datasetlog(tommyAsin, "tommyAsinMetrics")

        # Return tommyAsin
        return tommyAsin.coalesce(100)
