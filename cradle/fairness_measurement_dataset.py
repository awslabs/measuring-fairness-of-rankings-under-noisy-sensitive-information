# This script must contain a Krait class named {scriptObjectName}, which contains a static
# method 'execute' that receives three inputs:
# - "spark", which is the pyspark SparkSession in which the code is running
# - "input", a krait Dict of NodeID (str) -> pyspark DataFrame holding all the input Datasets
# - "execParams", a krait Dict of str -> str that contains the profile variables
# 'execute' must return a pyspark DataFrame
import datetime

from pyspark.sql import functions as F, SparkSession
from pyspark.sql import DataFrame
from pyspark.sql.types import LongType


def tostring(dataframe: DataFrame, n=20, truncate=True, vertical=False):
    """Prints the first ``n`` rows of the dataframe to string.
    This function is based on http://spark.apache.org/docs/latest/api/python/_modules/pyspark/sql/dataframe.html#DataFrame.show
    """
    if isinstance(truncate, bool) and truncate:
        return dataframe._jdf.showString(n, 20, vertical)
    else:
        return dataframe._jdf.showString(n, int(truncate), vertical)


def cradlelog(msg: str):
    print("[CRADLELOG - {}] {}".format(datetime.datetime.now().isoformat(), msg), flush=True)


def datasetlog(dataframe: DataFrame, dataname: str):
    cradlelog(dataname)
    print(tostring(dataframe, 100, False), flush=True)


def get_tommy_asin(
    spark: SparkSession, marketplaceID: int, final_date: datetime.datetime, window_length: int = 1
) -> DataFrame:
    """
    Gets the tommy query groups dataset

    :param spark: Spark session for this cluster
    :param marketplaceID: ID of the markteplace
    :param final_date: Final date for the dataset dump.
    :param window_length: Returned data will cover dates from final_date - windw_length to final_date. Set to 1 if lower value is provided
    :return: Tommy query groups dataset
    """
    start_date = final_date - datetime.timedelta(days=window_length)
    tommy = spark.read.table("O_TOMMY_ASIN")
    tommy = tommy.filter(F.col("marketplace_id") == marketplaceID)
    tommy = tommy.filter(F.col("search_date") >= start_date.strftime("%Y-%m-%d"))
    tommy = tommy.filter(F.col("search_date") <= final_date.strftime("%Y-%m-%d"))
    return tommy


def filter_tommy(df: DataFrame) -> DataFrame:
    """
    Filters the tommy dataset, removing spam
    """
    cond_is_not_spam = df.is_spam_or_untrusted == 0
    cond_correct_platform = df.site_variant.isin(
        ["PC Browser", "Mobile Browser", "Mobile Application", "Tablet Browser"]
    )

    df = df.filter(cond_is_not_spam).filter(cond_correct_platform)
    return df


def get_browse_node_assignements(spark: SparkSession, regionID: int, marketplaceID: int) -> DataFrame:
    """
    Gets the O_ASIN_BROWSE_NODE_ASSGMNTS dataset based on Anes

    :param spark: Spark session for this cluster
    :param regionID: Id of the region for the choosen marketplace
    :param marketplaceID: Id of the marketplace
    :return: Browse node assignement dataset
    """
    bnass = (
        spark.read.table("O_ASIN_BROWSE_NODE_ASSGMNTS")
        .filter(F.col("region_id") == regionID)
        .filter(F.col("marketplace_id") == marketplaceID)
    )
    return bnass


# Main scrript for the targeting dataset
# Cradle profile: https://datacentral.a2z.com/cradle/#/Alster/profiles/e3c08c3d-4190-42c6-b465-fd8a8e685cb8
class Script:
    @staticmethod
    def execute(spark, input, execParams):
        # profile variables are accessible via the execParams argument and can be read with execParams["myVariable"]
        # new profile variables can be declared by including a comment in your script with the format below
        # for more information about using profile variables see https://w.amazon.com/index.php/Cradle/ScalaProfileVariables

        # attribute = execParams["attribute"]  # ${attribute}
        # marketplaceName = execParams["marketplaceName"]  # ${marketplaceName}
        tommyDatasetOffset = int(execParams["tommyDatasetOffset"])  # ${tommyDatasetOffset}
        tommyWindowLength = int(execParams["tommyWindowLength"])  # ${tommyWindowLength}
        marketplaceID = int(execParams["marketplaceID"])  # ${marketplaceID}
        regionID = int(execParams["regionID"])  # ${regionID}

        cradlelog("Execution params")
        cradlelog(execParams)

        # TOMMY Setup
        finish_date = datetime.datetime(
            int(execParams["datasetDate"][0:4]),
            int(execParams["datasetDate"][4:6]),
            int(execParams["datasetDate"][6:8]),
        ) - datetime.timedelta(days=tommyDatasetOffset)

        tommyAsin = get_tommy_asin(spark, marketplaceID, finish_date, tommyWindowLength)
        tommyAsin = filter_tommy(tommyAsin).select(
            [
                "asin",
                "position",
                "organic_position",
                "page",
                "num_clicks",
                "num_adds",
                "num_purchases",
                "num_searches",
                "keywords",
                "marketplace_id",
                "search_date",
                "session",
                "query_group_id",
                "request_id",
                "qid",
                "search_index",
                "active_refinement_count",
            ]
        )
        # tommyAsin = tommyAsin.sample(0.001)
        # datasetlog(tommyAsin, "tommyAsin")

        # Retain only unrefined sessions and products that were actually shown to the user
        tommyAsin = tommyAsin.filter(F.col("num_searches") >= 1)
        tommyAsin = tommyAsin.filter(F.col("active_refinement_count") == 0)

        # Retain only in-scope querie
        selectedQueries = spark.table("metricsByKeyword").select("keywords")
        tommyAsin = tommyAsin.join(selectedQueries, on="keywords", how="left_semi")

        # Join against browse noode assignements on asin
        browseAssignements = get_browse_node_assignements(spark, regionID, marketplaceID).select(
            [F.col("asin"), F.col("browse_node_id").cast(LongType())]
        )
        tommyAsinWBrowse = tommyAsin.join(browseAssignements, on="asin", how="left")
        # datasetlog(tommyAsinWBrowse, "tommyAsinWBrowse")

        # Join against browseNodeToAttribute on browse node
        browseNodeToAttribute = spark.read.table("browseNodeToAttribute").select(
            [F.col("attribute_value"), F.col("browse_node_id").cast(LongType())]
        )
        tommyAsinWithDepartments = tommyAsinWBrowse.join(browseNodeToAttribute, on="browse_node_id", how="left")
        # datasetlog(tommyAsinWithDepartments, "tommyAsinWithDepartments")

        # Aggregate back to the level of searches
        impressedAsinsDataset = tommyAsinWithDepartments.groupby(
            [
                "asin",
                "position",
                "organic_position",
                "page",
                "num_clicks",
                "num_adds",
                "num_purchases",
                "num_searches",
                "keywords",
                "marketplace_id",
                "search_date",
                "session",
                "query_group_id",
                "request_id",
                "qid",
                "search_index",
                "active_refinement_count",
            ]
        ).agg(F.collect_set(F.col("attribute_value")).alias("attribute_values"))
        # datasetlog(impressedAsinsDataset, "impressedAsinsDataset")
        return impressedAsinsDataset.coalesce(100).select(
            [
                F.col("asin"),
                F.col("position"),
                F.col("organic_position"),
                F.col("page"),
                F.concat_ws(",", F.col("attribute_values")),
                F.col("num_clicks"),
                F.col("num_adds"),
                F.col("num_purchases"),
                F.col("num_searches"),
                F.col("keywords"),
                F.col("marketplace_id"),
                F.col("search_date"),
                F.col("session"),
                F.col("query_group_id"),
                F.col("request_id"),
                F.col("qid"),
                F.col("search_index"),
                F.col("active_refinement_count"),
            ])

