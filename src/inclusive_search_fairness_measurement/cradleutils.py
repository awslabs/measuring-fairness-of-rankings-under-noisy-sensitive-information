# This script contains functions used for logging in the cradle jobs
import datetime

from pyspark.sql import DataFrame


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