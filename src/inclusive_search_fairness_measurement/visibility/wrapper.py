import numpy as np

from pyspark.sql.types import IntegerType
from pyspark.sql import functions as F
from inclusive_search_fairness_measurement.visibility.visibility_score_metric import visibility_score


def compute_visibility(asin_data, n=20) -> float:
    """
    Compute visibility for the given ranking

    :param asin_data: List of tuples (position, attribute, sensitive_value)
    :param n: Rank cut-off value
    :return: Visibility metric value
    """

    positions, is_sensitive, asin = zip(*asin_data)
    positions_np = np.array(positions)
    is_sensitive_np = np.array(is_sensitive)
    value = visibility_score(positions_np, is_sensitive_np, n)
    return int(value)


# UDF function for collective computation
compute_visibility_udf = F.udf(compute_visibility, returnType=IntegerType())