import numpy as np

from pyspark.sql.types import DoubleType
from pyspark.sql import functions as F
from inclusive_search_fairness_measurement.rrd.normalized_discounted_ratio_metric import normalized_discounted_ratio


def compute_rRD(asin_data) -> float:
    """
    Computes rRD for the given ranking

    :param asin_data: List of tuples (position, attribute, sensitive_value)
    :return: rRD metric value
    """

    positions, is_sensitive, asin = zip(*asin_data)
    positions_np = np.array(positions)
    is_sensitive_np = np.array(is_sensitive)
    value = normalized_discounted_ratio(positions_np, is_sensitive_np, n=20,
                                        start=2, step=1, population_demo=[0.5, 0.5])
    return float(value)


# UDF function for collective computation
compute_rRD_udf = F.udf(compute_rRD, returnType=DoubleType())