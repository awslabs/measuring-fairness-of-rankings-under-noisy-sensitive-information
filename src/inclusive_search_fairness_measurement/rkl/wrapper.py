from pyspark.sql.types import DoubleType
from pyspark.sql import functions as F
import numpy as np

from inclusive_search_fairness_measurement.rkl.normalized_kl_divergence_metric import normalized_discounted_kl_divergence


def compute_rKL(asin_data) -> float:
    """
    Computes the rKL metric for the given ranking

    :param asin_data: List of tuples (position, attribute, sensitive_value)
    :return: rKL metric
    """

    positions, is_sensitive, asin = zip(*asin_data)
    positions_np = np.array(positions)
    is_sensitive_np = np.array(is_sensitive)
    value = normalized_discounted_kl_divergence(positions_np, is_sensitive_np, n=20,
                                                start=2, step=1, population_demo=[0.5, 0.5])
    return float(value)


# UDF function for collective computation
compute_rKL_udf = F.udf(compute_rKL, returnType=DoubleType())