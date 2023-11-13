import numpy as np

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from inclusive_search_fairness_measurement.rnd.normalized_discounted_difference_metric import \
    normalized_discounted_difference


class rNDMetric():
    full_name = "rND"

    def __init__(self, ratio: float = 0.1):
        self.ratio = ratio

    def compute(self, positions: np.array, is_sensitive: np.array, s) -> float:
        """
        Computes rND of a ranking list
        :param positions: Numpy array of positions
        :param is_sensitive: Numpy array of indicators for sensitive attribute values
        :param s: Population fraction of the protected class
        :return: rND value
        """
        rnd_n = int(positions.size * self.ratio)
        rnd_start = 1
        rnd_step = 1
        return normalized_discounted_difference(positions, is_sensitive, rnd_n, rnd_start, rnd_step, s)

    def correct(self, estimated: float, q: float, p: float, s: float) -> float:
        """
        Computes the corrected DP
        :param estimated: Estimated, non-corrected value
        :param q: Error rate of the proxy model for protected class G1
        :param p: Error rate of the proxy model for non-protected class G0
        :param s: Population fraction of the protected class
        :return: Corrected value
        """
        raise NotImplementedError()


class rNDMetricFirstWorldview(rNDMetric):
    def correct(self, estimated: float, q: float, p: float, s: float = None) -> float:
        """
        Computes the corrected rND estimation under the first worldview

        :param estimated: Estimated rND value
        :param q: Error rate of the proxy model for the protected group
        :param p: Error rate of the proxy model for the non-protected group
        :param s: Population fraction of the protected group
        :return: Corrected estimation
        """

        if float(q + p) == 1.0:
            raise ValueError("Invalid sum of error rates, division by zero, failed to continue")

        return abs(estimated / (1.0 - q - p))


class rNDMetricSecondWorldview(rNDMetric):
    def correct(self, estimated: float, q: float, p: float, s: float = None) -> float:
        """
        Computes the corrected rND estimation

        :param estimated: Estimated rND value
        :param q: Error rate of the proxy model for the protected group
        :param p: Error rate of the proxy model for the non-protected group
        :param s: Population percentage of the protected group
        :return: Corrected estimation
        """

        p_a_hat_1 = (1 - q) * s + p * (1 - s)
        p_a_hat_0 = q * s + (1 - p) * (1 - s)  # 1 - p_a_hat_1
        term1 = ((1 - q) * s) / p_a_hat_1
        term2 = (q * s) / p_a_hat_0
        return abs(estimated * (term1 - term2))


def rnd_factory(ratio: float, worldview: int) -> rNDMetric:
    """
    Factory method for choosing the right subclass
    :param ratio: rND ratio
    :param worldview: 1 for Assumption I and 2 for Assumption II
    :return: rNDMetric subclass
    """
    if worldview == 1:
        return rNDMetricFirstWorldview(ratio)
    elif worldview == 2:
        return rNDMetricSecondWorldview(ratio)
    else:
        raise ValueError(f'Unsupported worldview: {worldview}')


# metric computation for asin data
def compute_rND(asin_data) -> float:
    """
    Computes the rND metric for the given ranking

    :param asin_data: List of tuples (position, attribute, sensitive_value)
    :return: rND metric
    """

    positions, is_sensitive, asin = zip(*asin_data)
    positions_np = np.array(positions)
    is_sensitive_np = np.array(is_sensitive)
    value = normalized_discounted_difference(positions_np, is_sensitive_np, n=20, start=2, step=1, s=0.5)
    return float(value)


# UDF function for collective computation
compute_rND_udf = F.udf(compute_rND, returnType=DoubleType())