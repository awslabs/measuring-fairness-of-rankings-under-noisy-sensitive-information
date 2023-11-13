import numpy as np

from typing import Callable
from inclusive_search_fairness_measurement.exposure.exposure_metric import ndcg_dropoff, exposure, linear_dropoff, \
    exposure_baseline


class ExposureMetric():
    full_name = "Exp"
    dropoff: Callable[[int], np.ndarray]

    def __init__(self, exposure_dropoff: Callable[[int], np.ndarray] = ndcg_dropoff):
        """
        Class holding methods for computation of the exposure metric and its correction
        See https://quip-amazon.com/oWKcAviFacef/WIP-unchecked-Fairness-in-ranking-exposure for details
        """
        self.dropoff = exposure_dropoff

    def compute(self, positions: np.array, is_sensitive: np.array, *args) -> float:
        """
        Find the value of exposure given a ranking list
        :param positions: Numpy array of positions
        :param is_sensitive: Numpy array of indicators for sensitive attribute values
        :return: List's exposure
        """

        return exposure(positions, is_sensitive, self.dropoff)

    def compute_baseline(self, positions: np.array, is_sensitive: np.array, sample_rate: float,
                         sample_weight: str = 'uniform', m: int = 1) -> float:
        """
        Find the value of exposure given a ranking list
        :param positions: Numpy array of positions
        :param is_sensitive: Numpy array of indicators for sensitive attribute values
        :param sample_rate: Fraction of the labels sampled for metric computation
        :param sample_weight: 'weighted' or 'uniform'
        :param m: Size of the bucket
        :return: List's exposure
        """

        return exposure_baseline(positions, is_sensitive, self.dropoff, sample_rate, sample_weight, m)

    def correct(self, estimated: float, q: float, p: float, s: float = None) -> float:
        """
        Computes the corrected exposure
        :param estimated: Estimated, non-corrected value
        :param q: Error rate of the proxy model for protected class
        :param p: Error rate of the proxy model for non-protected class
        :param s: Population fraction of the protected class. Unused
        :return: Corrected value
        """
        raise NotImplementedError()


class ExposureMetricFirstWorldview(ExposureMetric):
    def correct(self, estimated: float, q: float, p: float, s: float = None) -> float:
        """
        Computes the corrected exposure
        :param estimated: Estimated, non-corrected value
        :param q: Error rate of the proxy model for protected class G1
        :param p: Error rate of the proxy model for non-protected class G0
        :param s: Population fraction of the protected class. Unused
        :return: Corrected value
        """
        return (estimated - p + q) / (1 - p - q)


class ExposureMetricSecondWorldview(ExposureMetric):
    def correct(self, estimated: float, q: float, p: float, s: float = None) -> float:
        """
        Computes the corrected exposure
        :param estimated: Estimated, non-corrected value
        :param q: Error rate of the proxy model for protected class
        :param p: Error rate of the proxy model for non-protected class
        :param s: Population fraction of the protected class.
        :return: Corrected value
        """
        p_a_hat_1 = (1 - q) * s + p * (1 - s)
        p_a_hat_0 = 1 - p_a_hat_1
        term1 = ((1 - q) * s) / p_a_hat_1
        term2 = (q * s) / p_a_hat_0
        term3 = (2 * q * s) / p_a_hat_0
        res = (estimated + 1) * (term1 - term2) + term3 - 1
        return res


def exposure_factory(exposure_dropoff_str: str, worldview: int) -> ExposureMetric:
    """
    Factory method for choosing the right ExposureMetric subclass
    :param exposure_dropoff_str: The string specifying the type of dropoff to be used
    :param worldview: 1 for Assumption I and 2 for Assumption II
    :return: A subclass ExposureMetric according to the given worldview
    """
    if exposure_dropoff_str == "linear_dropoff":
        exposure_dropoff = linear_dropoff
    elif exposure_dropoff_str == "ndcg_dropoff":
        exposure_dropoff = ndcg_dropoff
    else:
        raise ValueError(f'Unsupported drop off method for exposure: {exposure_dropoff_str}')

    if worldview == 1:
        return ExposureMetricFirstWorldview(exposure_dropoff=exposure_dropoff)
    elif worldview == 2:
        return ExposureMetricSecondWorldview(exposure_dropoff=exposure_dropoff)
    else:
        raise ValueError(f'Unsupported worldview: {worldview}')