import numpy as np

from inclusive_search_fairness_measurement.pairwise_parity.pairwise_parity_metric import pairwise_parity


class PairwiseParityMetric:
    full_name = "DP"

    def __init__(self):
        pass

    def compute(self, positions: np.array, is_sensitive: np.array, *args) -> float:
        """
        Computes the demographic parity of the ranking list
        :param positions: Numpy array of positions
        :param is_sensitive: Numpy array of indicators for sensitive attribute values
        :return: Pairwise parity
        """
        return pairwise_parity(positions, is_sensitive)

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


class PairwiseParityMetricFirstWorldview(PairwiseParityMetric):
    def correct(self, estimated: float, q: float, p: float, s: float = None) -> float:
        """
        Computes the corrected parity under the first worldview
        :param estimated: Estimated, non-corrected parity value
        :param q: Error rate of the proxy model for protected class G1
        :param p: Error rate of the proxy model for non-protected class G0
        :param s: Population fraction of the protected class
        :return: Corrected parity
        """

        x = (p - p * s + s - s * q)
        numerator = x * (1 - x)
        denom = s * (1 - s) * (1 - p - q)
        factor = numerator / denom
        corrected_parity = estimated * factor

        return corrected_parity


class PairwiseParityMetricSecondWorldview(PairwiseParityMetric):
    def correct(self, estimated: float, q: float, p: float, s: float = None) -> float:
        """
        Computes the corrected parity under the second worldview

        :param estimated: Estimated, non-corrected parity value
        :param q: Error rate of the proxy model for protected class
        :param p: Error rate of the proxy model for non-protected class
        :param s: Population percentage of the protected class
        :return: Corrected parity
        """

        return estimated * (1 - p - q)


def dp_factory(worldview: int) -> PairwiseParityMetric:
    """
    Factory method for choosing the right subclass of PairwiseParityMetric
    :param worldview: 1 for Assumption I and 2 for Assumption II
    :return: A subclass of PairwiseParityMetric
    """

    if worldview == 1:
        return PairwiseParityMetricFirstWorldview()
    elif worldview == 2:
        return PairwiseParityMetricSecondWorldview()
    else:
        raise ValueError(f'Unsupported worldview: {worldview}')