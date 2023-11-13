import math
import numpy as np


def normalized_discounted_ratio(positions: np.array, is_sensitive: np.array,
                                        n=20, start=10, step=10, population_demo=[]) -> float:
    """
    Computes normalized discounted ratio (rRD) for a given ranking

    :param positions: Numpy array of positions, smaller values correspond to higher ranks.
    :param is_sensitive: Numpy array of binary values indicating whether the item has a sensitive attribute
    :param n: Last rank, the default value is 10.
    :param start: Initial cut-ff rank
    :param step: Step size. E.g., step=10 means we compute the difference at cutt-off
    values k = start + 10, start + 20, start + 30, ...
    :param population_demo: Population demographics. The first element is the fraction of the protected group.
    When empty, the distribution is inferred from the ranking.
    :return: rRD value
    """

    if not positions.size == is_sensitive.size:
        raise ValueError('Length of input arrays are different, failed to continue')
        # return -1.0

    if start <= 1 or n <= 1:
        raise ValueError('Invalid value for start or n, failed to continue')
        # return -1.0

    if np.count_nonzero(is_sensitive.astype(int)) == 0 or np.all(is_sensitive.astype(int)):
        raise ValueError('All attribute values are the same, failed to continue')
        # return -1.0

    if len(population_demo) > 0 and float(population_demo[1]) == 0.0:
        raise ValueError('Invalid population demographic, division by zero, failed to continue')
        # return -1.0

    last_rank = positions.size

    if min(n, last_rank) < start:
        raise ValueError('Start value is larger than the largest rank, failed to continue')
        # return -1.0

    if not np.array_equal(is_sensitive, is_sensitive.astype(bool)):
        raise ValueError('is_sensitive variable must contain only 0 and 1, failed to continue')
        # return -1.0

    # Sort based on position
    updated_position_indexes = positions.argsort()
    is_sensitive = is_sensitive.copy()
    is_sensitive = is_sensitive[updated_position_indexes]

    # Computing rRD
    sensitive_count = np.sum(is_sensitive)
    non_sensitive_count = is_sensitive.size - sensitive_count
    frac_value = sensitive_count / non_sensitive_count

    if population_demo:
        frac_value = population_demo[0] / population_demo[1]

    v = 0.0
    z = 0.0

    for i in range(start, min(n, last_rank) + 1, step):
        sensitive_count_i = np.sum(is_sensitive[:i])
        non_sensitive_count_i = i - sensitive_count_i
        if non_sensitive_count_i.astype(int) == 0:
            raise ValueError('Division by zero error, failed to compute the metric')
            # return -1.0
        frac_value_i = sensitive_count_i / non_sensitive_count_i
        v += (1.0 / math.log2(i)) * abs(frac_value_i - frac_value)
        z += (1.0 / math.log2(i)) * max(frac_value, i-1 - frac_value)
    res = v / z

    return res
