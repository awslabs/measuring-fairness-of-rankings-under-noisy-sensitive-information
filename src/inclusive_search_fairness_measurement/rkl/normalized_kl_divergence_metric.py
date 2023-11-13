import math
import numpy as np


def __kl_divergence(p: np.array, q: np.array) -> float:
    r = 0.0
    for i in range(p.size):
        if p[i] != 0.0:
            r += p[i] * math.log2(p[i] / q[i])
    return r


def normalized_discounted_kl_divergence(positions: np.array, is_sensitive: np.array,
                                        n=20, start=10, step=10, population_demo=[]) -> float:
    """
    Computes normalized discounted KL-divergence (rKL) for a binary-valued attribute

    :param positions: Numpy array of positions, smaller values correspond to higher ranks.
    :param is_sensitive: Numpy array of binary values indicating whether the item has a sensitive attribute
    :param n: Last rank, the default value is 10.
    :param start: Initial cut-ff rank
    :param step: Step size. E.g., step=10 means we compute the difference at cutt-off
    values k = start + 10, start + 20, start + 30, ...
    :param population_demo: Population demographics. The first element is the fraction of the protected group G1.
    When empty, the distribution is inferred from the ranking.
    :return: rKL value
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

    # Computing rKL
    sensitive_portion = np.sum(is_sensitive.astype(int)) / is_sensitive.size
    non_sensitive_portion = 1 - sensitive_portion

    if population_demo:
        sensitive_portion = population_demo[0]
        non_sensitive_portion = population_demo[1]

    v = 0.0
    z = 0.0
    q = np.array([sensitive_portion, non_sensitive_portion])
    farthest_p = np.array([1.0, 0.0])
    if sensitive_portion > non_sensitive_portion:
        farthest_p = np.array([0.0, 1.0])

    max_kl = __kl_divergence(farthest_p, q)
    for i in range(start, min(n, last_rank) + 1, step):
        sensitive_portion_i = np.sum(is_sensitive[:i]) / i
        non_sensitive_portion_i = 1 - sensitive_portion_i
        p = np.array([sensitive_portion_i, non_sensitive_portion_i])
        kl_i = __kl_divergence(p, q)
        v += (1.0 / math.log2(i+1)) * kl_i
        z += (1.0 / math.log2(i+1)) * max_kl
    res = v / z

    return res
