import math
import numpy as np


def normalized_discounted_difference(positions: np.array, is_sensitive: np.array,
                                     n=20, start=10, step=10, s: float = None) -> float:
    """
    Computes normalized discounted difference (rND) for the sensitive attribute.

    :param positions: Numpy array of positions, smaller values correspond to higher ranks.
    :param is_sensitive: Numpy array of binary values indicating whether the item has a sensitive attribute
    :param n: Last rank, the default value is 10.
    :param start: Initial cut-ff rank
    :param step: Step size. E.g., step=10 means we compute the difference at cutt-off
    values k = start + 10, start + 20, start + 30, ...
    :param s: Population fraction of the protected group G1
    when None, it would be inferred from the given ranking
    :return: rND value
    """

    if not positions.size == is_sensitive.size:
        raise ValueError('Length of input arrays are different, failed to continue')
        # return -1.0

    if start < 1 or n <= 1:
        raise ValueError('Invalid value for start or n, failed to continue')
        # return -1.0

    if np.sum(is_sensitive.astype(int)) < 1:
        raise ValueError('Zero occurrence of the sensitive value, failed to continue')
        # return -1.0

    last_rank = positions.size

    if min(n, last_rank) < start:
        raise ValueError('Start value is larger than the largest rank, failed to continue')
        # return -1.0

    # Sort based on position
    updated_position_indexes = positions.argsort()
    is_sensitive = is_sensitive.copy()
    is_sensitive = is_sensitive[updated_position_indexes]

    # Computing rND
    sensitive_portion = np.sum(is_sensitive.astype(int)) / is_sensitive.size

    if s is not None:
        sensitive_portion = s

    v = 0.0
    z = 0.0
    for i in range(start, min(n, last_rank) + 1, step):
        sensitive_portion_i = np.sum(is_sensitive[:i].astype(int)) / i
        v += (1.0 / math.log2(i+1)) * (sensitive_portion_i - sensitive_portion)
        z += (1.0 / math.log2(i+1))
    res = v / z

    return abs(res)
