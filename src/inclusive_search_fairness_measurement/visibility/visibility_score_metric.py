import numpy as np


def visibility_score(positions: np.array, is_sensitive: np.array, n=10) -> float:
    """
    Computes visibility score for the given ranking

    :param item_ids: Numpy array of item ids
    :param positions: Numpy array of positions
    :param is_sensitive: Numpy array of binary values indicating whether the item has a sensitive attribute
    :param n: Last rank, the default value is 10.
    :return: 1 if all sensitive attribute values are present in the ranking, 0 otherwise
    """

    if not positions.size == is_sensitive.size:
        raise ValueError('Length of input arrays are different, failed to continue')

    if n <= 0:
        raise ValueError('Invalid value for n (n <= 0), failed to continue')

    # Sort based on position
    updated_position_indexes = positions.argsort()
    is_sensitive = is_sensitive[updated_position_indexes]

    # Computing visibility_score
    # value = np.sum(sorted_data[:n, 2]) / min(n, sorted_data.shape[0])

    if np.count_nonzero(is_sensitive[:n]) > 0:
        return 1
    else:
        return 0


