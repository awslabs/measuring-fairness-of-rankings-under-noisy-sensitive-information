import numpy as np

from numba import jit, prange


@jit(nopython=True, parallel=True)
def pairwise_parity(positions: np.array, is_sensitive: np.array) -> float:
    """
    Returns pairwise parity, see https://arxiv.org/pdf/2105.03153.pdf.
    Assumption: The sensitive attribute is assumed to be binary-valued.

    :param positions: Numpy array of positions
    :param is_sensitive: Numpy array of indicators for sensitive attribute values
    :return: Pairwise parity
    """

    if positions.size != is_sensitive.size:
        raise ValueError("Length of input arrays are different, failed to continue")

    num_sensitive = np.sum(is_sensitive)

    if int(num_sensitive) == 0 or int(num_sensitive) == is_sensitive.size:
        return 1.0

    updated_position_indexes = positions.argsort()
    is_sensitive = is_sensitive.copy()
    is_sensitive = is_sensitive[updated_position_indexes]

    denom = num_sensitive * (is_sensitive.size - num_sensitive)

    # slow code
    sensitive_term = 0.0
    non_sensitive_term = 0.0
    is_non_sensitive = 1 - is_sensitive
    for i in prange(is_sensitive.size-1):
        if int(is_sensitive[i]) == 1:
            sensitive_term += np.sum(is_non_sensitive[i + 1:])
        else:
            non_sensitive_term += np.sum(is_sensitive[i + 1:])

    parity = (non_sensitive_term - sensitive_term) / denom

    return float(parity)


if __name__ == "__main__":
    a = np.array([1, 2, 3])
    print(a * a)