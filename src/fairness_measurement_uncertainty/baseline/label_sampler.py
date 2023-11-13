import math
from typing import Tuple

import numpy as np


def sample_labels(n: int, sample_rate: float, sample_weight: str, m=1) -> Tuple[np.array, np.array]:
    """
    Returns the tuple (sampled indices, weight of samples)
    See paper https://arxiv.org/abs/2108.05152
    :param n: Number of samples
    :param sample_rate: sampling rate [0, 1]
    :param sample_weight: String with values "uniform" and "weighted" specdifying the type of sampling
    :param m: size of the bucket
    :return: (sampled indices, weight of samples)
    """
    if n <= 0:
        raise RuntimeError(f'n must be a positive integer: {n}')

    weights = np.ones(n)

    if sample_weight == 'uniform':
        pass
    elif sample_weight == 'weighted':
        # TODO either remove the following line or remove m from arguments
        m = math.ceil(n * 0.5)

        ranks = np.arange(1, n+1)
        n_array= np.ones(n) * n
        weights = (1.0/(2 * n)) * np.log2(n_array/ranks)
        norm_weights = weights / np.sum(weights)
        # print(np.sum(norm_weights[0: min(n, 0 + m)]))
        for i in range(0, n, m):
            weights[i: min(n, i + m)] = np.sum(norm_weights[i: min(n, i + m)])
    else:
        raise RuntimeError(f'Unsupported sample weight str {sample_weight}')

    # normalize weights
    norm_weights = weights / np.sum(weights)

    num_samples = int(n * sample_rate)
    samples = np.random.choice(n, num_samples, p=norm_weights, replace=False)

    chosen_indices = np.zeros(n)
    chosen_indices[samples] = 1

    return chosen_indices, weights


