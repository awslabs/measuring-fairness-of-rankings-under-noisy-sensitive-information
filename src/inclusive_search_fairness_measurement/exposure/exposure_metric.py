import numpy as np

from typing import Callable
from baseline.label_sampler import sample_labels


def ndcg_dropoff(n_items: int) -> np.ndarray:
    """
    Computes an array with normalized logarithmic exposure dropoff given nummber of items
    :param n_items: Number of items in the ranking
    :return: Numpy array containing the ndcg dropoff values
    """

    positions = np.arange(n_items)
    dropoff = 1 / np.log2(2 + positions)
    return dropoff / np.sum(dropoff)


def linear_dropoff(n_items: int) -> np.ndarray:
    """
    Compute an array with constantly decreasing exposure dropoff give number of items
    :param n_items: Number of items in the ranking
    :return: Numpy array containing the linearly decreasing values
    """
    positions_reverse = np.arange(n_items)[::-1]
    return positions_reverse / np.sum(positions_reverse)


def exposure(positions: np.array, is_sensitive: np.array, dropoff_func: Callable[[int], np.ndarray]) -> float:
    """
    Computes the value of the exposure metric for the given list
    See https://quip-amazon.com/oWKcAviFacef/WIP-unchecked-Fairness-in-ranking-exposure for details

    :param positions: Numpy array of positions
    :param is_sensitive: Numpy array of indicators for sensitive attribute values
    :param dropoff_func: Function for computing an array of dropoff values for the given nummber of items
    :return: List's exposure
    """

    if positions.size != is_sensitive.size:
        raise ValueError("Length of input arrays are different, failed to continue")

    num_sensitive = np.sum(is_sensitive)
    if int(num_sensitive) == 0 or int(num_sensitive) == is_sensitive.size:
        return 1.0

    updated_position_indexes = positions.argsort()
    is_sensitive = is_sensitive.copy()
    is_sensitive = is_sensitive[updated_position_indexes]
    dropoff = dropoff_func(positions.size)
    alpha = np.sum(dropoff * is_sensitive)
    return 2 * alpha - 1


def exposure_baseline(positions: np.array, is_sensitive: np.array,
                      dropoff_func: Callable[[int], np.ndarray], sample_rate: float, sample_weight: str, m=1) -> float:
    """
    Computes exposure using the baseline method https://md.ekstrandom.net/pubs/fair-estimate-www2021
    :param positions: Numpy array of positions
    :param is_sensitive: Numpy array of indicators for sensitive attribute values
    :param dropoff_func: Function for computing an array of dropoff values for the given nummber of items
    :param sample_rate: Fraction of labels sampled
    :param sample_weight: Weights used for sampling
    :param m: Size of the bucket
    :return: List's exposure
    """

    if positions.size != is_sensitive.size:
       raise ValueError("Length of input arrays are different, failed to continue")

    num_sensitive = np.sum(is_sensitive)
    if int(num_sensitive) == 0 or int(num_sensitive) == is_sensitive.size:
       return 1.0

    selected_indices, weights = sample_labels(positions.size, sample_rate, sample_weight, m)
    n = positions.size

    updated_position_indexes = positions.argsort()
    is_sensitive = is_sensitive.copy()
    is_sensitive = is_sensitive[updated_position_indexes]
    dropoff = dropoff_func(n)
    first_array = dropoff * is_sensitive * selected_indices
    second_array = weights
    alpha = np.sum(np.divide(first_array, second_array))

    is_not_sensitive = np.ones(n) - is_sensitive
    first_array = dropoff * is_not_sensitive * selected_indices
    second_array = weights
    beta = np.sum(np.divide(first_array, second_array))

    return alpha - beta
