import pandas as pd
import numpy as np
import random

from label_predicting.flip_classifier import FlipModel


def linear_decreasing_exposure(n: int, s: float) -> np.array:
    """
    Returns an array whose ith element is the probability of seeing a protected member at position i of a ranking
    The exposure probabilities decrease linearly

    :param n: Total number of date points
    :param s: Probability of protected group
    :return: Array of exposure probabilities using function -ai + b
    """

    b1 = 2 * n * s / (n - 1)
    b2 = (n + 1 - 2 * s) / (n - 1)
    b_upper_bound = min(b1, b2)
    # b can be anything in the range (s, b_upper_bound)
    # To make it deterministic, we choose the middle point
    b = (s + b_upper_bound) / 2
    a = 2 * (b - s) / (n + 1)
    e = [-1 * a * i + b for i in range(1, n+1)]
    return np.array(e)


def linear_increasing_exposure(n: int, s: float) -> np.array:
    """
    Returns an array whose ith element equal to the probability of seeing a protected member at position i of a ranking
    The exposure probabilities increase linearly

    :param n: Total number of date points
    :param s: Probability of protected group
    :return: Array of exposure probabilities using function ai - b
    """

    e = linear_decreasing_exposure(n, 1.0 - s)
    return 1.0 - e


def constant_exposure(n: int, s: float, p: float) -> np.array:
    """
    Returns an array of exposure probabilities populated with constant probability p up to position min(n, n* s / p)
    and 0 for the rest of it

    :param n: Total number of date points
    :param s: Probability of protected group
    :param p: Constant exposure probability
    :return: Array of exposure probabilities
    """

    if p < s:
        raise ValueError("p must be greater than s, failed to continue")

    max_index = min(n, int(n * s / p))
    e = [p] * max_index
    if n > max_index:
        for i in range(n - max_index):
            e.append(0.0)
    return np.array(e)


def generate_simulated_ranking(n: int, s: float, e: np.array, attribute_name: str = 'sex', score_field: str = "score",
                               protected_label: int = 1, non_protected_label: int = 0) -> pd.DataFrame:
    """
    Generates a simulated ranking (This function is no longer used in our code)
    Assumption: The attribute value is binary-valued

    :param n: Total number of date points
    :param s: Probability of protected group
    :param e: The exposure probability array. e[i]: probability that a protected member appears in the i+1th location
    :param attribute_name: Column name for attribute values
    :param score_field: Column name for scores
    :param protected_label: Attribute value of the protected group
    :param non_protected_label: Attribute value of the non-protected group
    :return: A simulated ranking with n data points sxn protected members
    """

    acceptable_error = 0.2
    sum_e = np.sum(e)
    if not s * n - acceptable_error < sum_e < s * n + acceptable_error:
        raise ValueError('Invalid sum of probabilities in e, failed to continue')

    if len(e) != n:
        raise ValueError('The size of array e is not equal to n, failed to continue')

    for i in range(e.size):
        if e[i] < 0.0 or e[i] > 1.0:
            raise ValueError('Invalid probabilities in array e, failed to continue')

    total_p = n * s
    total_np = n * (1 - s)
    init_data = np.zeros((n, 2))
    count_p = 0
    count_np = 0
    data = pd.DataFrame(init_data, columns=[score_field, attribute_name])

    scores = [0] * n
    labels = [""] * n
    for i in range(n):
        # if i % 10000 == 0:
        #     print(i)
        r = random.uniform(0, 1)
        if r <= e[i]:
            if count_p < total_p:
                labels[i] = protected_label
                count_p += 1
            else:
                labels[i] = non_protected_label
                count_np += 1
        else:
            if count_np < total_np:
                labels[i] = non_protected_label
                count_np += 1
            else:
                labels[i] = protected_label
                count_p += 1
        scores[i] = n - i
    data[score_field] = scores
    data[attribute_name] = labels

    print(count_p, count_np)
    return data


def sample_ranking_normal(n: int, s: float, mu_f: float, sigma_f: float, mu_m: float, sigma_m: float,
                          attribute_name: str = 'sex', score_field: str = "score",
                          protected_label: int = 1, non_protected_label: int = 0) -> pd.DataFrame:
    """
    Generates a simulated ranking for Assumption I (only A and S)
    :param n: total number of samples
    :param s: fraction of samples belonging to the protected group
    :param mu_f: mean of the normal distribution of scores for the protected group
    :param sigma_f: std of the normal distribution of scores for the protected group
    :param mu_m: mean of the normal distribution of scores for the non-protected group
    :param sigma_m: std of the normal distribution of scores for the non-protected group
    :param attribute_name: name of the sensitive attribute
    :param score_field: name of the score column
    :param protected_label: sensitive attribute value of the protected group
    :param non_protected_label: sensitive attribute value of the non-protected group
    :return: Dataframe containing synthetic dataset generated according to Assumption I (without proxy labels)
    """

    sensitive_count = int(n * s)
    if sensitive_count == n or sensitive_count == 0:
        raise ValueError("invalid s, failed to continue")

    s = np.random.normal(mu_f, sigma_f, size=(sensitive_count, 1))
    ns = np.random.normal(mu_m, sigma_m, size=(n - sensitive_count, 1))

    s_value = np.ones((sensitive_count, 1)) * protected_label
    ns_value = np.ones((n - sensitive_count, 1)) * non_protected_label

    scores = np.vstack((s, ns))
    values = np.vstack((s_value.astype(int), ns_value.astype(int)))
    data = np.hstack([scores, values])
    dataframe = pd.DataFrame(data, columns=[score_field, attribute_name])
    return dataframe


def sample_ranking_AEO(n: int, s: float, q: float, p: float, mu_f: float, sigma_f: float, mu_m: float, sigma_m: float,
                          attribute_name: str = 'sex', score_field: str = "score",
                          protected_label: int = 1, non_protected_label: int = 0) -> pd.DataFrame:
    """
    Generates data according to Assumption II (contains A, S and A hat)
    :param n: total number of samples
    :param s: fraction of samples belonging to the protected group
    :param q: error rate of the classifier for the protected group
    :param p: error rate of the classifier for the non-protected group
    :param mu_f: mean of the normal distribution of scores for the protected group
    :param sigma_f: std of the normal distribution of scores for the protected group
    :param mu_m: mean of the normal distribution of scores for the non-protected group
    :param sigma_m: std of the normal distribution of scores for the non-protected group
    :param attribute_name: name of the sensitive attribute
    :param score_field: name of the score column
    :param protected_label: sensitive attribute value of the protected group
    :param non_protected_label: sensitive attribute value of the non-protected group
    :return: Dataframe containing synthetic dataset generated according to Assumption II
    """
    num_sensitive = int(n * s)
    num_non_sens = n - num_sensitive

    if num_non_sens <= 0 or num_sensitive <= 0:
        raise ValueError("invalid s, failed to continue")

    # Generating estimated labels
    original_labels = np.zeros((n,))
    original_labels[:num_sensitive] = protected_label
    original_labels[num_sensitive:] = non_protected_label
    model = FlipModel()
    model.set_model(protected_label, non_protected_label, q, p)
    estimated_labels = model.predict(None, pd.Series(original_labels)).to_numpy()

    # generating scores based on estimated labels
    num_sensitive_prime = np.sum(estimated_labels == protected_label)
    num_non_sens_prime = n - num_sensitive_prime
    scores_sens = np.random.normal(mu_f, sigma_f, size=(n,))
    scores_non_sens = np.random.normal(mu_m, sigma_m, size=(n,))
    scores = np.zeros((n,))
    protected_indices = estimated_labels == protected_label
    scores[protected_indices] = scores_sens[protected_indices]
    non_protected_indices = estimated_labels == non_protected_label
    scores[non_protected_indices] = scores_non_sens[non_protected_indices]

    data = np.hstack([original_labels.reshape(n, 1), estimated_labels.reshape(n, 1), scores.reshape(n, 1)])
    dataframe = pd.DataFrame(data, columns=[attribute_name, "estimated_labels", score_field])
    return dataframe

    # An alternative method for generating data
    # estimated_labels = np.zeros((n,))
    # num_sensitive_prime = int(n * s * (1 - q) + n * (1 - s) * p)
    # num_non_sens_prime = n - num_sensitive_prime
    # estimated_labels[:num_sensitive_prime] = protected_label
    # scores = np.zeros((n,))
    # scores_sens = np.random.normal(mu_f, sigma_f, size=(num_sensitive_prime,))
    # scores_non_sens = np.random.normal(mu_m, sigma_m, size=(num_non_sens_prime,))
    # scores[:num_sensitive_prime] = scores_sens
    # scores[num_sensitive_prime:] = scores_non_sens
    #
    # orig_labels = np.zeros((n,))
    # index_1 = int((1 - q) * s * n)
    # orig_labels[:index_1] = protected_label
    # index_2 = index_1 + int(p * (1 - s) * n)
    # orig_labels[index_1:index_2] = non_protected_label
    # index_3 = index_2 + int(q * s * n)
    # orig_labels[index_2:index_3] = protected_label
    # orig_labels[index_3:] = non_protected_label
    #
    # data = np.hstack([orig_labels.reshape(n, 1), estimated_labels.reshape(n, 1), scores.reshape(n, 1)])
    # dataframe = pd.DataFrame(data, columns=[attribute_name, "estimated_labels", score_field])
    # return dataframe
