import numpy as np
import pandas as pd

from typing import Dict
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MinMaxScaler


def feature_importance(model, X, cat_features, y) -> Dict:
    """
    Computes importance of features using the permutation technique
    :param model: A trained model (must have fit and predict functions)
    :param X: Input features
    :param cat_features: Categorical features
    :param y: Input labels
    :return: A dictionary mapping the features to their importance
    """

    # Sample X
    remove_n = int(X.shape[0] * 0.5)
    drop_indices = np.random.choice(X.index, remove_n, replace=False)
    X = X.drop(drop_indices)
    y = y.drop(drop_indices)

    # Ordinal encoding of categorical features
    # for feature in cat_features:
    #     X[feature] = X[feature].mask(X[feature] == ' ').factorize()[0]
    for column in cat_features:
        tempdf = pd.get_dummies(X[column], prefix=column)
        df = pd.merge(
            left=X,
            right=tempdf,
            left_index=True,
            right_index=True,
        )
        X = df.drop(columns=column)
    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
    feature_names = list(X.columns.values)

    # Fit the model
    model.fit(X, y)

    # Computing the feature importance
    result = permutation_importance(model, X, y, n_repeats=50)
    importance_vals = result.importances_mean

    # Summarize feature importance
    feature_importance_vals = {}
    for i, v in enumerate(importance_vals):
        feature_importance_vals[feature_names[i]] = abs(v)

    return feature_importance_vals
