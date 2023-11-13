import pyltr
import pandas as pd
import numpy as np


class LambdaMARTRanker():
    def __init__(self):
        self.model = None
        # other metrics: https://github.com/jma127/pyltr
        # metric = pyltr.metrics.KendallTau
        # metric = pyltr.metrics.NDCG(k=10)
        metric = None
        model = pyltr.models.LambdaMART(
            metric=metric,
            n_estimators=10,
            learning_rate=0.02,
            max_features=0.5,
            query_subsample=0.5,
            max_leaf_nodes=10,
            min_samples_leaf=64
            # verbose=1
        )
        self.model_python = model

    def train(self, x: pd.DataFrame, y: np.array) -> "LambdaMARTRanker":
        """
        Train the LambdaMART model on x

        :param x: Train data
        :param y: Target values
        :return: Returns itself (a trained model)
        """

        self.model = self.model_python
        self.model.fit(x, y, np.ones(x.shape[0]))

        return self

    def predict(self, x: pd.DataFrame, y: pd.Series) -> pd.Series:
        """
        Predicts labels for the given data

        :param x: The data whose attribute is predicted with the given model
        :param y: Ground truth labels
        :return: Predicted labels
        """

        output_y = self.model.predict(x)

        return output_y



