import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


class RFModel:
    def __init__(self):
        self.model = None
        self.model_python = RandomForestClassifier()

    def train(self, x: pd.DataFrame, y: np.array) -> "RFModel":
        """
        Trains a random forest classifier

        :param x: Train data
        :param y: Target values
        :return: Returns itself (a trained model)
        """

        self.model = self.model_python
        self.model.fit(x, y)

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
