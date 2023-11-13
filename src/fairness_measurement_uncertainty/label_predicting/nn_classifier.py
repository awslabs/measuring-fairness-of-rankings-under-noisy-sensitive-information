import pandas as pd
import numpy as np

from sklearn.neural_network import MLPClassifier
from label_predicting.string_preprocessing import StringProcessor


class NNModel:
    def __init__(self):
        self.model = None
        self.model_python = MLPClassifier(max_iter=1000)
        self.str_process_obj = None

    def train(self, x: pd.DataFrame, y: np.array, word_level=False) -> "NNModel":
        """
        Trains an MLP

        :param x: Train data
        :param y: Target values
        :return: Returns itself (a trained model)
        """

        train_x = x.copy()

        if word_level:
            # Processing the textual input
            self.str_process_obj = StringProcessor()
            self.str_process_obj.read_embeddings()
            train_x = self.str_process_obj.avg_embeddings(x)

        # self.model = MLPClassifier(solver='adam', alpha=1e-1, hidden_layer_sizes=(16,), random_state=1, max_iter=1000)
        # self.model = MLPClassifier(solver='adam', hidden_layer_sizes=(16,), random_state=1, max_iter=1000)
        self.model = MLPClassifier(max_iter=1000)
        self.model.fit(train_x, y)

        return self

    def predict(self, x: pd.DataFrame, y: pd.Series, word_level=False) -> pd.Series:
        """
        Predicts labels for the given data

        :param x: The data whose attribute is predicted with the given model
        :param y: Ground truth labels
        :param word_level: True if the input is string and needs word-level preprocessing
        :return: Predicted labels
        """

        test_x = x.copy()

        if word_level:
            test_x = self.str_process_obj.avg_embeddings(x)

        output_y = self.model.predict(test_x)

        return output_y
