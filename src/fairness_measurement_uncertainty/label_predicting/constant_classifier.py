import numpy as np
import pandas as pd


# No longer used
class ConstantModel:
    def __init__(self):
        self.model = None

    def set_model(self, s_att_val: str, ns_att_val: str, q: float, p: float):
        self.model = {s_att_val: q, ns_att_val: p}

    def train(self, x: pd.DataFrame, y: np.array) -> "ConstantModel":
        """
        Trains the model, does nothing here
        """
        return self

    def predict(self, x: pd.DataFrame, y: pd.Series) -> pd.Series:
        """
        Random assignment of labels based on the provided probabilities

        :param x: The data whose attribute is predicted with the given model
        :param y: The original labels to be flipped with probabilities given in the model
        :return: Predicted labels
        """

        attribute_values = [e for e in self.model]
        p = [self.model[e] for e in attribute_values]
        output_y = np.random.choice(a=attribute_values, size=x.shape[0], p=p)

        return pd.Series(output_y)
