from typing import Dict

import numpy as np
import pandas as pd


class FlipModel:
    def __init__(self):
        self.model = None
        self.model_python = None

    @staticmethod
    def from_config(config: Dict) -> "FlipModel":
        model = FlipModel()
        g1_error_rate = float(config["model.flip.g1_error_rate"])
        g2_error_rate = float(config["model.flip.g2_error_rate"])
        model.set_model(
            config["dataset.att_values_numeric"][config["dataset.sens_att_value"]],
            config["dataset.att_values_numeric"][config["dataset.non_sens_att_value"]],
            g1_error_rate,
            g2_error_rate,
        )
        return model

    def set_model(self, s_att_val: float, ns_att_val: float, q: float, p: float):
        self.model = {s_att_val: q, ns_att_val: p}

    def train(self, x: pd.DataFrame, y: np.array) -> "FlipModel":
        """
        Train the model, does nothing here
        """
        return self

    def predict(self, x: pd.DataFrame, y: pd.Series) -> pd.Series:
        """
        Predicts labels for the given data using flip classifier

        :param x: The data whose attribute is predicted with the given model
        :param y: The original labels to be flipped with probabilities given in the model
        :return: Predicted labels
        """

        all_values = list(self.model.keys())
        output_y = y.copy(deep=True)
        alternative_values = {}
        for k in all_values:
            l = list(all_values)
            l.remove(k)
            alternative_values[k] = l
        for att_value, prob in self.model.items():
            random_values = np.random.uniform(0, 1, size=len(y))
            local_value_mask = y == att_value
            change_local_value = random_values <= prob
            other_value = np.random.choice(alternative_values[att_value], size=len(y), replace=True)
            change_indices = local_value_mask & change_local_value
            output_y[change_indices] = other_value[change_indices]
        return output_y


if __name__ == "__main__":
    model = FlipModel()
    model.set_model("f", "m", 0.3, 0.3)
    init_y = pd.Series(["f", "m", "f", "m", "f"])
    y = model.predict(None, init_y)
    print(f"y: {y}")
    print(f"y: {init_y}")
