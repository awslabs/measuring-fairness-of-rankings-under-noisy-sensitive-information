import pandas as pd

from typing import List, Dict, Callable
from sklearn.preprocessing import MinMaxScaler
from data_read.dataset import Dataset
from fairness_measurement_uncertainty.data_read.data_simulator import sample_ranking_normal, sample_ranking_AEO


class SimulatedDatasetBuilder:
    def __init__(
            self,
            sensitive_att_name: str,
            sens_att_value: float,
            non_sens_att_value: float,
            score_field: str,
            att_values: List[int],
            test_size: float,
            ranking_sampler_args: Dict,
            ranking_sampler: Callable,
    ):
        self.sensitive_att_name = sensitive_att_name
        self.sens_att_value = sens_att_value
        self.non_sens_att_value = non_sens_att_value
        self.score_field = score_field
        self.att_values = att_values

        self.test_size = test_size
        self.ranking_sampler_args = ranking_sampler_args
        self.ranking_sampler = ranking_sampler

        self.sensitive_pred_features = ["dummy"]
        self.score_pred_features = ["dummy"]

    @staticmethod
    def from_config(sim_params: Dict) -> "SimulatedDatasetBuilder":
        """
        Instantiates a SimulatedDatasetBuilder object
        :param sim_params: Parameters used in the experiment
        :return: A SimulatedDatasetBuilder object
        """
        sensitive_att_name = sim_params["dataset.sensitive_att_name"]
        str_val_map = sim_params["dataset.att_values_numeric"]
        sens_att_value_str = sim_params["dataset.sens_att_value"]
        sens_att_value = int(str_val_map[sens_att_value_str])
        non_sens_att_value_str = sim_params["dataset.non_sens_att_value"]
        non_sens_att_value = int(str_val_map[non_sens_att_value_str])
        worldview = int(sim_params["worldview"])

        score_field = sim_params["dataset.score_field"]
        att_values = [non_sens_att_value, sens_att_value]
        test_size = float(sim_params["dataset.test_size"])

        data_gen_str = sim_params["dataset.gen"]
        ranking_sampler_args = {
            "s": float(sim_params["dataset.s"]),
            "n": int(sim_params["dataset.n"]),
            "attribute_name": sensitive_att_name,
            "score_field": score_field,
            "protected_label": sens_att_value,
            "non_protected_label": non_sens_att_value
        }
        if worldview == 1 and data_gen_str == "normal":
            ranking_sampler_args["mu_f"] = float(sim_params["dataset.normal.mu_f"])
            ranking_sampler_args["mu_m"] = float(sim_params["dataset.normal.mu_m"])
            ranking_sampler_args["sigma_f"] = float(sim_params["dataset.normal.sigma_f"])
            ranking_sampler_args["sigma_m"] = float(sim_params["dataset.normal.sigma_m"])
            ranking_sampler = sample_ranking_normal
        elif worldview == 2 and data_gen_str == "AEO":
            ranking_sampler_args["q"] = float(sim_params["model.flip.g1_error_rate"])
            ranking_sampler_args["p"] = float(sim_params["model.flip.g2_error_rate"])
            ranking_sampler_args["mu_f"] = float(sim_params["dataset.normal.mu_f"])
            ranking_sampler_args["mu_m"] = float(sim_params["dataset.normal.mu_m"])
            ranking_sampler_args["sigma_f"] = float(sim_params["dataset.normal.sigma_f"])
            ranking_sampler_args["sigma_m"] = float(sim_params["dataset.normal.sigma_m"])
            ranking_sampler = sample_ranking_AEO
        else:
            raise RuntimeError(f"unsupported distribution type: {data_gen_str} for worldview {worldview}")


        return SimulatedDatasetBuilder(
            sensitive_att_name,
            sens_att_value,
            non_sens_att_value,
            score_field,
            att_values,
            test_size,
            ranking_sampler_args,
            ranking_sampler,
        )

    def get_dataset(self) -> Dataset:
        """
        Returns a dataset object containing the splits
        :return: A Dataset object
        """
        # Generate synthetic data
        features = self.ranking_sampler(**self.ranking_sampler_args)

        # Read data and split it
        X = features.drop([self.sensitive_att_name], axis=1)
        y = features[self.sensitive_att_name]

        # Add dummy features
        X["dummy"] = y.copy()

        # MinMax scaling
        scaler = MinMaxScaler()
        X = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)

        dataset_obj = Dataset.from_full_data(X, y, self.test_size, self.score_field)

        return dataset_obj