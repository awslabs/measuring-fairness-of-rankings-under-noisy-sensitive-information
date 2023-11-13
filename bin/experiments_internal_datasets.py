import argparse
import json
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Dict
from sklearn.metrics import confusion_matrix
from experiments.all_metrics_results import plot_tommy
from experiments.experiments_pipeline import Pipeline
from experiments.results import Results
from src.InclusiveSearchFairnessMeasurement.bin.experiments_real import RealPipeline


class TommyPipeline(RealPipeline):
    def __init__(self, dataset_builder, metric, model, params: Dict):
        super().__init__(dataset_builder, metric, model, params)
        self.q = None  # Group-conditional error rate of G0
        self.p = None  # Group-conditional error rate of G1
        self.beta = None  # Fraction of population belonging to G1

    def setup_tommy(self):
        # Read the dataset
        self.dataset_builder.read_dataset()

    @staticmethod
    def from_config(params: Dict) -> "TommyPipeline":
        """
        Returns a TommyPipeline object instantiated with the given parameters
        :param params: Dictionary of parameters needed to instantiate a Pipeline object
        :return: A TommyPipeline object
        """
        pipeline = Pipeline.from_config(params)
        return TommyPipeline(pipeline.dataset_builder, pipeline.metric, pipeline.model, pipeline.params)

    def train_proxy(self):
        """
        Trains the proxy model on ASINs' names
        """
        dataset_obj = self.dataset_builder.get_dataset()
        self.model.train(dataset_obj.train_features, dataset_obj.train_labels, word_level=True)
        predicted_labels_dev = self.model.predict(dataset_obj.dev_features, dataset_obj.dev_labels, word_level=True)
        print("predicted dev labels", predicted_labels_dev)
        conf_matrix = confusion_matrix(dataset_obj.dev_labels, predicted_labels_dev,
                                       labels=self.dataset_builder.att_values)
        # Estimating p and q
        tn, fp, fn, tp = conf_matrix.ravel()
        self.q = fn / (tp + fn)  # empirical q
        self.p = fp / (tn + fp)  # empirical p
        print(f"p: {self.p}, q:{self.q}")

        # Estimating beta
        is_sensitive_train = np.array(dataset_obj.train_labels == self.dataset_builder.sens_att_value).astype(int)
        s_estimate = np.sum(is_sensitive_train) / is_sensitive_train.size
        self.beta = s_estimate

    def run_experiments(self) -> dict:
        """
        Runs experiments given the settings and return
        :return: A dictionary containing statistics of the experiments
        """

        # Gather data
        rankings_list = self.dataset_builder.get_lists(self.params["pipeline.percent_queries"])

        proxy_values = []
        corrected_values = []
        true_values = []

        # Measure average fairness of rankings
        for ranking in rankings_list:
            if len(ranking["labels"]) < 1:
                continue
            ranking_labels = np.array(ranking["labels"])
            # Compute true fairness of the list
            orig_is_sensitive = np.array(
                ranking_labels == self.dataset_builder.sens_att_value
            ).astype(int)
            positions = -np.array(ranking["positions"])

            # Exclude rankings where all the true labels are 0 or 1
            if int(np.sum(orig_is_sensitive)) == 0 or int(np.sum(orig_is_sensitive)) == orig_is_sensitive.shape[0]:
                continue
            true_value = self.metric.compute(positions, orig_is_sensitive, self.beta)

            # Compute the proxy value
            features = pd.DataFrame()
            features['input_string'] = pd.Series(ranking["names"])
            predicted_labels = self.model.predict(features, ranking_labels, word_level=True)
            is_sensitive = np.array(
                predicted_labels == self.dataset_builder.sens_att_value
            ).astype(int)

            # Exclude rankings where all the predicted labels are 0 or 1
            if int(np.sum(is_sensitive)) == 0 or int(np.sum(is_sensitive)) == is_sensitive.shape[0]:
                continue
            estimated_value = self.metric.compute(positions, is_sensitive, self.beta)

            # Compute the corrected value
            corrected_value = self.metric.correct(estimated_value, self.q, self.p, self.beta)

            proxy_values.append(estimated_value)
            corrected_values.append(corrected_value)
            true_values.append(true_value)

        # Computing the average estimates
        print("len of rankings_list", len(rankings_list), "in exp internal")
        true_value = np.mean(np.array(true_values))
        estimated_value = np.mean(np.array(proxy_values))
        corrected_value = np.mean(np.array(corrected_values))

        stat_values = {'model_name': self.model_name,
                       'true_value': true_value, 'estimated_value': estimated_value,
                       'corrected_value': corrected_value,
                       'metric_name': self.metric_name}

        return stat_values


def main():
    # Getting the config file
    parser = argparse.ArgumentParser(
        prog="experiments_pipeline",
        description="Run all experiments for https://quip-amazon.com/w0xzADvWGcbo"
                    "/Measuring-Fairness-in-Ranking-under-the-Uncertainty-Assumption",
    )
    default_config = Path(__file__).parent.parent.joinpath("config_files").joinpath("dataset_config_real.json")
    parser.add_argument(
        "--config_file", help="Location of the config file", default=str(default_config)
    )
    args = parser.parse_args()
    config_file = args.config_file

    # Read params
    with open(config_file) as json_file:
        params = json.load(json_file)
    print(params)

    var2loop = params["pipeline.variable2loop"]
    val2loop = params["pipeline.values2loop"]
    var2loop_name = params["pipeline.variable2loop_name"]
    num_exp = params["pipeline.num_exp"]
    params["dataset.path"] = Path(__file__).parent.joinpath("data").joinpath(params["dataset.path"])

    stats_list = []
    results_obj = Results()

    pipeline = TommyPipeline.from_config(params)
    pipeline.setup_tommy()
    pipeline.train_proxy()

    # Run experiments
    for val in val2loop:
        pipeline.params[var2loop] = val
        # pipeline.params['metric.name'] = 'exposure'

        print(val, pipeline.dataset_builder.selected_features)

        for i in range(num_exp):
            stat_values = pipeline.run_experiments()
            true_value = stat_values['true_value']
            estimated_value = stat_values['estimated_value']
            corrected_value = stat_values['corrected_value']
            results_obj.update(true_value, estimated_value, corrected_value)
            print(f"{val}: {true_value}, {estimated_value}, {corrected_value}")
            stats_list.append(dict(stat_values))
            print("")
        results_obj.make_list()
    results_obj.set_plot_info(val2loop, var2loop_name, pipeline.metric.full_name)

    fig_name = ("tommy_variable2loop={var2loop}.png")

    # Write results
    d = pd.DataFrame(stats_list)
    res_path = Path(__file__).parent.joinpath("results").joinpath("tommy_real.csv")
    d.to_csv(res_path)

    # Plot the results
    plot_tommy(fig_name, results_obj)


if __name__ == "__main__":
    main()