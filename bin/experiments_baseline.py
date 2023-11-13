import argparse
import copy
import json
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Dict
from sklearn.metrics import confusion_matrix
from experiments.all_metrics_results import plot_baseline
from experiments.results import Results, BaselineResults
from experiments_real import RealPipeline


class BaselinePipeline(RealPipeline):
    def __init__(self, dataset_builder, metric, model, params: Dict):
        super().__init__(dataset_builder, metric, model, params)

    @staticmethod
    def from_config(params: Dict) -> "BaselinePipeline":
        """
        Returns a BaselinePipeline object instantiated with the given parameters
        :param params: Dictionary of parameters needed to instantiate a Pipeline object
        :return: A BaselinePipeline object
        """
        pipeline = RealPipeline.from_config(params)
        return BaselinePipeline(pipeline.dataset_builder, pipeline.metric, pipeline.model, pipeline.params)

    def run_baseline(self, var2loop, val2loop) -> dict:
        """
        Run the baseline given the settings and return
        :return: A dictionary containing statistics of the experiments
        """

        # Gather data
        dataset_obj = self.dataset_builder.get_dataset()
        orig_is_sensitive = np.array(
            dataset_obj.test_labels == self.dataset_builder.sens_att_value
        ).astype(int)

        # Estimating population demographics
        is_sensitive_train = np.array(dataset_obj.train_labels == self.dataset_builder.sens_att_value).astype(
            int
        )
        s_estimate = np.sum(is_sensitive_train) / is_sensitive_train.size

        worldview = int(self.params["worldview"])

        # Setting features for predicting sensitive attributes
        sensitive_pred_features = self.dataset_builder.sensitive_pred_features

        # Predicting labels in development and test data
        print(dataset_obj.train_features.columns.values)
        self.model.train(dataset_obj.train_features[sensitive_pred_features], dataset_obj.train_labels)

        # Predict labels in development and test data
        self.model.train(dataset_obj.train_features[sensitive_pred_features], dataset_obj.train_labels)
        predicted_labels_test = self.model.predict(dataset_obj.test_features_drop_scores[sensitive_pred_features],
                                                   dataset_obj.test_labels)
        predicted_labels_dev = self.model.predict(dataset_obj.dev_features[sensitive_pred_features],
                                                  dataset_obj.dev_labels)

        # Update scores should they be predicted by LTR models
        if worldview == 2:
            score_pred_features = self.dataset_builder.score_pred_features
            # add A hat to the train
            predicted_labels_train = self.model.predict(dataset_obj.train_features[sensitive_pred_features],
                                                        dataset_obj.train_labels)
            score_train = pd.DataFrame(dataset_obj.train_features[score_pred_features])
            score_train["sensitive_hat"] = predicted_labels_train
            self.update_scores(dataset_obj, score_train, score_pred_features)

        # Compute baseline
        positions = -np.array(dataset_obj.test_features[self.dataset_builder.score_field])
        baseline_results = []
        for val in val2loop:
            self.params[var2loop] = val
            sample_rate = self.params["baseline.sample_rate"]
            sample_weighting = self.params["baseline.sample_weighting"]
            m = self.params["baseline.bucket_size"]
            baseline_value = self.metric.compute_baseline(positions, orig_is_sensitive,
                                                          sample_rate, sample_weighting, m)
            baseline_results.append(baseline_value)

        # Compute actual fairness
        true_value = self.metric.compute(positions, orig_is_sensitive, s_estimate)

        # Computing the demographic parity according to predictions of the proxy model
        is_sensitive_dev = np.array(
            predicted_labels_dev == self.dataset_builder.sens_att_value
        ).astype(int)
        s_prime = np.sum(is_sensitive_dev) / is_sensitive_dev.size

        # Compute the proxy fairness
        is_sensitive = np.array(
            predicted_labels_test == self.dataset_builder.sens_att_value
        ).astype(int)
        positions = -np.array(dataset_obj.test_features[self.dataset_builder.score_field])
        estimated_value = self.metric.compute(positions, is_sensitive, s_prime)

        # Bias correction
        conf_matrix = confusion_matrix(dataset_obj.dev_labels, predicted_labels_dev,
                                       labels=self.dataset_builder.att_values)
        tn, fp, fn, tp = conf_matrix.ravel()
        emp_g1_error_rate = fn / (tp + fn)  # empirical q
        emp_g2_error_rate = fp / (tn + fp)  # empirical p
        corrected_value = self.metric.correct(estimated_value, emp_g1_error_rate, emp_g2_error_rate, s_estimate)

        stat_values = {'model_name': self.model_name,
                       'true_value': true_value, 'estimated_value': estimated_value,
                       'corrected_value': corrected_value,
                       'metric_name': self.metric_name, 'baseline_results': baseline_results}

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
    worldview = int(params['worldview'])
    dataset = params["dataset.name"]
    params["dataset.path"] = Path(__file__).parent.joinpath("data").joinpath(params["dataset.path"])

    results_obj = Results()
    results_obj_baseline = BaselineResults(n=len(val2loop))

    # Run experiments
    localparams = copy.deepcopy(params)
    localparams['metric.name'] = "exposure"
    pipeline = BaselinePipeline.from_config(localparams)
    pipeline.setup(localparams["feature.n"], localparams["feature.range"], worldview=worldview)
    stat_values_list = []
    for i in range(num_exp):
        stat_values = pipeline.run_baseline(var2loop, val2loop)
        true_value = stat_values['true_value']
        estimated_value = stat_values['estimated_value']
        corrected_value = stat_values['corrected_value']
        baseline_results = stat_values['baseline_results']
        results_obj.update(true_value, estimated_value, corrected_value)
        results_obj_baseline.update_results(baseline_results, true_value)
        print(f"{true_value}, {estimated_value}, {corrected_value}")
        print("")
        stat_values_list.append(dict(stat_values))
    results_obj.make_list(n=len(val2loop))

    # Plot results
    fig_name = (
        f"{dataset}_baseline_variable2loop={var2loop}.png"
    )
    results_obj_baseline.set_plot_info(val2loop, var2loop_name, pipeline.metric.full_name)
    plot_baseline(fig_name, results_obj, results_obj_baseline)

    # Write results into a file
    d = pd.DataFrame(stat_values_list)
    res_path = Path(__file__).parent.joinpath("results").joinpath(f"{dataset}_baseline.csv")
    d.to_csv(res_path)


if __name__ == "__main__":
    main()
