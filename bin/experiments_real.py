import argparse
import copy
import json
import pandas as pd

from pathlib import Path
from typing import Dict
from experiments.all_metrics_results import plot_all_metrics_results, stats_report, plot_baseline, plot_assumption_1
from experiments.results import Results
from experiments.experiments_pipeline import Pipeline


class RealPipeline(Pipeline):
    def __init__(self, dataset_builder, metric, model, params: Dict):
        super().__init__(dataset_builder, metric, model, params)

    def setup(self, n, range, worldview: int = 1):
        """
        Reads the raw dataset and performs the necessary feature selection/feature importance computation
        :param n: Number of features to be selected for the proxy model
        :param range: Number of features to skip in the list of important features for predicting scores
        :param worldview: 1 is for Assumption I, and 2 for Assumption II
        """
        self.dataset_builder.read_dataset()
        self.dataset_builder.set_predictive_features()
        # Feature selection / computation of feature importance
        # self.dataset_builder.feature_selection_importance(self.model.model_python, n, range)
        # if worldview == 1:
        #     self.dataset_builder.feature_selection_importance(self.model.model_python, n, range)
        #     # self.dataset_builder.feature_selection()
        # # elif worldview == 2:
        # #     self.dataset_builder.overlapping_feature_selection(self.model.model_python, n, range)

    @staticmethod
    def from_config(params: Dict) -> "RealPipeline":
        """
        Returns a RealPipeline object instantiated with the given parameters
        :param params: Dictionary of parameters needed to instantiate a Pipeline object
        :return: A RealPipeline object
        """
        pipeline = Pipeline.from_config(params)
        return RealPipeline(pipeline.dataset_builder, pipeline.metric, pipeline.model, pipeline.params)


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
    print(f"dataset path: {params['dataset.path']}")

    # Run experiments
    res_obj_list = []
    stats_list = []
    metrics = ["dp", "exposure", "rnd"]
    for metric_name in metrics:
        results_obj = Results()
        for val in val2loop:
            localparams = copy.deepcopy(params)
            localparams[var2loop] = val
            localparams['metric.name'] = metric_name

            pipeline = RealPipeline.from_config(localparams)
            pipeline.setup(localparams["feature.n"], localparams["feature.range"], worldview=worldview)
            print(val, pipeline.dataset_builder.selected_features)

            for i in range(num_exp):
                stat_values = pipeline.run_experiment()
                true_value = stat_values['true_value']
                estimated_value = stat_values['estimated_value']
                corrected_value = stat_values['corrected_value']
                results_obj.update(true_value, estimated_value, corrected_value)
                print(f"{val}: {true_value}, {estimated_value}, {corrected_value}")
                stats_list.append(dict(stat_values))
                print("")
            results_obj.make_list()
        results_obj.set_plot_info(val2loop, var2loop_name, pipeline.metric.full_name)
        res_obj_list.append(results_obj)
    fig_name = (
        f"{dataset}_variable2loop={var2loop}.png"
    )

    # Write results
    d = pd.DataFrame(stats_list)
    res_path = Path(__file__).parent.joinpath("results").joinpath(f"{dataset}_real.csv")
    d.to_csv(res_path)

    # Plot results
    if worldview == 1:
        if dataset == 'fifa':
            dataset = "FIFA Players"
        elif dataset == 'goodreads':
            dataset = "Goodreads Authors"
        plot_assumption_1(fig_name, res_obj_list, dataset)
    else:
        plot_all_metrics_results(fig_name, res_obj_list)

    # Print stats
    stats_report(stats_list)


if __name__ == "__main__":
    main()
