import argparse
import copy
import json

from pathlib import Path
from typing import Dict
from experiments.all_metrics_results import plot_all_metrics_results
from experiments.results import Results
from experiments.experiments_pipeline import Pipeline


class SyntheticPipeline(Pipeline):
    def __init__(self, dataset_builder, metric, model, params: Dict):
        super().__init__(dataset_builder, metric, model, params)

    @staticmethod
    def from_config(params: Dict) -> "SyntheticPipeline":
        """
        Returns a SyntheticPipeline object instantiated with the given parameters
        :param params: Dictionary of parameters needed to instantiate a Pipeline object
        :return: A SyntheticPipeline object
        """

        pipeline = Pipeline.from_config(params)
        return SyntheticPipeline(pipeline.dataset_builder, pipeline.metric, pipeline.model, pipeline.params)


def main():
    # Getting the config file
    parser = argparse.ArgumentParser(
        prog="experiments_pipeline",
        description="Run all experiments for https://quip-amazon.com/w0xzADvWGcbo"
                    "/Measuring-Fairness-in-Ranking-under-the-Uncertainty-Assumption",
    )
    default_config = Path(__file__).parent.parent.joinpath("config_files").\
        joinpath("dataset_config_simulated.json")
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
    worldview = params["worldview"]
    metric_name = params['metric.name']

    # Run experiments - no explicit loop over different metrics
    # results_obj = Results()
    # for val in val2loop:
    #     localparams = copy.deepcopy(params)
    #     localparams[var2loop] = val
    #     print(f"val2loop: {val}")
    #
    #     pipeline = SyntheticPipeline.from_config(localparams)
    #
    #     for i in range(num_exp):
    #         true_value, estimated_value, corrected_value = pipeline.run_experiment()
    #         results_obj.update(true_value, estimated_value, corrected_value)
    #         print(f"{true_value:.3f}, {estimated_value:.3f}, {corrected_value:.3f}")
    #     results_obj.make_list()
    #
    # fig_name = (
    #     f"simulated_metric={metric_name}_variable2loop={var2loop}.png"
    # )
    # results_obj.set_plot_info(val2loop, var2loop_name, pipeline.metric.full_name)
    # results_obj.plot_results(fig_name)

    # Run experiments
    res_obj_list = []
    metrics = ["dp", "exposure", "rnd"]
    for metric_name in metrics:
        results_obj = Results()
        for val in val2loop:
            localparams = copy.deepcopy(params)
            localparams[var2loop] = val
            # print(f"val2loop: {val}")
            localparams['metric.name'] = metric_name

            pipeline = SyntheticPipeline.from_config(localparams)

            for i in range(num_exp):
                stat_values = pipeline.run_experiment()
                true_value = stat_values['true_value']
                estimated_value = stat_values['estimated_value']
                corrected_value = stat_values['corrected_value']
                results_obj.update(true_value, estimated_value, corrected_value)
                # print(f"{true_value:.3f}, {estimated_value:.3f}, {corrected_value:.3f}")
            results_obj.make_list()
        results_obj.set_plot_info(val2loop, var2loop_name, pipeline.metric.full_name)
        res_obj_list.append(results_obj)
    fig_name = (
        f"sim_variable_{var2loop}_assumption_{worldview}.png"
    )

    # Plot results
    plot_all_metrics_results(fig_name, res_obj_list, error_diff=False)


if __name__ == "__main__":
    main()