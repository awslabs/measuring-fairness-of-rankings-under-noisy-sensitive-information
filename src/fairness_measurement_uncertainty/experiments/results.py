from pathlib import Path

import matplotlib.pyplot as plt


class Results():
    def __init__(self):
        self.estimated_diff_values = []
        self.corrected_diff_values = []
        self.errer_ratio_values = []

        self.estimated_diff_values_list = []
        self.corrected_diff_values_list = []
        self.errer_ratio_values_list = []

        self.x_labels = []
        self.x_name = ''
        self.y_name = ''

    def set_plot_info(self, x_labels, x_name, y_name):
        self.x_labels = x_labels
        self.x_name = x_name
        self.y_name = y_name

    def update(self, true_value: float, estimated_value: float, corrected_value: float):
        """
        Computes metrics
        :param true_value: True value
        :param estimated_value: Proxy estimate
        :param corrected_value: Corrected bias
        """

        ratio_value = abs(true_value - corrected_value) / (abs(true_value - estimated_value) + 0.000001)
        self.estimated_diff_values.append(true_value - estimated_value)
        self.corrected_diff_values.append(true_value - corrected_value)
        self.errer_ratio_values.append(ratio_value)

    def update_baseline(self, true_value: float, estimated_value: float):
        """
        Computes metrics for the baseline estimates
        :param true_value: True value
        :param estimated_value: Baseline estimate
        """
        self.estimated_diff_values.append(true_value - estimated_value)

    def make_list(self, n: int = 1):
        """
        Adds the set of metric values to a list and creates a new set
        :param n: Number of times to repeat the lists (used for plotting the baseline)
        """
        for i in range(n):
            self.estimated_diff_values_list.append(list(self.estimated_diff_values))
            self.corrected_diff_values_list.append(list(self.corrected_diff_values))
            self.errer_ratio_values_list.append(list(self.errer_ratio_values))
        self.estimated_diff_values = []
        self.corrected_diff_values = []
        self.errer_ratio_values = []

    def make_list_baseline(self):
        self.estimated_diff_values_list.append(list(self.estimated_diff_values))
        self.estimated_diff_values = []

    def plot_results(self, figname: str):
        """
        Plots the results (no longer used - see all_metrics_results.py)
        :param figname: Name of the output figure
        :return:
        """

        # ******** plot setting ********
        width = 6
        num_x = len(self.x_labels)
        left_positions = [i for i in range(1, 1 + num_x * width, width)]
        right_positions = [i + 2 for i in range(1, 1 + num_x * width, width)]

        # ******** plot ratio of correct_diff to estimated_diff ********
        fig, ax = plt.subplots()
        c1 = "blue"
        xtick_values = [(left_positions[i] + right_positions[i]) / 2 for i in range(num_x)]
        ax.boxplot(self.errer_ratio_values_list, showfliers=False,
                         boxprops=dict(facecolor=c1, color=c1), patch_artist=True, positions=xtick_values)

        ax.set_xticks(xtick_values)
        ax.set_xticklabels(self.x_labels)
        ax.set_xlabel(f"{self.x_name}", fontsize=14)
        ax.set_ylabel(f'{self.y_name}: difference of absolute values of errors', fontsize=14)
        plt.grid(True)
        file_path = Path(__file__).parent.parent.parent.parent.joinpath("figures").joinpath("0_" + figname)
        plt.savefig(file_path, bbox_inches='tight')

        # ******** plot estimated and corrected values together ********
        fig, ax = plt.subplots()
        c1 = "red"
        c2 = "blue"
        bp1 = ax.boxplot(self.estimated_diff_values_list, showfliers=False, positions=left_positions,
                         boxprops=dict(facecolor=c1, color=c1), patch_artist=True)
        bp2 = ax.boxplot(self.corrected_diff_values_list, showfliers=False, positions=right_positions,
                         boxprops=dict(facecolor=c2, color=c2), patch_artist=True)

        xtick_values = [(left_positions[i] + right_positions[i]) / 2 for i in range(num_x)]
        ax.set_xticks(xtick_values)
        ax.set_xticklabels(self.x_labels)
        ax.set_xlabel(f"{self.x_name}", fontsize=14)
        ax.set_ylabel(f'{self.y_name}: difference with true values', fontsize=14)
        ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["Difference of estimated and true values",
                                                       "Difference of corrected and true values"])
        plt.grid(True)
        file_path = Path(__file__).parent.parent.parent.parent.joinpath("figures").joinpath(figname)
        plt.savefig(file_path, bbox_inches='tight')


class BaselineResults(Results):
    def __init__(self, n):
        super(BaselineResults, self).__init__()
        self.estimated_diff_values_list = [[] for i in range(n)]
        self.n = n  # Size of the variable values (e.g., number of sampling rates)

    def update_results(self, results, true_value):
        for i in range(self.n):
            self.estimated_diff_values_list[i].append(true_value - results[i])