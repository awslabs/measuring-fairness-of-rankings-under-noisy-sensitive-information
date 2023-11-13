import numpy as np

from pathlib import Path
from matplotlib import pyplot as plt
from fairness_measurement_uncertainty.experiments.results import Results

y_label_font_size = 8
x_label_font_size = 8
title_size = 8
labelsize = 5
dpi_val = 800
meanprops={"marker":"x",
            "markerfacecolor":"black",
            "markersize":"2"}
medianprops_c2 = {"color": "darkblue"}


def plot_error_boxes(ax, list1: list, list2: list, x_labels, x_name, y_name, set_x_label=False, set_y_label=False,
                     set_legend=False, legends=["Proxy measurement", "Corrected (OURS)"], showmeans=True):
    """
    Plots the error_proxy and error_rec
    :param ax: Matplotlib ax
    :param list1: First set of data to be plotted
    :param list2: Second set of data to be plotted
    :param x_labels: Labels for x axis
    :param x_name: Name of x axis
    :param y_name: Name of y axis
    :param set_x_label: Set the x label if true
    :param set_y_label: Set the y label if true
    :param set_legend: Set the legend if true
    :param legends: The legend strings
    :param showmeans: Show mean values only if it is True
    """

    # ******** plot setting ********
    width = 6
    num_x = len(x_labels)
    left_positions = [i for i in range(1, 1 + num_x * width, width)]
    right_positions = [i + 2 for i in range(1, 1 + num_x * width, width)]
    c1 = "pink"
    c2 = "lightblue"
    bp1 = ax.boxplot(list1, showfliers=False, positions=left_positions,
                     boxprops=dict(facecolor=c1, color=c1), patch_artist=True, notch=False, widths=1.5,
                     showmeans=showmeans, meanprops=meanprops)
    bp2 = ax.boxplot(list2, showfliers=False, positions=right_positions,
                     boxprops=dict(facecolor=c2, color=c2), patch_artist=True, notch=False, widths=1.5,
                     showmeans=showmeans, meanprops=meanprops, medianprops=medianprops_c2)

    xtick_values = [(left_positions[i] + right_positions[i]) / 2 for i in range(num_x)]
    ax.set_xticks(xtick_values)

    # choose format of the xtick labels
    if x_name == "n":
        # Change format of exponentially increasing values
        ax.set_xticklabels(['{:.0e}'.format(x) for x in x_labels])
    else:
        ax.set_xticklabels(x_labels)

    ax.tick_params(labelsize=labelsize, axis='both')
    # axes[0, index].set_xlabel(f"{res.x_name}", fontsize=10)
    if set_x_label:
        ax.set_xlabel(f"{x_name}", fontsize=x_label_font_size)
    if set_y_label:
        ax.set_ylabel('Bias measurement error', fontsize=y_label_font_size)
    ax.set_title(r'$' + y_name + '$', fontsize=title_size)
    if set_legend:
        ax.legend([bp1["boxes"][0], bp2["boxes"][0]], legends,
                              loc='lower right', prop={'size': 6})
    ax.grid(axis='y')
    ax.set_box_aspect(1)


def plot_error_ratio(ax, res: Results, set_x_label=False, set_y_label=False, set_legend=False):
    """
    Plots the error_diff
    :param ax: Matplotlib ax
    :param res: Data to be plotted
    :param set_x_label: Set the x label if true
    :param set_y_label: Set the y label if true
    :param set_legend: Set the legend if true
    """

    # ******** plot setting ********
    width = 6
    num_x = len(res.x_labels)
    left_positions = [i for i in range(1, 1 + num_x * width, width)]
    right_positions = [i + 2 for i in range(1, 1 + num_x * width, width)]
    c1 = "lightgreen"
    xtick_values = [(left_positions[i] + right_positions[i]) / 2 for i in range(num_x)]
    ax.boxplot(res.errer_ratio_values_list, showfliers=False, boxprops=dict(facecolor=c1, color=c1),
               patch_artist=True, positions=xtick_values, notch=False, widths=2)

    ax.set_xticks(xtick_values)
    ax.set_xticklabels(res.x_labels)
    ax.tick_params(labelsize=labelsize, axis='both')
    if set_x_label:
        ax.set_xlabel(f"{res.x_name}", fontsize=x_label_font_size)
    if set_y_label:
        ax.set_ylabel(r'$error_{ratio}$', fontsize=y_label_font_size)
    ax.grid(axis='y')
    ax.set_box_aspect(1)


def plot_all_metrics_results(fig_name: str, results: [Results], error_diff: bool = True):
    """
    Plots results of all metrics
    :param fig_name: File name
    :param results: List containing results of different metrics
    :param error_diff: Plot both error values and error ratio if True, otherwise only plot the error values
    :return: Plot the results
    """
    num_rows = 2
    if not error_diff:
        num_rows = 1

    file_path = Path(__file__).parent.parent.parent.parent.joinpath("figures").joinpath(fig_name)
    fig, axes = plt.subplots(num_rows, len(results))
    fig.subplots_adjust(wspace=0.3)
    index = 0
    plt.tick_params(axis='y', which='major', labelsize=labelsize)

    for res in results:
        set_x_label = False
        set_y_label = False
        set_legend = False
        # ******** plot estimated and corrected values together ********
        if index == 0:
            set_y_label = True
        if index == 1 and not error_diff:
            set_x_label = True
        if index == 2:
            set_legend = True
        if error_diff:
            plot_error_boxes(axes[0, index], res.estimated_diff_values_list, res.corrected_diff_values_list,
                             res.x_labels, res.x_name, res.y_name, set_x_label, set_y_label, set_legend)
        else:
            plot_error_boxes(axes[index], res.estimated_diff_values_list, res.corrected_diff_values_list,
                             res.x_labels, res.x_name, res.y_name, set_x_label, set_y_label, set_legend, showmeans=False)
            axes[index].set_box_aspect(1)

        # ******** plot ratio of correct_diff to estimated_diff ********
        if error_diff:
            set_x_label = False
            set_y_label = False
            if index == 1:
                set_x_label = True
            if index == 0:
                set_y_label = True
            plot_error_ratio(axes[1, index], res, set_x_label, set_y_label, set_legend)

        index += 1

    # fig.align_ylabels(axes[0, :])
    # fig.align_ylabels(axes[1, :])
    # fig.tight_layout()
    print(file_path)
    fig.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.savefig(file_path, bbox_inches='tight', dpi=dpi_val)


# def plot_baseline(fig_name: str, res: Results):
    # file_path = Path(__file__).parent.parent.parent.parent.joinpath("figures").joinpath(fig_name)
    # fig, ax = plt.subplots()
    # fig.subplots_adjust(wspace=0.3)
    # plt.tick_params(axis='y', which='major', labelsize=5)
    # plot_error_boxes(ax, res, True, True, True, ["estimated (SoTA baseline)", "recovered"])
    # ax.set_box_aspect(1)
    # print(file_path)
    # plt.savefig(file_path, bbox_inches='tight')


def plot_baseline(fig_name: str, corrected_res: Results, baseline_res: Results):
    """
    Plots the baseline against our method
    :param fig_name: Name of the output figure
    :param corrected_res: Results object containing the corrected values of the bias
    :param baseline_res: Results object containing the baseline estimates
    """

    file_path = Path(__file__).parent.parent.parent.parent.joinpath("figures").joinpath(fig_name)
    fig, ax = plt.subplots()
    fig.subplots_adjust(wspace=0.3)
    plot_error_boxes(ax, baseline_res.estimated_diff_values_list, corrected_res.corrected_diff_values_list,
                     baseline_res.x_labels, baseline_res.x_name, baseline_res.y_name,
                     True, True, True, ["Weighted sampling (baseline)", "Corrected (OURS)"])
    ax.tick_params(axis='both', labelsize=labelsize+7)
    ax.set_xlabel("Sampling rate", fontsize=x_label_font_size+7)
    ax.set_ylabel('Bias measurement error', fontsize=y_label_font_size+7)
    ax.set_title(r'$' + baseline_res.y_name + '$', fontsize=title_size+7)
    plt.rcParams['legend.title_fontsize'] = 16
    print(file_path)
    plt.savefig(file_path, bbox_inches='tight')


def plot_tommy(fig_name: str, res: Results):
    """
    Plots the Tommy ASIN results
    :param fig_name: Name of the figure
    :param res: Results object containing the estimates for different sampling rates
    """

    file_path = Path(__file__).parent.parent.parent.parent.joinpath("figures").joinpath(fig_name)
    fig, ax = plt.subplots(2)
    fig.subplots_adjust(hspace=0.2)
    plot_error_boxes(ax[0], res.estimated_diff_values_list, res.corrected_diff_values_list,
                     res.x_labels, res.x_name, res.y_name,
                     True, True, False, ["Proxy measurement", "Corrected (OURS)"])
    plot_error_ratio(ax[1], res, True, True, True)
    ax[0].tick_params(axis='both', labelsize=labelsize+1)
    ax[1].tick_params(axis='both', labelsize=labelsize+1)

    # ax[0].set_xlabel("%queries", fontsize=x_label_font_size+1)
    ax[1].set_xlabel("%queries", fontsize=x_label_font_size+1)

    ax[0].set_ylabel('Bias measurement error', fontsize=y_label_font_size+1)
    ax[1].set_ylabel(r'$error_{ratio}$', fontsize=y_label_font_size+1)

    # ax[0].set_title(r'$' + res.y_name + '$', fontsize=title_size+7)
    # ax[1].set_title(r'$' + res.y_name + '$', fontsize=title_size + 7)

    #plt.rcParams['legend.title_fontsize'] = 16
    print(file_path)
    plt.savefig(file_path, bbox_inches='tight')


def plot_assumption_1(fig_name: str, results: [Results], dataset):
    """
    Plots results of assumption I
    :param fig_name: Name of the ourput figure
    :param results: List of Results objects containing the estimates for different metrics
    :param dataset: Name of the dataset
    """
    file_path = Path(__file__).parent.parent.parent.parent.joinpath("figures").joinpath(fig_name)
    fig, axes = plt.subplots(2)
    plt.tick_params(axis='y', which='major', labelsize=labelsize)

    # ******** plot setting ********
    width = 6
    num_x = 3
    left_positions = [i for i in range(1, 1 + num_x * width, width)]
    right_positions = [i + 2 for i in range(1, 1 + num_x * width, width)]
    xtick_values = [(left_positions[i] + right_positions[i]) / 2 for i in range(num_x)]
    c1 = "pink"
    c2 = "lightblue"
    c3 = 'lightgreen'
    metric_names = [r"$DP$", r"$Exp$", r"$rND$"]

    # plot error
    index = 0
    for res in results:
        bp1 = axes[0].boxplot(res.estimated_diff_values_list, showfliers=False, positions=[left_positions[index]],
                              boxprops=dict(facecolor=c1, color=c1), patch_artist=True, notch=False, widths=1,
                              showmeans=True, meanprops=meanprops)
        bp2 = axes[0].boxplot(res.corrected_diff_values_list, showfliers=False, positions=[right_positions[index]],
                              boxprops=dict(facecolor=c2, color=c2), patch_artist=True, notch=False, widths=1,
                              showmeans=True, meanprops=meanprops, medianprops=medianprops_c2)
        axes[0].legend([bp1["boxes"][0], bp2["boxes"][0]], ["Proxy measurement", "Corrected (OURS)"],
                       loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6})
        index += 1
    axes[0].set_xticks(xtick_values)
    axes[0].set_xticklabels(metric_names)
    axes[0].tick_params(labelsize=labelsize+1, axis='both')
    # axes[0, index].set_xlabel(f"{res.x_name}", fontsize=10)
    axes[0].set_ylabel('Bias measurement error', fontsize=y_label_font_size)
    axes[0].set_title(dataset, fontsize=title_size)
    axes[0].grid(axis='y')
    axes[0].set_box_aspect(1)

    # plot error ratio
    index = 0
    for res in results:
        axes[1].boxplot(res.errer_ratio_values_list, showfliers=False, boxprops=dict(facecolor=c3, color=c3),
                        patch_artist=True, positions=[xtick_values[index]], notch=False, widths=2)
        index += 1

    axes[1].tick_params(labelsize=labelsize+1, axis='both')
    axes[1].set_xticks(xtick_values)
    axes[1].set_xticklabels(metric_names)
    axes[1].set_xlabel("Metric name", fontsize=x_label_font_size)
    axes[1].set_ylabel(r'$error_{ratio}$', fontsize=y_label_font_size)
    axes[1].grid(axis='y')
    axes[1].set_box_aspect(1)

    print(file_path)
    plt.savefig(file_path, bbox_inches='tight', dpi=dpi_val)


def stats_report(stats: [dict]):
    """
    Prints statistics about the whole experiments
    :param stats: list of stats for each experiment
    """
    # {'test_res': assumption_test_result, 'model_name': self.model_name,
    #  'g1_error': emp_g1_error_rate, 'g2_error': emp_g2_error_rate}

    model_stats = {}
    metric_error_stats = {}
    for stat in stats:
        model_name = stat['model_name']
        g1_error = stat['g1_error']
        g2_error = stat['g2_error']
        accuracy = stat['accuracy']

        if model_name not in model_stats:
            model_stats[model_name] = {'g1_error': [], 'g2_error': [], 'accuracy': [],
                                       'count_90': 0, 'count_95': 0, 'total_count': 0.0}
        model_stats[model_name]['g1_error'].append(g1_error)
        model_stats[model_name]['g2_error'].append(g2_error)
        model_stats[model_name]['accuracy'].append(accuracy)
        if stat['test_res'] > 0.1:
            model_stats[model_name]['count_90'] += 1
            model_stats[model_name]['count_95'] += 1
        elif stat['test_res'] > 0.05:
            model_stats[model_name]['count_95'] += 1
        model_stats[model_name]['total_count'] += 1

        metric_model_str = stat['metric_name'] + ', ' + model_name
        if metric_model_str not in metric_error_stats:
            metric_error_stats[metric_model_str] = {'error_est': [], 'error_rec': [], 'error_diff': []}
        true_value = stat['true_value']
        estimated_value = stat['estimated_value']
        corrected_value = stat['corrected_value']
        error_est = true_value - estimated_value
        error_rec = true_value - corrected_value
        error_diff = abs(true_value - corrected_value) - abs(true_value - estimated_value)
        metric_error_stats[metric_model_str]['error_est'].append(error_est)
        metric_error_stats[metric_model_str]['error_rec'].append(error_rec)
        metric_error_stats[metric_model_str]['error_diff'].append(error_diff)

    for e in model_stats:
        avg_g1_error = np.mean(np.asarray(model_stats[e]['g1_error']))
        avg_g2_error = np.mean(np.asarray(model_stats[e]['g2_error']))
        avg_accuracy = np.mean(np.asarray(model_stats[e]['accuracy']))
        total_count = model_stats[e]['total_count']
        portion_90 = model_stats[e]['count_90'] / total_count
        portion_95 = model_stats[e]['count_95'] / total_count
        print(e, avg_g1_error, avg_g2_error, avg_accuracy, portion_90, portion_95)

    for e in metric_error_stats:
        a = np.asarray(metric_error_stats[e]['error_est'])
        b = np.asarray(metric_error_stats[e]['error_rec'])
        c = np.asarray(metric_error_stats[e]['error_diff'])
        print(e, np.mean(a), np.std(a), np.mean(b), np.std(b), np.mean(c), np.std(c))
