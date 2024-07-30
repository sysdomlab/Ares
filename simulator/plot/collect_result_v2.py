import collections
import os
import subprocess
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict
from time import sleep

from matplotlib.patches import Patch

from plot.collect_result import get_data_from_raw_log, get_all_jct_from_raw_log, get_fair_jct_from_raw_log, \
    calculate_ftf, get_avg_jct, get_makespan, get_all_jct, calculate_ftf_in_algo, get_fair_jct, get_scheduling_data, \
    print_improve_reduce, algo_name, priority, trace_name
from policy.utils_gavel import get_gavel_policies


def plot_grouped_bar_v2(ax, src, wl_set, metric, fontsize=32, legend_fontsize=21):
    linewidth = 2
    markersize = 10

    # Grouping the data by 'workload' and 'algo' and calculating mean JCT
    grouped_df = src.groupby(['workload', 'algo'])['res'].mean().unstack()

    grouped_df = grouped_df.reindex(columns=sorted(grouped_df.columns, key=lambda x: priority[algo_name[x]]))

    # Plotting the bar chart
    grouped_df.plot(kind='bar', ax=ax, legend=False)

    if metric == "Unfair Fraction":
        ax.set_ylim(0, 1)

    if metric == "Worst FTF":
        # Use logarithmic scale for y-axis
        ax.set_yscale('log')

        # Limiting the y-axis to the range 0 to 100
        # ax.set_ylim(0, 100)
        yticks = [float(f"{i:.1f}") for i in [1, 10, 100]]
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks, fontsize=fontsize, color='black')

    # Adding labels and title
    ax.title.set_text(f"{trace_name[wl_set]}")
    ax.title.set_fontsize(fontsize)
    ax.title.set_color('black')
    ax.set_xlabel('Trace ID', fontsize=fontsize, color='black')
    ax.set_ylabel(f'{metric}', fontsize=fontsize, color='black')
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=fontsize, color='black')
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(ax.get_yticks(), fontsize=fontsize, color='black')

    # Set the yticks
    yticks = ax.get_yticks()
    stride = len(yticks) // 7 + 1
    yticks = [float(f"{i:.1f}") for i in yticks[::stride]]
    ax.set_yticks(yticks)  # Show every 2nd y-tick for example
    ax.set_yticklabels(yticks, fontsize=fontsize, color='black')


def fig2_sim_all_jct_and_ftf_v2():
    workload_sets = ["philly", "saturn", "newtrace"]
    save_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "combined_plot")
    fontsize = 30
    figsize = (33, 5)
    rect = [0, 0, 1, 0.9]

    # 1. avg jct
    plt.style.use('ggplot')
    fig, axs = plt.subplots(1, 3, figsize=figsize)  # Create 1 row and 3 columns of subplots

    for i, workload_set in enumerate(workload_sets):
        results_data = get_data_from_raw_log(workload_set, "avg_jct")
        df = pd.DataFrame(results_data)
        all_data, improve, reduce = print_improve_reduce(df, [])
        # print(reduce)
        # 取平均
        # print(reduce.mean())
        # print(reduce.mean().min())
        print(f"min_val: {reduce.mean().min()}, max_val: {reduce.mean().max()}")
        plot_grouped_bar_v2(axs[i], df, workload_set, "Avg. JCT (hrs)", fontsize)

    # Customize the legend
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=7, loc='upper center', bbox_to_anchor=(0.5, 1.06), fontsize=fontsize,
               frameon=False)

    plt.tight_layout(rect=rect)  # Adjust layout to prevent clipping of labels and legend
    plt.savefig(f"{save_path}_jct.jpg")
    plt.savefig(f"{save_path}_jct.pdf")
    plt.show()
    plt.clf()
    print(f"finish plot {save_path}")

    # 2. Unfair Job Fraction
    plt.style.use('ggplot')
    fig, axs = plt.subplots(1, 3, figsize=figsize)  # Create 1 row and 3 columns of subplots

    for i, workload_set in enumerate(workload_sets):
        real_jct = get_all_jct_from_raw_log(workload_set)
        fair_jct = get_fair_jct_from_raw_log(workload_set)
        ftf = calculate_ftf(real_jct, fair_jct)

        results_data = [
            {
                "workload": workload,
                "algo": algo,
                "res": sum([i > 1 for i in job.values()]) / len(job.values()),  # job.values() 中大于1的数量
            }
            for workload, algo_data in ftf.items()
            for algo, job in algo_data.items()
        ]

        df = pd.DataFrame(results_data)
        all_data, improve, reduce = print_improve_reduce(df, [])
        print(f"min_val: {reduce.mean().min()}, max_val: {reduce.mean().max()}")
        plot_grouped_bar_v2(axs[i], df, workload_set, "Unfair Fraction", fontsize)

    # Customize the legend
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=7, loc='upper center', bbox_to_anchor=(0.5, 1.06), fontsize=fontsize,
               frameon=False)

    plt.tight_layout(rect=rect)  # Adjust layout to prevent clipping of labels and legend
    plt.savefig(f"{save_path}_unfair.jpg")
    plt.savefig(f"{save_path}_unfair.pdf")
    plt.show()
    plt.clf()
    print(f"finish plot {save_path}")

    # 3. Worst finish time fairness
    plt.style.use('ggplot')
    fig, axs = plt.subplots(1, 3, figsize=figsize)  # Create 1 row and 3 columns of subplots

    for i, workload_set in enumerate(workload_sets):
        real_jct = get_all_jct_from_raw_log(workload_set)
        fair_jct = get_fair_jct_from_raw_log(workload_set)
        ftf = calculate_ftf(real_jct, fair_jct)

        results_data = [
            {
                "workload": workload,
                "algo": algo,
                "res": max(job.values()),
            }
            for workload, algo_data in ftf.items()
            for algo, job in algo_data.items()
        ]

        df = pd.DataFrame(results_data)
        all_data, improve, reduce = print_improve_reduce(df, [])
        print(df.groupby(['workload', 'algo'])['res'].mean().unstack()["Ares"])
        print(f"min_val: {reduce.mean().min()}, max_val: {reduce.mean().max()}")
        plot_grouped_bar_v2(axs[i], df, workload_set, "Worst FTF", fontsize)

    # Customize the legend
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=7, loc='upper center', bbox_to_anchor=(0.5, 1.06), fontsize=fontsize,
               frameon=False)

    plt.tight_layout(rect=rect)  # Adjust layout to prevent clipping of labels and legend
    plt.savefig(f"{save_path}_ftf.jpg")
    plt.savefig(f"{save_path}_ftf.pdf")
    plt.show()
    plt.clf()
    print(f"finish plot {save_path}")


def plot_grouped_bar_v3(ax, src, metric, colors, fontsize):
    # Grouping the data by 'algo' and calculating median and percentiles across all workloads
    grouped_df = src.groupby('algo')['res'].median()
    error_lower = src.groupby('algo')['res'].quantile(0.25)
    error_upper = src.groupby('algo')['res'].quantile(0.75)

    # Sorting columns by priority
    grouped_df = grouped_df.reindex(sorted(grouped_df.index, key=lambda x: priority[algo_name[x]]))
    error_lower = error_lower.reindex(grouped_df.index)
    error_upper = error_upper.reindex(grouped_df.index)

    # Plotting the bar chart with error bars
    bars = grouped_df.plot(kind='bar', ax=ax, legend=False, color=colors, width=0.5,
                           yerr=[grouped_df - error_lower, error_upper - grouped_df], capsize=4)

    if metric == "Unfair Fraction":
        ax.set_ylim(0, 1)

    ax.set_ylabel(f'{metric}', fontsize=fontsize, color='black')
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(ax.get_yticks(), fontsize=fontsize, color='black')

    # Remove x-axis ticks and labels
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_xlabel('', fontsize=fontsize, color='black')

    # Set the yticks
    yticks = ax.get_yticks()
    stride = len(yticks) // 7 + 1
    yticks = [float(f"{i:.1f}") for i in yticks[::stride]]
    ax.set_yticks(yticks)  # Show every n-th y-tick
    ax.set_yticklabels(yticks, fontsize=fontsize, color='black')

    return bars


def fig2_sim_all_jct_and_ftf_v3():
    algorithms = ["Ares", "Gavel", "Themis", "AlloX", "Tiresias", "Optimus", "Pollux"]
    colors = ['#e24a33', '#348abd', '#988ed5', '#777777', "#fbc15e", "#8eba41", "#ffb4b8"]
    workload_sets = ["philly", "saturn", "newtrace"]
    save_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "combined_plot")
    fontsize = 28

    metrics = ["Avg. JCT (hrs)", "P99 JCT (hrs)", "Unfair Fraction", "Worst FTF", "Makespan (hrs)",
               # "Avg. job restarts"
               ]

    plt.style.use('ggplot')

    for workload_set in workload_sets:
        fig, axs = plt.subplots(1, len(metrics), figsize=(28, 4))  # Create 1 row and 3 columns of subplots

        for i, metric in enumerate(metrics):
            if metric == "Avg. JCT (hrs)":
                results_data = get_data_from_raw_log(workload_set, "avg_jct")
                df = pd.DataFrame(results_data)
            if metric == "P99 JCT (hrs)":
                results_data = get_data_from_raw_log(workload_set, "p99_jct")
                df = pd.DataFrame(results_data)
            elif metric == "Unfair Fraction":
                real_jct = get_all_jct_from_raw_log(workload_set)
                fair_jct = get_fair_jct_from_raw_log(workload_set)
                ftf = calculate_ftf(real_jct, fair_jct)
                results_data = [
                    {
                        "trace": trace,
                        "algo": algo,
                        "res": sum([i > 1 for i in job.values()]) / len(job.values()),
                    }
                    for trace, algo_data in ftf.items()
                    for algo, job in algo_data.items()
                ]
                df = pd.DataFrame(results_data)
            elif metric == "Worst FTF":
                real_jct = get_all_jct_from_raw_log(workload_set)
                fair_jct = get_fair_jct_from_raw_log(workload_set)
                ftf = calculate_ftf(real_jct, fair_jct)
                results_data = [
                    {
                        "trace": trace,
                        "algo": algo,
                        "res": max(job.values()),
                    }
                    for trace, algo_data in ftf.items()
                    for algo, job in algo_data.items()
                ]
                df = pd.DataFrame(results_data)
            elif metric == "Makespan (hrs)":
                results_data = get_data_from_raw_log(workload_set, "makespan")
                df = pd.DataFrame(results_data)
            elif metric == "Avg. job restarts":
                results_data = get_data_from_raw_log(workload_set, "restarts")
                df = pd.DataFrame(results_data)

            bars = plot_grouped_bar_v3(axs[i], df, metric, colors, fontsize)

        # Create legend for all subplots
        handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
        fig.legend(handles, algorithms, ncol=len(algorithms), loc='upper center',
                   bbox_to_anchor=(0.5, 1.06), fontsize=fontsize, frameon=False)

        # Adjust layout to prevent clipping of labels and legend
        plt.tight_layout(rect=[0, 0, 1, 0.85])
        plt.savefig(f"{save_path}_{workload_set}.jpg")
        plt.savefig(f"{save_path}_{workload_set}.pdf")
        plt.show()
        plt.clf()
        print(f"Finished plot {save_path}_{workload_set}")


def plot_grouped_err_bar_combined(ax, algorithms, testbed_data, simulation_data, colors, ylabel='Avg. JCT (hrs)'):
    fontsize = 26

    testbed_jct_means = [np.mean(i) for i in testbed_data]
    simulation_jct_means = [np.mean(i) for i in simulation_data]
    diff = [abs(i - j) / i for i, j in zip(testbed_jct_means, simulation_jct_means)]
    print(f"Fidelity of simulator – difference between simulator and physical cluster: "
          f"{[f'{i:.2f}' for i in diff]}, "
          f"avg {np.average(diff):.4f}, max {np.max(diff):.4f}, min {np.min(diff):.4f}")

    testbed_jct_errors = [[mean - min(data), max(data) - mean] for mean, data in zip(testbed_jct_means, testbed_data)]
    simulation_jct_errors = [[0, 0] for mean, data in zip(simulation_jct_means, simulation_data)]

    testbed_jct_errors = np.array(testbed_jct_errors).T
    simulation_jct_errors = np.array(simulation_jct_errors).T

    x = np.arange(len(algorithms))
    width = 0.3

    rects1 = ax.bar(x - width / 2 - 0.05, testbed_jct_means, width, label='Real',
                    yerr=testbed_jct_errors, capsize=5, error_kw={'elinewidth': 2},
                    color=colors, hatch='/')
    rects2 = ax.bar(x + width / 2 + 0.05, simulation_jct_means, width, label='Simulated',
                    yerr=simulation_jct_errors, capsize=5, error_kw={'elinewidth': 2},
                    color=colors, hatch='x')

    if ylabel == "Unfair Fraction":
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.1, 0.25))

    if ylabel == "Worst FTF":
        # Use logarithmic scale for y-axis
        ax.set_yscale('log')

        # Limiting the y-axis to the range 0 to 100
        # ax.set_ylim(0, 100)
        yticks = [float(f"{i:.1f}") for i in [1, 10, 100]]
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks, fontsize=fontsize, color='black')

    ax.set_ylabel(ylabel, fontsize=fontsize, color='black')
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.tick_params(axis='y', labelsize=fontsize, colors='black')
    ax.tick_params(axis='x', labelsize=fontsize, colors='black')

    # ax.legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.15), borderaxespad=0,
    #           fontsize=legend_fontsize, frameon=False)

    return rects1, rects2


def fig1_physical_jct_ftf_with_err_bar_v2():
    testbed_path = "/home/cchen/yfliu/ares/bkp/testbed/"
    simulation_path = ("/home/cchen/yfliu/cluster_schedule/pollux/simulator/simulator_logs/"
                       "Simulator-Fidelity/workloads-4h-40j/workload-6/")
    algorithms = ["Ares", "Gavel", "Themis", "AlloX", "Tiresias", "Optimus", "Pollux"]
    algos = [algo_name[i] for i in algorithms]
    colors = ['#e24a33', '#348abd', '#988ed5', '#777777', "#fbc15e", "#8eba41", "#ffb4b8"]
    fontsize = 22

    # 1. Avg JCT
    testbed_jct = [
        [get_avg_jct(testbed_path + f"{algo}/logs/{algo}_{i}.log") for i in range(1, 4)]
        for algo in algos
    ]
    simulation_jct = [
        [get_avg_jct(simulation_path + f"{algo}.txt")] * 3
        for algo in algos
    ]
    avg_data = [np.average(i) for i in testbed_jct]
    improved = [(i - avg_data[0]) / i for i in avg_data]
    print(avg_data)
    print(f"Avg JCT reduced: {improved}")

    # 2. Unfair Fraction
    def compute_unfair(a, b=simulation_path + "ares.txt"):
        job_ftf = calculate_ftf_in_algo(get_all_jct(a), get_fair_jct(b))
        return sum([i > 1 for i in job_ftf.values()]) / len(job_ftf.values())

    testbed_unfair = [
        [compute_unfair(testbed_path + f"{algo}/logs/{algo}_{i}.log") for i in range(1, 4)]
        for algo in algos
    ]
    simulation_unfair = [
        [compute_unfair(simulation_path + f"{algo}.txt")] * 3
        for algo in algos
    ]
    avg_data = [1 - np.average(i) for i in testbed_unfair]
    improved = [avg_data[0] / i for i in avg_data]
    print(avg_data)
    print(f"Fair Fraction reduced: {improved}")

    # 3. Worst FTF
    def compute_worst_ftf(a, b=simulation_path + "ares.txt"):
        job_ftf = calculate_ftf_in_algo(get_all_jct(a), get_fair_jct(b))
        return max(job_ftf.values())

    testbed_worst_ftf = [
        [compute_worst_ftf(testbed_path + f"{algo}/logs/{algo}_{i}.log") for i in range(1, 4)]
        for algo in algos
    ]
    simulation_worst_ftf = [
        [compute_worst_ftf(simulation_path + f"{algo}.txt")] * 3
        for algo in algos
    ]
    avg_data = [np.average(i) for i in testbed_worst_ftf]
    improved = [(i - avg_data[0]) / i for i in avg_data]
    print(avg_data)
    print(f"Worst FTF reduced: {improved}")

    plt.style.use('ggplot')
    fig, axs = plt.subplots(1, 3, figsize=(14, 4), sharey=False)

    plot_grouped_err_bar_combined(axs[0], algorithms, testbed_jct, simulation_jct, colors, ylabel='Avg. JCT (hrs)')
    plot_grouped_err_bar_combined(axs[1], algorithms, testbed_unfair, simulation_unfair, colors, ylabel='Unfair Fraction')
    plot_grouped_err_bar_combined(axs[2], algorithms, testbed_worst_ftf, simulation_worst_ftf, colors, ylabel='Worst FTF')

    handles = ([plt.Rectangle((0, 0), 1, 1,
                              hatch=hatch, facecolor='white', edgecolor='black', lw=1)
                for hatch in ["/", "x"]] +
               [plt.Rectangle((0, 0), 1, 1, color=color)
                for color in colors])
    fig.legend(handles, ["Real", "Simulated"] + algorithms, ncol=5, loc='upper center',
               bbox_to_anchor=(0.5, 1.05), fontsize=fontsize, frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.8])
    save_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "combined_testbed_simulation")
    plt.savefig(save_path + ".jpg")
    plt.savefig(save_path + ".pdf")
    plt.show()


def plot_grouped_err_bar_v3(ax, algorithms, testbed_data, colors, hatches, ylabel='Avg. JCT (hrs)'):
    fontsize = 26
    testbed_jct_means = [np.mean(i) for i in testbed_data]
    testbed_jct_errors = [[mean - min(data), max(data) - mean] for mean, data in zip(testbed_jct_means, testbed_data)]
    testbed_jct_errors = np.array(testbed_jct_errors).T

    x = np.arange(len(algorithms))
    # width = 0.5  # 设置柱子宽度为1以减少空隙

    rects = []
    for i, (mean, err, color, hatch) in enumerate(zip(testbed_jct_means, testbed_jct_errors.T, colors, hatches)):
        rect = ax.bar(x[i], mean,
                      # width,
                      yerr=err.reshape((2, 1)),
                      label=algorithms[i],
                      color=color,
                      # edgecolor=color,
                      capsize=10,
                      error_kw={'elinewidth': 3})
        rects.append(rect)

    if ylabel == "Unfair Fraction":
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.1, 0.25))

    if ylabel == "Worst FTF":
        # Use logarithmic scale for y-axis
        ax.set_yscale('log')

        # Limiting the y-axis to the range 0 to 100
        # ax.set_ylim(0, 100)
        yticks = [float(f"{i:.1f}") for i in [1, 10, 100]]
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks, fontsize=fontsize, color='black')

    ax.set_ylabel(ylabel, fontsize=fontsize, color='black')
    ax.set_xticks([])
    ax.tick_params(axis='y', labelsize=fontsize, colors='black')
    return rects


def fig1_physical_jct_ftf_with_err_bar_v1():
    testbed_path = "/home/cchen/yfliu/ares/bkp/testbed/"
    simulation_path = ("/home/cchen/yfliu/cluster_schedule/pollux/simulator/simulator_logs/"
                       "Simulator-Fidelity/workloads-4h-40j/workload-6/")
    algorithms = ["Ares", "Gavel", "Themis", "AlloX", "Tiresias", "Optimus", "Pollux"]
    algos = [algo_name[i] for i in algorithms]
    colors = ['#e24a33', '#348abd', '#988ed5', '#777777', "#fbc15e", "#8eba41", "#ffb4b8"]
    hatches = ['', '/', '\\', '|', '-', "", ""]  # 定义每个算法的花纹

    # 1. jct
    data_jct = [
        [get_avg_jct(testbed_path + f"{algo}/logs/{algo}_{i}.log") for i in range(1, 3)]
        for algo in algos
    ]
    print(f"data_jct: {data_jct}")

    # 2. unfair
    def compute_unfair(a, b=simulation_path + "ares.txt"):
        job_ftf = calculate_ftf_in_algo(get_all_jct(a), get_fair_jct(b))
        return sum([i > 1 for i in job_ftf.values()]) / len(job_ftf.values())

    data_unfair = [
        [compute_unfair(testbed_path + f"{algo}/logs/{algo}_{i}.log") for i in range(1, 3)]
        for algo in algos
    ]
    print(f"data_unfair: {data_unfair}")

    # 3. worst
    def compute_worst_ftf(a, b=simulation_path + "ares.txt"):
        job_ftf = calculate_ftf_in_algo(get_all_jct(a), get_fair_jct(b))
        return max(job_ftf.values())

    data_worst_ftf = [
        [compute_worst_ftf(testbed_path + f"{algo}/logs/{algo}_{i}.log") for i in range(1, 3)]
        for algo in algos
    ]
    print(f"data_worst_ftf: {data_worst_ftf}")

    # # 3. makespan
    # data_makespan = [
    #     [get_makespan(testbed_path + f"{algo}/logs/{algo}_{i}.log") for i in range(1, 3)]
    #     for algo in algos
    # ]
    # print(f"data_makespan: {data_makespan}")

    plt.style.use('ggplot')
    fig, axs = plt.subplots(1, 3, figsize=(14, 4), sharey=False)

    rects1 = plot_grouped_err_bar_v3(axs[0], algorithms, data_jct, colors, hatches, ylabel='Avg. JCT (hrs)')
    rects2 = plot_grouped_err_bar_v3(axs[1], algorithms, data_unfair, colors, hatches, ylabel='Unfair Fraction')
    rects3 = plot_grouped_err_bar_v3(axs[2], algorithms, data_worst_ftf, colors, hatches, ylabel='Worst FTF')
    # rects3 = plot_grouped_err_bar_v3(axs[3], algorithms, data_makespan, colors, hatches, ylabel='Makespan (hrs)')

    # Create a combined legend
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.5, 1.05),
               loc='upper center', ncol=4, fontsize=22, frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.8])
    save_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "combined_testbed")
    plt.savefig(save_path + ".jpg")
    plt.savefig(save_path + ".pdf")
    plt.show()


def plot_cdf(ax, data, xlabel, fontsize=32, legend_fontsize=19, linewidth=2):
    max_ftf, min_ftf = 0, 1 << 32
    for algorithm, job in data.items():
        sorted_ftf_values = np.sort(list(job.values()))
        max_ftf = max(max_ftf, np.max(sorted_ftf_values))
        min_ftf = min(min_ftf, np.min(sorted_ftf_values))
        # print(max_ftf)
        yvals = np.arange(len(sorted_ftf_values)) / float(len(sorted_ftf_values))
        ax.plot(sorted_ftf_values, yvals, label=algorithm, linewidth=linewidth)

    print(min_ftf, max_ftf)
    ax.set_xlim(min_ftf, max_ftf)  # Set y-axis limits
    ax.set_ylim(0, 1)  # Set y-axis limits

    ax.set_xscale('log')
    # ax.set_xticks(ax.get_xticks())
    print([f"{tick:.2}" for tick in ax.get_xticks()])
    m = {
        '0.001': '0.001',
        '0.01': '0.01',
        '0.1': '0.1',
        '1.0': '1.0',
        '1e+01': '10',
        '1e+02': '100',
        '1e+03': '1000',
    }
    xticklabels = [m[f"{tick:.2}"] for tick in ax.get_xticks()]
    ax.set_xticklabels(xticklabels, fontsize=fontsize, color='black')
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels([f"{tick:.2}" for tick in ax.get_yticks()], fontsize=fontsize, color='black')
    ax.set_xlabel(xlabel, fontsize=fontsize, color='black')
    ax.set_ylabel("Fraction of Jobs", fontsize=fontsize, color='black')


def fig3_ftf_cdf():
    simulation_path = ("/home/cchen/yfliu/cluster_schedule/pollux/simulator/simulator_logs/"
                       "Simulation-16nodes/saturn/workload-2")
    algorithms = ['Ares', 'Gavel', 'Themis', 'AlloX', 'Tiresias', 'Optimus', 'Pollux']

    # First plot data
    jct_data = {
        algo: {
            job_name: jct / 3600
            for job_name, jct in get_all_jct(simulation_path + f"/{algo_name[algo]}.txt").items()
        }
        for algo in algorithms
    }

    # Second plot data
    ftf_data = {
        algo: calculate_ftf_in_algo(
            get_all_jct(simulation_path + f"/{algo_name[algo]}.txt"),
            get_fair_jct(simulation_path + "/ares.txt"),
        )
        for algo in algorithms
    }

    plt.style.use('ggplot')
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))  # Create 1 row and 2 columns of subplots
    plot_cdf(axs[0], jct_data, "Job Completion Time (hrs)")
    plot_cdf(axs[1], ftf_data, "Finish Time Fairness Ratio")

    # Customize the legend for both plots
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.04), fontsize=30, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.8])  # Adjust layout to prevent clipping of labels and legend
    save_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "combined_cdf")
    plt.savefig(save_path + ".pdf")
    plt.savefig(save_path + ".png")
    plt.show()
    print(f'finish plot {save_path}')


def fig4_visualized_schedules():
    simulation_path = ("/home/cchen/yfliu/cluster_schedule/pollux/simulator/simulator_logs/"
                       "Simulation-16nodes/saturn/workload-2")
    # algorithms = ['Ares', 'Optimus', 'Pollux', 'Tiresias']
    algorithms = ['Ares', 'Pollux', 'Gavel', 'Tiresias']
    algo_data = {
        algo: get_scheduling_data(simulation_path + f"/{algo_name[algo]}.txt")
        for algo in algorithms
    }

    num_machines, num_gpus_per_machine = 16, 4
    colors = ['#ffffff', '#a1daea', '#007db8', '#ffaa5b', '#e60018']
    task_labels = ['Free', 'Small', 'Medium', 'Large', '(X)Large']

    fig, axs = plt.subplots(2, 2, figsize=(32, 10))
    axs = axs.flatten()
    fontsize = 36
    title_fontsize = 42
    spine_width = 2

    max_time_steps = max(len(algo_data[algo]) for algo in algorithms)
    for ax, (algo, scheduling_data) in zip(axs, algo_data.items()):
        time_steps = len(scheduling_data)

        for gpu_id in range(num_machines * num_gpus_per_machine):
            machine_id = gpu_id // num_gpus_per_machine
            gpu_index = gpu_id % num_gpus_per_machine
            for time_step in range(time_steps):
                task_type = scheduling_data[time_step, machine_id, gpu_index]
                color = colors[task_type]
                width = 1
                if algo_name[algo] in get_gavel_policies():
                    time_step *= 6
                    width = 6
                ax.add_patch(plt.Rectangle((time_step, gpu_id), width, 1, color=color, alpha=0.7))

        ax.set_xlabel('Round', fontsize=fontsize)
        ax.set_ylabel('GPU ID', fontsize=fontsize)
        ax.set_xticks(range(0, max_time_steps + 1, 200))
        ax.set_yticks(range(0, num_machines * num_gpus_per_machine + 1, num_gpus_per_machine * 8))
        ax.tick_params(axis='x', labelsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)
        ax.set_title(f"{algo}", fontsize=title_fontsize, loc='right')

        ax.set_xlim(0, max_time_steps)
        ax.set_ylim(0, num_machines * num_gpus_per_machine)

        # Set the spine width
        for spine in ax.spines.values():
            spine.set_linewidth(spine_width)

    legend_patches = [Patch(color=color, label=label) for color, label in zip(colors[1:], task_labels[1:])]
    fig.legend(handles=legend_patches,
               loc='upper center',
               bbox_to_anchor=(0.5, 1.025),
               ncol=5,
               fontsize=fontsize,
               frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(os.path.join(os.path.abspath(os.path.dirname(__file__)), f"visualized_schedules") + ".pdf")
    plt.savefig(os.path.join(os.path.abspath(os.path.dirname(__file__)), f"visualized_schedules") + ".png")
    plt.show()
    plt.clf()


def overhead():
    workload_sets = ["scale_1x", "scale_2x", "scale_4x", "scale_8x", "scale_16x", "scale_32x"]
    gpu_size = ["64", "128", "256", "512", "1024", "2048"]
    medians = []
    percentiles_25 = []
    percentiles_75 = []
    for workload_set in workload_sets:
        results_data = get_data_from_raw_log(workload_set, "overhead", trans=False)
        df = pd.DataFrame(results_data)
        df = df.groupby(['workload', 'algo'])['res'].median().unstack()["ares"]
        median_val = df.median()
        percentile_25_val = df.quantile(0.25)
        percentile_75_val = df.quantile(0.75)
        medians.append(median_val)
        percentiles_25.append(percentile_25_val)
        percentiles_75.append(percentile_75_val)

    # 计算误差条
    lower_errors = np.array(medians) - np.array(percentiles_25)
    upper_errors = np.array(percentiles_75) - np.array(medians)
    asymmetric_error = [lower_errors, upper_errors]

    # 绘制带误差条的折线图
    labelsize = 20
    fontsize = 16

    plt.style.use('ggplot')
    plt.figure(figsize=(5, 4))
    plt.errorbar(gpu_size, medians, yerr=asymmetric_error, fmt='-o', capsize=5, capthick=2, elinewidth=2)
    plt.xlabel('Cluster size (#GPUs)', fontsize=labelsize, color='black')
    plt.ylabel('Policy runtime (s)', fontsize=labelsize, color='black')
    plt.xticks(fontsize=fontsize, color='black', rotation=45)
    plt.yticks(fontsize=fontsize, color='black')
    plt.tight_layout()  # 调整布局以防止标签被裁剪
    plt.savefig(os.path.join(os.path.abspath(os.path.dirname(__file__)), f"overhead.pdf"))
    plt.savefig(os.path.join(os.path.abspath(os.path.dirname(__file__)), f"overhead.png"))
    plt.show()


def sensitivity():
    def get_val(_df, algo):
        """
        Helper function to calculate the median, 25th, and 75th percentiles
        for a given algorithm within the dataframe.

        Parameters:
        _df (pd.DataFrame): The dataframe containing the data.
        algo (str): The algorithm name to filter the dataframe.

        Returns:
        tuple: A tuple containing the median, 25th, and 75th percentiles.
        """
        # Group the dataframe by 'workload' and 'algo', calculate the mean of 'res' and unstack the dataframe
        tmp_df = _df.groupby(['workload', 'algo'])['res'].mean().unstack()
        # Filter the dataframe for the given algorithm
        tmp_df = tmp_df[algo]
        # Return the median and the 25th and 75th percentiles
        return tmp_df.median(), tmp_df.quantile(0.25), tmp_df.quantile(0.75)

    # Define the sets of workloads to evaluate
    workload_sets = ["philly", "saturn", "newtrace"]
    workload_names = ["Philly", "Helios", "newTrace"]
    markers = ['o', 's', '^']  # Different markers for each workload set

    # Set the style and figure size for the plot
    plt.style.use('ggplot')
    plt.figure(figsize=(5, 4))
    labelsize = 20
    fontsize = 16

    for workload_set, workload_name, marker in zip(workload_sets, workload_names, markers):
        # Initialize lists to store median and percentile values
        medians = []
        percentiles_25 = []
        percentiles_75 = []
        # Retrieve data from raw log files
        results_data = get_data_from_raw_log(workload_set, "avg_jct", trans=False)
        # Convert the results to a dataframe
        df = pd.DataFrame(results_data)

        # Iterate over the range of scalability thresholds
        for i in np.arange(0.55, 1.0, 0.05):
            # Get the median and percentile values for the current threshold
            median_val, percentile_25_val, percentile_75_val = get_val(df, f"ares-{i:.2f}")
            # Append the values to the respective lists
            medians.append(median_val)
            percentiles_25.append(percentile_25_val)
            percentiles_75.append(percentile_75_val)

        # Convert lists to numpy arrays for easier manipulation
        thresholds = np.arange(0.55, 1.0, 0.05)
        medians = np.array(medians)
        percentiles_25 = np.array(percentiles_25)
        percentiles_75 = np.array(percentiles_75)

        # Plot the line and the shaded region
        plt.plot(thresholds, medians, f'-{marker}', label=workload_name)
        plt.fill_between(thresholds, percentiles_25, percentiles_75, alpha=0.2)

    # Add labels to the plot
    plt.xlabel('Efficiency Threshold', fontsize=labelsize, color='black')
    plt.ylabel('Average JCT (hrs)', fontsize=labelsize, color='black')
    plt.xticks(fontsize=fontsize, color='black', rotation=45)
    plt.yticks(fontsize=fontsize, color='black')
    plt.legend(fontsize=fontsize)  # Add legend for the different workload sets
    plt.tight_layout()  # Adjust layout to prevent label cutoff
    plt.savefig(os.path.join(os.path.abspath(os.path.dirname(__file__)), f"sensitivity.pdf"))
    plt.savefig(os.path.join(os.path.abspath(os.path.dirname(__file__)), f"sensitivity.png"))
    plt.show()


if __name__ == '__main__':
    # nohup python3 collect_result.py > collect_result.log 2>&1 &
    os.chdir(f"/home/cchen/yfliu/cluster_schedule/pollux/simulator/simulator_logs/Simulation-16nodes")

    # workload_sets = ["workloads-0.5", "workloads-1.0", "workloads-1.5", "workloads-2.0", "workloads-realistic"]
    fig2_sim_all_jct_and_ftf_v2()
    # fig2_sim_all_jct_and_ftf_v3()

    # fig1_physical_jct_ftf_with_err_bar_v1()
    fig1_physical_jct_ftf_with_err_bar_v2()

    fig3_ftf_cdf()

    fig4_visualized_schedules()

    os.chdir(f"/home/cchen/yfliu/cluster_schedule/pollux/simulator/simulator_logs/Simulation-scale")
    overhead()

    os.chdir(f"/home/cchen/yfliu/cluster_schedule/pollux/simulator/simulator_logs/Simulation-sensitivity")
    sensitivity()
