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
    print_improve_reduce


def plot_grouped_bar_v2(src, wl_set, metric, save_path):
    fontsize = 32
    legend_fontsize = 21
    left, right, top, bottom = 0.17, 1, 1, 0.14
    linewidth = 2
    markersize = 10

    # Set the style of the plot
    plt.style.use('ggplot')

    # Grouping the data by 'workload' and 'algo' and calculating mean JCT
    grouped_df = src.groupby(['workload', 'algo'])['res'].mean().unstack()

    # Plotting the bar chart
    ax = grouped_df.plot(kind='bar', figsize=(11, 6))

    # Adding labels and title
    # plt.title(f'{metric} for Each Algorithm Under Each Workload of {wl_set}')
    plt.xlabel('Trace ID', fontsize=fontsize, color='black')
    plt.ylabel(f'{metric}', fontsize=fontsize, color='black')
    plt.xticks(rotation=0, fontsize=fontsize, color='black')
    plt.yticks(fontsize=fontsize, color='black')

    # Set the yticks
    yticks = ax.get_yticks()
    stride = len(yticks) // 7 + 1
    yticks = [float(f"{i:.1f}") for i in yticks[::stride]]
    ax.set_yticks(yticks)  # Show every 2nd y-tick for example
    ax.set_yticklabels(yticks, fontsize=fontsize, color='black')

    # Customize the legend
    plt.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.1),
               borderaxespad=0, fontsize=legend_fontsize, frameon=False)
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.savefig(f"{save_path}.jpg")
    plt.savefig(f"{save_path}.pdf")
    plt.show()
    # plt.clf()
    print(f"finish plot {save_path}")


def fig2_sim_all_jct_and_ftf(workload_set):
    # 1. avg jct
    results_data = get_data_from_raw_log(workload_set, "avg_jct")
    df = pd.DataFrame(results_data)
    all_data, improve, reduce = print_improve_reduce(df, [])
    print(f"min_val: {reduce.min().min()}, max_val: {reduce.max().max()}")
    plot_grouped_bar_v2(df, workload_set, "Average JCT(hrs)",
                        os.path.join(os.path.abspath(os.path.dirname(__file__)), f"{workload_set}-avg_jct"))

    # 2. finish time fairness
    real_jct = get_all_jct_from_raw_log(workload_set)
    fair_jct = get_fair_jct_from_raw_log(workload_set)
    ftf = calculate_ftf(real_jct, fair_jct)

    results_data = [
        {
            "workload": workload,
            "algo": algo,
            "res": sum(job.values()) / len(job),
        }
        for workload, algo_data in ftf.items()
        for algo, job in algo_data.items()
    ]
    df = pd.DataFrame(results_data)
    all_data, improve, reduce = print_improve_reduce(df, [])
    print(f"min_val: {improve.min().min()}, max_val: {improve.max().max()}")
    plot_grouped_bar_v2(df, workload_set, "Average FTF",
                        os.path.join(os.path.abspath(os.path.dirname(__file__)), f"{workload_set}-avg_ftf"))


def plot_grouped_err_bar_v2(algorithms, testbed_data, simulation_data, save_path, ylabel='Average JCT (hrs)'):
    fontsize = 32
    legend_fontsize = 30
    plt.style.use('default')

    testbed_jct_means = [np.mean(i) for i in testbed_data]
    simulation_jct_means = [np.mean(i) for i in simulation_data]
    diff = [abs(i - j) / i for i, j in zip(testbed_jct_means, simulation_jct_means)]
    print(f"Fidelity of simulator – difference between simulator and physical cluster: "
          f"{np.mean(diff)}")
    # 计算误差条范围（最大值和最小值之间的范围）
    testbed_jct_errors = [[mean - min(data), max(data) - mean]
                          for mean, data in zip(testbed_jct_means, testbed_data)]
    simulation_jct_errors = [[0, 0] for mean, data in zip(simulation_jct_means, simulation_data)]
    # 转换为 numpy 数组
    testbed_jct_errors = np.array(testbed_jct_errors).T
    simulation_jct_errors = np.array(simulation_jct_errors).T
    # 设置柱状图的位置
    x = np.arange(len(algorithms))  # x 轴位置
    width = 0.35  # 柱子宽度

    fig, ax = plt.subplots(figsize=(9, 6))
    rects1 = ax.bar(x - width / 2, testbed_jct_means, width, label='Real',
                    yerr=testbed_jct_errors, capsize=10, error_kw={'elinewidth': 2})
    rects2 = ax.bar(x + width / 2, simulation_jct_means, width, label='Simulated',
                    yerr=simulation_jct_errors, capsize=5, error_kw={'elinewidth': 1})

    # 添加一些文本标签
    ax.set_xlabel('Algorithms', fontsize=fontsize, color='black')
    ax.set_ylabel(ylabel, fontsize=fontsize, color='black')
    # ax.set_title('Average JCT Comparison Between Testbed and Simulation')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)
    plt.yticks(fontsize=fontsize, color='black')
    plt.xticks(fontsize=fontsize, color='black')
    # ax.legend(fontsize=legend_fontsize, frameon=False)
    ax.legend(
        ncol=2,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.15),
        borderaxespad=0,
        fontsize=legend_fontsize,
        frameon=False
    )

    fig.tight_layout()

    # 保存并显示图像
    plt.savefig(save_path + ".jpg")
    plt.savefig(save_path + ".pdf")
    plt.show()


def fig1_physical_jct_ftf_with_err_bar():
    testbed_path = "/home/cchen/yfliu/ares/bkp/14h/"
    simulation_path = ("/home/cchen/yfliu/cluster_schedule/pollux/simulator/simulator_logs/"
                       "SimulatorFidelity/workloads-4h-40j/workload-6/")
    algorithms = ['Ares', 'Optimus', 'Pollux', 'Tiresias']

    # 1. JCT Comparison
    ares_testbed_jct = [get_avg_jct(testbed_path + f"ares/logs/ares_{i}.log") for i in range(1, 4)]
    ares_simulation_jct = [get_avg_jct(simulation_path + "ares.txt")] * 3
    optimus_testbed_jct = [get_avg_jct(testbed_path + f"optimus/logs/optimus_{i}.log") for i in range(1, 4)]
    optimus_simulation_jct = [get_avg_jct(simulation_path + "optimus.txt")] * 3
    pollux_testbed_jct = [get_avg_jct(testbed_path + f"pollux/logs/pollux_{i}.log") for i in range(1, 4)]
    pollux_simulation_jct = [get_avg_jct(simulation_path + "pollux.txt")] * 3
    tiresias_testbed_jct = [get_avg_jct(testbed_path + f"tiresias/logs/tiresias_{i}.log") for i in range(1, 4)]
    tiresias_simulation_jct = [get_avg_jct(simulation_path + "tiresias.txt")] * 3
    testbed_data = [ares_testbed_jct, optimus_testbed_jct, pollux_testbed_jct, tiresias_testbed_jct]
    simulation_data = [ares_simulation_jct, optimus_simulation_jct, pollux_simulation_jct, tiresias_simulation_jct]
    print(testbed_data)
    print(simulation_data)
    avg_data = [np.average(i) for i in testbed_data]
    improvement = [(i - avg_data[0]) / i for i in avg_data]
    print(avg_data)
    print(improvement)
    # input()

    plot_grouped_err_bar_v2(algorithms, testbed_data, simulation_data,
                            os.path.join(os.path.abspath(os.path.dirname(__file__)), f"jct_comparison"))

    # 2. FTF Comparison
    def compute_avg_ftf(a, b):
        job_ftf = calculate_ftf_in_algo(
            get_all_jct(a),
            get_fair_jct(b)
        )
        return sum(job_ftf.values()) / len(job_ftf)

    ares_testbed_ftf = [compute_avg_ftf(testbed_path + f"ares/logs/ares_{i}.log", simulation_path + "ares.txt")
                        for i in range(1, 4)]
    ares_simulation_ftf = [compute_avg_ftf(simulation_path + "ares.txt", simulation_path + "ares.txt")] * 3
    optimus_testbed_ftf = [compute_avg_ftf(testbed_path + f"optimus/logs/optimus_{i}.log", simulation_path + "ares.txt")
                           for i in range(1, 4)]
    optimus_simulation_ftf = [compute_avg_ftf(simulation_path + "optimus.txt", simulation_path + "ares.txt")] * 3
    pollux_testbed_ftf = [compute_avg_ftf(testbed_path + f"pollux/logs/pollux_{i}.log", simulation_path + "ares.txt")
                          for i in range(1, 4)]
    pollux_simulation_ftf = [compute_avg_ftf(simulation_path + "pollux.txt", simulation_path + "ares.txt")] * 3
    tiresias_testbed_ftf = [
        compute_avg_ftf(testbed_path + f"tiresias/logs/tiresias_{i}.log", simulation_path + "ares.txt")
        for i in range(1, 4)]
    tiresias_simulation_ftf = [compute_avg_ftf(simulation_path + "tiresias.txt", simulation_path + "ares.txt")] * 3
    testbed_data = [ares_testbed_ftf, optimus_testbed_ftf, pollux_testbed_ftf, tiresias_testbed_ftf]
    simulation_data = [ares_simulation_ftf, optimus_simulation_ftf, pollux_simulation_ftf, tiresias_simulation_ftf]
    print(testbed_data)
    print(simulation_data)

    plot_grouped_err_bar_v2(algorithms, testbed_data, simulation_data,
                            os.path.join(os.path.abspath(os.path.dirname(__file__)), f"ftf_comparison"))

    # 3. Makespan Comparison
    ares_testbed_makespan = [get_makespan(testbed_path + f"ares/logs/ares_{i}.log") for i in range(1, 4)]
    ares_simulation_makespan = [get_makespan(simulation_path + "ares.txt")] * 3
    optimus_testbed_makespan = [get_makespan(testbed_path + f"optimus/logs/optimus_{i}.log") for i in range(1, 4)]
    optimus_simulation_makespan = [get_makespan(simulation_path + "optimus.txt")] * 3
    pollux_testbed_makespan = [get_makespan(testbed_path + f"pollux/logs/pollux_{i}.log") for i in range(1, 4)]
    pollux_simulation_makespan = [get_makespan(simulation_path + "pollux.txt")] * 3
    tiresias_testbed_makespan = [get_makespan(testbed_path + f"tiresias/logs/tiresias_{i}.log") for i in range(1, 4)]
    tiresias_simulation_makespan = [get_makespan(simulation_path + "tiresias.txt")] * 3
    testbed_data = [ares_testbed_makespan, optimus_testbed_makespan, pollux_testbed_makespan,
                    tiresias_testbed_makespan]
    simulation_data = [ares_simulation_makespan, optimus_simulation_makespan, pollux_simulation_makespan,
                       tiresias_simulation_makespan]
    print(testbed_data)
    print(simulation_data)
    avg_data = [np.average(i) for i in testbed_data]
    improvement = [(i - avg_data[0]) / i for i in avg_data]
    print(avg_data)
    print(improvement)

    plot_grouped_err_bar_v2(algorithms, testbed_data, simulation_data,
                            os.path.join(os.path.abspath(os.path.dirname(__file__)), f"makespan_comparison"),
                            ylabel='Makespan (hrs)')


def plot_cdf(data, xlabel, save_path):
    # >>> cdf plotting
    fontsize = 21
    legend_fontsize = 19
    linewidth = 2

    plt.figure(figsize=(10, 4))
    plt.style.use('ggplot')
    # print(wl_set)
    for algorithm, job in data.items():
        # print(f"workload-{workload}, {algorithm}: \t"
        #       f"avg {sum(job.values()) / len(job):.2f}, max {max(job.values()):.2f}, min {min(job.values()):.2f}")
        sorted_ftf_values = np.sort(list(job.values()))
        print(sorted_ftf_values)
        yvals = np.arange(len(sorted_ftf_values)) / float(len(sorted_ftf_values))
        plt.plot(sorted_ftf_values, yvals, label=algorithm, linewidth=linewidth)

    # plt.title(f"CDF of FTF for {wl_set}/workload-{workload}")
    plt.xticks(fontsize=fontsize, color='black')
    plt.yticks(fontsize=fontsize, color='black')
    plt.xlabel(xlabel, fontsize=fontsize, color='black')
    plt.ylabel("Fraction of Jobs", fontsize=fontsize, color='black')
    plt.legend(
        ncol=4,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.15),
        borderaxespad=0,
        fontsize=legend_fontsize,
        frameon=False
    )
    plt.xscale('log')
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.savefig(save_path + ".pdf")
    plt.savefig(save_path + ".png")
    plt.show()
    plt.clf()
    print(f'finish plot {save_path}')


def fig3_ftf_cdf():
    simulation_path = ("/home/cchen/yfliu/cluster_schedule/pollux/simulator/simulator_logs/"
                       "Simulation-16nodes/saturn/workload-2")
    algorithms = ['Ares', 'Optimus', 'Pollux', 'Tiresias']
    ftf_data = {
        algo: calculate_ftf_in_algo(
            get_all_jct(simulation_path + f"/{algo.lower()}.txt"),
            get_fair_jct(simulation_path + "/ares.txt"),
        )
        for algo in algorithms
    }
    print(ftf_data)
    plot_cdf(ftf_data, "Finish Time Fairness Ratio",
             os.path.join(os.path.abspath(os.path.dirname(__file__)), f"ftf_cdf"))

    ftf_data = {
        algo: {
            job_name: jct / 3600
            for job_name, jct in get_all_jct(simulation_path + f"/{algo.lower()}.txt").items()
        }
        for algo in algorithms
    }
    print(ftf_data)
    plot_cdf(ftf_data, "Job Completion Time (hrs)",
             os.path.join(os.path.abspath(os.path.dirname(__file__)), f"jct_cdf"))


def fig4_visualized_schedules():
    simulation_path = ("/home/cchen/yfliu/cluster_schedule/pollux/simulator/simulator_logs/"
                       "Simulation-16nodes/saturn/workload-2")
    algorithms = ['Ares', 'Optimus', 'Pollux', 'Tiresias']
    algo_data = {
        algo: get_scheduling_data(simulation_path + f"/{algo.lower()}.txt")
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
                ax.add_patch(plt.Rectangle((time_step, gpu_id), 1, 1, color=color, alpha=0.7))

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
        results_data = get_data_from_raw_log(workload_set, "overhead")
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
    fontsize = 16
    legend_fontsize = 19
    linewidth = 2
    markersize = 10

    plt.style.use('ggplot')
    plt.figure(figsize=(8, 3))
    plt.errorbar(gpu_size, medians, yerr=asymmetric_error, fmt='-o', capsize=5, capthick=2, elinewidth=2)
    plt.xlabel('Cluster size (#GPUs)', fontsize=fontsize, color='black')
    plt.ylabel('Policy runtime (s)', fontsize=fontsize, color='black')
    plt.xticks(fontsize=fontsize, color='black')
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
        print(tmp_df)
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
    plt.figure(figsize=(8, 3))
    fontsize = 16

    for workload_set, workload_name, marker in zip(workload_sets, workload_names, markers):
        # Initialize lists to store median and percentile values
        medians = []
        percentiles_25 = []
        percentiles_75 = []
        # Retrieve data from raw log files
        results_data = get_data_from_raw_log(workload_set, "avg_jct")
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

        # Calculate the lower and upper errors for the error bars
        lower_errors = np.array(medians) - np.array(percentiles_25)
        upper_errors = np.array(percentiles_75) - np.array(medians)
        asymmetric_error = [lower_errors, upper_errors]

        # Plot the line with error bars
        plt.errorbar(np.arange(0.55, 1.0, 0.05), medians, yerr=asymmetric_error, fmt=f'-{marker}', capsize=5,
                     capthick=2, elinewidth=2, label=workload_name)

    # Add labels to the plot
    plt.xlabel('scalability control threshold', fontsize=fontsize, color='black')
    plt.ylabel('Average JCT (hrs)', fontsize=fontsize, color='black')
    plt.xticks(fontsize=fontsize, color='black')
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
    workload_sets = ["philly", "saturn", "newtrace"]
    # for workload_set in workload_sets:
    #     fig2_sim_all_jct_and_ftf(workload_set)
    # fig1_physical_jct_ftf_with_err_bar()
    # fig3_ftf_cdf()
    # fig4_visualized_schedules()

    # os.chdir(f"/home/cchen/yfliu/cluster_schedule/pollux/simulator/simulator_logs/Simulation-scale")
    # overhead()

    os.chdir(f"/home/cchen/yfliu/cluster_schedule/pollux/simulator/simulator_logs/Simulation-sensitivity")
    sensitivity()
