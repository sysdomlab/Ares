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
    calculate_ftf, get_jct


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


def sim_all_jct_and_ftf(workload_set):
    # 1. avg jct
    results_data = get_data_from_raw_log(workload_set, "avg_jct")
    df = pd.DataFrame(results_data)
    print(df)
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
    print(df)
    plot_grouped_bar_v2(df, workload_set, "Average FTF",
                        os.path.join(os.path.abspath(os.path.dirname(__file__)), f"{workload_set}-avg_ftf"))


def plot_grouped_err_bar_v2(algorithms, testbed_data, simulation_data, save_path):
    fontsize = 32
    legend_fontsize = 20
    plt.style.use('default')

    testbed_jct_means = [np.mean(i) for i in testbed_data]
    simulation_jct_means = [np.mean(i) for i in simulation_data]
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

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width / 2, testbed_jct_means, width, label='Real',
                    yerr=testbed_jct_errors, capsize=10, error_kw={'elinewidth': 2})
    rects2 = ax.bar(x + width / 2, simulation_jct_means, width, label='Simulated',
                    yerr=simulation_jct_errors, capsize=5, error_kw={'elinewidth': 1})

    # 添加一些文本标签
    ax.set_xlabel('Algorithms', fontsize=fontsize, color='black')
    ax.set_ylabel('Average JCT (h)', fontsize=fontsize, color='black')
    # ax.set_title('Average JCT Comparison Between Testbed and Simulation')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)
    plt.yticks(fontsize=fontsize, color='black')
    plt.xticks(fontsize=fontsize, color='black')
    ax.legend(fontsize=legend_fontsize, frameon=False)

    fig.tight_layout()

    # 保存并显示图像
    plt.savefig(save_path + ".jpg")
    plt.savefig(save_path + ".pdf")
    plt.show()


def physical_jct_ftf_with_err_bar():
    testbed_path = "/home/cchen/yfliu/ares/bkp/14h/"
    simulation_path = ("/home/cchen/yfliu/cluster_schedule/pollux/simulator/simulator_logs/"
                       "SimulatorFidelity/workloads-4h-40j/workload-6/")
    ares_testbed_jct = sorted([get_jct(testbed_path + f"ares/logs/ares_{i}.log") for i in range(1, 4)])
    ares_simulation_jct = [get_jct(simulation_path + "ares.txt")] * 3
    optimus_testbed_jct = sorted([get_jct(testbed_path + f"optimus/logs/optimus_{i}.log") for i in range(1, 4)])
    optimus_simulation_jct = [get_jct(simulation_path + "optimus.txt")] * 3
    pollux_testbed_jct = sorted([get_jct(testbed_path + f"pollux/logs/pollux_{i}.log") for i in range(1, 4)])
    pollux_simulation_jct = [get_jct(simulation_path + "pollux.txt")] * 3
    tiresias_testbed_jct = sorted([get_jct(testbed_path + f"tiresias/logs/tiresias_{i}.log") for i in range(1, 4)])
    tiresias_simulation_jct = [get_jct(simulation_path + "tiresias.txt")] * 3
    testbed_jct_data = [ares_testbed_jct, optimus_testbed_jct, pollux_testbed_jct, tiresias_testbed_jct]
    simulation_jct_data = [ares_simulation_jct, optimus_simulation_jct, pollux_simulation_jct, tiresias_simulation_jct]

    algorithms = ['Ares', 'Optimus', 'Pollux', 'Tiresias']
    plot_grouped_err_bar_v2(algorithms, testbed_jct_data, simulation_jct_data,
                            os.path.join(os.path.abspath(os.path.dirname(__file__)), f"jct_comparison"))

    # ares_testbed_ftf =
    # ares_simulation_ftf =
    # optimus_testbed_ftf =
    # optimus_simulation_ftf =
    # pollux_testbed_ftf =
    # pollux_simulation_ftf =
    # tiresias_testbed_ftf =
    # tiresias_simulation_ftf =


if __name__ == '__main__':
    # nohup python3 collect_result.py > collect_result.log 2>&1 &
    os.chdir(f"/home/cchen/yfliu/cluster_schedule/pollux/simulator/simulator_logs/Simulation-16nodes")

    # workload_sets = ["workloads-0.5", "workloads-1.0", "workloads-1.5", "workloads-2.0", "workloads-realistic"]
    workload_sets = ["philly", "saturn", "newtrace"]
    # workload_sets = ["philly", "saturn"]
    for workload_set in workload_sets:
        sim_all_jct_and_ftf(workload_set)
        pass
    physical_jct_ftf_with_err_bar()
