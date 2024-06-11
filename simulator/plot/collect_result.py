import collections
import os
import subprocess
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict

from matplotlib.patches import Patch


# 遍历日志文件
def get_all_files_in_directory(directory, exclude_substr: list = None):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    if exclude_substr is not None:
        file_paths = [i for i in file_paths if all(j not in i for j in exclude_substr)]
        file_paths = [i for i in file_paths if "txt" in i]
    return file_paths


def get_avg_jct(log_file):
    result = subprocess.run(
        ["tail", "-n", "1", log_file],
        capture_output=True,
        text=True
    )
    res = float(result.stdout.strip().split()[-1]) / 60 / 60
    return res


def get_p99_jct(log_file):
    result = subprocess.run(
        ["tail", "-n", "2", log_file],
        capture_output=True,
        text=True
    )
    res_dict: dict = eval(result.stdout.split("\n")[0])
    jct_values = [i for i in res_dict.values()]
    res = np.percentile(jct_values, 99) / 60 / 60
    return res


def get_makespan(log_file):
    res = 0
    with open(log_file, 'r') as f:
        for line in f:
            if "SIMULATOR TIME" in line and "GPSSystem" not in line:
                res = float(line.split(":")[-1].strip().split(" ")[0]) / 60 / 60
    return res


def get_data_from_raw_log(wl_set, metric):
    results_data = []
    for file in sorted(get_all_files_in_directory(wl_set, ["fifo", "jpg"])):
        # 从文件名中提取工作负载和算法
        workload = file.split("/")[-2].split("-")[-1]
        algo = file.split("/")[-1].split(".")[0]
        # print(f"Workload: {workload}, Algorithm: {algo}")

        if metric == "avg_jct":
            res = get_avg_jct(file)
        elif metric == "p99_jct":
            res = get_p99_jct(file)
        elif metric == "makespan":
            res = get_makespan(file)
        else:
            raise ValueError(f"Invalid metric: {metric}")

        # 将结果数据添加到列表中
        results_data.append({
            "workload": workload,
            "algo": algo,
            "res": res
        })
    return results_data


def get_markdown_table(src):
    # 使用 pivot_table 函数将数据框重塑为行索引为工作负载，列索引为算法的表格
    pivot_table = src.pivot_table(index="workload", columns="algo", values="res")
    markdown_table = pivot_table.to_markdown()  # 将 pivot_table 转换为 markdown 格式的表格
    return markdown_table


def plot_grouped_bar(src, wl_set, metric):
    # Set the style of the plot
    plt.style.use('ggplot')

    # Grouping the data by 'workload' and 'algo' and calculating mean JCT
    grouped_df = src.groupby(['workload', 'algo'])['res'].mean().unstack()

    # Plotting the bar chart
    grouped_df.plot(kind='bar', figsize=(10, 6))

    # Adding labels and title
    plt.title(f'{metric} for Each Algorithm Under Each Workload of {wl_set}')
    plt.xlabel('Workload')
    plt.ylabel(f'{metric}(h)')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

    # Showing the plot
    plt.legend(title='Algorithm', bbox_to_anchor=(1, 1))
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.savefig(f"{wl_set}/{metric}.jpg")
    # plt.show()
    plt.clf()
    print(f"finish plot {wl_set}/{metric}.jpg")


def get_all_jct(log_file):
    result = subprocess.run(
        ["tail", "-n", "2", log_file],
        capture_output=True,
        text=True
    )
    res: dict = eval(result.stdout.split("\n")[0])
    return res


def get_fair_jct(log_file):
    result = "{}"
    with open(log_file, 'r') as f:
        for line in f:
            if "[GPSSystem]" in line and "Average JCT" not in line and "SIMULATOR TIME" not in line:
                result = line.replace("[GPSSystem]", "")
    res: dict = eval(result)
    return res


def get_all_jct_from_raw_log(wl_set):
    results_data = defaultdict(dict)
    for file in sorted(get_all_files_in_directory(wl_set, ["fifo", "jpg"])):
        workload = file.split("/")[-2].split("-")[-1]
        algo = file.split("/")[-1].split(".")[0]
        # print(f"Workload: {workload}, Algorithm: {algo}")

        res_dict: dict = get_all_jct(file)
        # print(res_dict)

        results_data[workload][algo] = res_dict
    return results_data


def get_fair_jct_from_raw_log(wl_set):
    results_data = {}
    for file in sorted(get_all_files_in_directory(wl_set, ["fifo", "jpg"])):
        if "ares" not in file:
            continue
        workload = file.split("/")[-2].split("-")[-1]
        # print(workload, file)

        res_dict: dict = get_fair_jct(file)
        # print(res_dict)

        results_data[workload] = res_dict
    return results_data


def calculate_ftf_in_algo(jct_data, fair_jct_data):
    return {
        job_name: jct / fair_jct_data[job_name]
        if len(job_name.split("-")) == 2
        else jct / fair_jct_data["-".join(job_name.split("-")[:-1])]
        for job_name, jct in jct_data.items()
    }


def calculate_ftf_in_workload(jct_data, fair_jct_data):
    # res = defaultdict(dict)
    # for algo, jct_values in jct_data.items():
    #     res[algo] = {}
    #     for job_name, jct in jct_values.items():
    #         fair_jct = fair_jct_data.get(job_name, None)
    #         res[algo][job_name] = jct / fair_jct
    return {
        algo: calculate_ftf_in_algo(jct_values, fair_jct_data)
        for algo, jct_values in jct_data.items()
    }


def calculate_ftf(jct_data, fair_jct_data):
    return {
        workload: calculate_ftf_in_workload(jct_data[workload], fair_jct_data[workload])
        for workload in jct_data.keys()
    }


def plot_cdf(data, wl_set):
    # Set the style of the plot
    for workload, v in data.items():
        plt.style.use('ggplot')
        # print(wl_set)
        for algorithm, job in v.items():
            # print(f"workload-{workload}, {algorithm}: \t"
            #       f"avg {sum(job.values()) / len(job):.2f}, max {max(job.values()):.2f}, min {min(job.values()):.2f}")
            sorted_ftf_values = np.sort(list(job.values()))
            yvals = np.arange(len(sorted_ftf_values)) / float(len(sorted_ftf_values))
            plt.plot(sorted_ftf_values, yvals, label=algorithm)

        plt.title(f"CDF of FTF for {wl_set}/workload-{workload}")
        plt.xlabel("Finish Time Fairness (FTF)")
        plt.ylabel("CDF")
        plt.legend()
        plt.grid(True)
        plt.xscale('log')
        plt.tight_layout()  # Adjust layout to prevent clipping of labels
        plt.savefig(f"{wl_set}/ftf_cdf_workload-{workload}.jpg")
        # plt.show()
        plt.clf()
        print(f"finish plot {wl_set}/ftf_cdf_workload-{workload}.jpg")


def plot_grouped_bar_with_error_bars(src, wl_set, metric):
    # Set the style of the plot
    plt.style.use('ggplot')

    # Grouping the data by 'workload' and 'algo' and calculating mean JCT
    grouped_df = src.groupby(['workload', 'algo'])['res'].mean().unstack()

    # Plotting the bar chart with error bars
    fig, ax = plt.subplots(figsize=(10, 6))
    # print(errors.shape)  # (8, 5, 2)
    # grouped_df.plot(kind='bar', ax=ax, yerr=errors, capsize=5)  # (5, 2, 8)
    grouped_df.plot(kind='bar', ax=ax, capsize=5)  # (5, 2, 8)

    # Adding labels and title
    plt.title(f'{metric} for Each Algorithm Under Each Workload of {wl_set}')
    plt.xlabel('Workload')
    plt.ylabel(f'{metric}')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

    # Showing the plot
    plt.legend(title='Algorithm', bbox_to_anchor=(1, 1))
    plt.yscale('log')
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.savefig(f"{wl_set}/{metric}.jpg")
    # plt.show()
    plt.clf()
    print(f"finish plot {wl_set}/{metric}.jpg")


def get_scheduling_data(log_file):
    with open(log_file, 'r') as f:
        log_data = f.read()
        matches = re.findall(pattern=r'allocations:\s\{(.+?)\}', string=log_data, flags=re.MULTILINE)

    task_flag = {
        "cifar10": 1,
        "ncf": 1,
        "deepspeech2": 2,
        "bert": 2,
        "yolov3": 3,
        "imagenet": 4,
    }
    num_machines, num_gpus_per_machine = 16, 4
    data = np.zeros((len(matches), num_machines, num_gpus_per_machine), dtype=int)
    for i, match in enumerate(matches):
        match = eval("{" + match + "}")
        # print(match)
        available_gpus = {i: 0 for i in range(16)}
        # print(available_gpus)
        for task, gpu_config in match.items():
            for gpu_id in gpu_config:
                machine_id = int(gpu_id[0])
                gpu_index = available_gpus[machine_id]
                available_gpus[machine_id] += 1
                # print(machine_id, gpu_index)
                # print(match)
                data[i, machine_id, gpu_index] = task_flag[task.split("-")[0]]
    return data


def get_scheduling_from_raw_log(wl_set):
    results_data = collections.defaultdict(dict)
    for file in sorted(get_all_files_in_directory(wl_set, ["fifo", "jpg"])):
        workload = file.split("/")[-2].split("-")[-1]
        algo = file.split("/")[-1].split(".")[0]
        # print(workload, file)

        res = get_scheduling_data(file)

        results_data[workload][algo] = res
    return results_data


def plot_scheduling(res_data, wl_set, wl_set_filter=None, wl_filter=None, algo_filter=None):
    if wl_set_filter is not None and wl_set not in wl_set_filter:
        return

    num_machines, num_gpus_per_machine = 16, 4

    plt.style.use('ggplot')
    for workload, algo_data in res_data.items():
        if wl_filter is not None and workload not in wl_filter:
            continue
        for algo, scheduling_data in algo_data.items():
            if algo_filter is not None and algo not in algo_filter:
                continue
            colors = ['#ffffff', '#a1daea', '#007db8', '#ffaa5b', '#e60018']
            task_labels = ['Free', 'Small', 'Medium', 'Large', '(X)Large']
            time_steps = len(scheduling_data)  # (time_steps, num_machines, num_gpus_per_machine)

            # 创建图形和坐标轴
            plt.figure(figsize=(32, 9))
            ax = plt.gca()

            # 绘制任务资源占用分布图
            for gpu_id in range(num_machines * num_gpus_per_machine):
                machine_id = gpu_id // num_gpus_per_machine
                gpu_index = gpu_id % num_gpus_per_machine
                for time_step in range(time_steps):
                    task_type = scheduling_data[time_step, machine_id, gpu_index]
                    color = colors[task_type]
                    ax.add_patch(plt.Rectangle((time_step, gpu_id), 1, 1, color=color, alpha=0.7))

            # 设置坐标轴标签和标题
            fontsize = 50
            plt.xlabel('Round', fontsize=fontsize)
            plt.ylabel('GPU ID', fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.title(f"Scheduling Decision for {algo} on {wl_set}/workload-{workload}", fontsize=fontsize)

            # 显示颜色标注
            legend_patches = [Patch(color=color, label=label) for color, label in zip(colors[1:], task_labels[1:])]
            ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1, 1), fontsize=fontsize)

            # 设置坐标轴范围
            plt.xlim(0, time_steps)
            plt.ylim(0, num_machines * num_gpus_per_machine)

            # 显示图形
            plt.tight_layout()
            plt.savefig(f"{wl_set}/visualized_schedules_{workload}-{algo}.jpg")
            # plt.show()
            plt.clf()
            plt.close()
            print(f"finish plot {wl_set}/visualized_schedules_{workload}-{algo}.jpg")


def main(workload_set):
    # 1. avg jct
    results_data = get_data_from_raw_log(workload_set, "avg_jct")
    df = pd.DataFrame(results_data)
    print(f">>> avg jct for {workload_set}:\n {df}")
    # print(get_markdown_table(df))
    plot_grouped_bar(df, workload_set, "Average JCT")

    # 2. p99 jct
    results_data = get_data_from_raw_log(workload_set, "p99_jct")
    df = pd.DataFrame(results_data)
    print(f">>> p99 jct for {workload_set}:\n {df}")
    plot_grouped_bar(df, workload_set, "P99 JCT")

    # 3. Makespan
    results_data = get_data_from_raw_log(workload_set, "makespan")
    df = pd.DataFrame(results_data)
    print(f">>> makespan for {workload_set}:\n {df}")
    plot_grouped_bar(df, workload_set, "Makespan")

    # 4. finish time fairness
    #   4.1 FTF CDF
    real_jct = get_all_jct_from_raw_log(workload_set)
    fair_jct = get_fair_jct_from_raw_log(workload_set)
    ftf = calculate_ftf(real_jct, fair_jct)
    plot_cdf(ftf, workload_set)

    #   4.2 avg FTF
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
    print(f">>> avg ftf for {workload_set}:\n {df}")
    plot_grouped_bar(df, workload_set, "Average FTF")

    #   4.3 worst FTF
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
    print(f">>> worst ftf for {workload_set}:\n {df}")
    plot_grouped_bar(df, workload_set, "Worst FTF")

    #   4.3 p99 FTF
    results_data = [
        {
            "workload": workload,
            "algo": algo,
            "res": np.percentile([i for i in job.values()], 99)
        }
        for workload, algo_data in ftf.items()
        for algo, job in algo_data.items()
    ]
    df = pd.DataFrame(results_data)
    print(f">>> p99 ftf for {workload_set}:\n {df}")
    plot_grouped_bar(df, workload_set, "P99 FTF")

    # # 5. visualize scheduling decision
    results_data = get_scheduling_from_raw_log(workload_set)
    plot_scheduling(results_data, workload_set,
                    wl_set_filter=[
                        "workloads-0.5",
                        "workloads-1.0",
                        "workloads-1.5",
                        "workloads-2.0",
                        "workloads-realistic",
                        "philly", "saturn", "newtrace"
                    ],
                    wl_filter=["1", "2", "3", "4", "5", "6", "7", "8"],
                    algo_filter=[
                        "ares",
                        "optimus",
                        "pollux",
                        "tiresias",
                        # "sjf"
                    ])

    pass


if __name__ == '__main__':
    # nohup python3 collect_result.py > collect_result.log 2>&1 &
    os.chdir(f"/home/cchen/yfliu/cluster_schedule/pollux/simulator/simulator_logs/Simulation-16nodes")

    # workload_sets = ["workloads-0.5", "workloads-1.0", "workloads-1.5", "workloads-2.0", "workloads-realistic"]
    workload_sets = ["philly", "saturn", "newtrace"]
    # workload_sets = ["newtrace"]
    for workload_set in workload_sets:
        main(workload_set)  # used for quick test and overall inspection
