import os
import subprocess
from multiprocessing import Pool
from tqdm import tqdm

from policy.utils_gavel import get_gavel_policies

workloads = [
    f"{name}/workload-{j}.csv"
    for name in [
        # "workloads-2.0",
        # "workloads-1.5",
        # "workloads-realistic",
        # "workloads-1.0",
        # "workloads-0.5",
        "newtrace",
        "philly",
        "saturn",
        # "workloads-4h-40j",
    ]
    for j in [1, 2, 3, 4, 5, 6, 7, 8]
]
policies = [
    # 'srjf',
    # 'fifo',
    # "max_sum_throughput_perf",

    'pollux',
    'optimus',
    'tiresias',
    'ares',
    "finish_time_fairness_perf",
    "max_min_fairness",
    "allox"
]
exp_name = "Simulation-16nodes"

python3 = "/home/cchen/miniconda3/envs/yfliu/bin/python3"

# 构建要执行的命令列表
commands = []
for policy in policies:
    for workload in workloads:
        workload_path = f'./workload/{workload}'
        wl_set, wl_name = workload.split("/")[0], workload.split("/")[1].split(".")[0]
        log_dir = f'./simulator_logs/{exp_name}/{wl_set}/{wl_name}'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_file = f'{log_dir}/{policy}.txt'
        json_file = f'{log_dir}/{policy}.json'
        command = (f'{python3} simulator.py'
                   f' --workload {workload_path} --policy {policy} --output {json_file}'
                   f' --nodes "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"'
                   f' --interval {360 if policy in get_gavel_policies() else 60}'
                   f' 2>&1 > {log_file}')
        commands.append(command)

print(f'Total commands: {len(commands)}')


def execute_command(command):
    try:
        p = subprocess.Popen(command, shell=True)
        p.wait()
    except Exception as e:
        print(f'Error executing command: {command}')


if __name__ == '__main__':
    # nohup python3 start_for_full_simulation.py > start_for_full_simulation.log 2>&1 &
    pool_size = os.cpu_count() // 4 * 3
    with Pool(pool_size) as pool:
        list(tqdm(pool.imap_unordered(execute_command, commands), total=len(commands)))
