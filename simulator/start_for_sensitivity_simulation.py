import os
import subprocess
from multiprocessing import Pool
from tqdm import tqdm

workloads = [
    f"{name}/workload-{j}.csv"
    for name in [
        # "newtrace",
        "philly",
        "saturn",
    ]
    for j in [1, 2, 3, 4, 5, 6, 7, 8]
]
policies = [
    'ares-0.95',
    'ares-0.90',
    'ares-0.85',
    'ares-0.80',
    'ares-0.75',
    'ares-0.70',
    'ares-0.65',
    'ares-0.60',
    'ares-0.55',
]
exp_name = "Simulation-sensitivity"

python3 = "/opt/miniconda3/envs/Ares/bin/python"

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
                   f' --workload {workload_path} --policy {policy.split("-")[0]} --output {json_file}'
                   f' --nodes "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"'
                   f' --ares_threshold {policy.split("-")[1]}'
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
