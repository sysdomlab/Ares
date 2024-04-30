import os
from multiprocessing import Pool
from tqdm import tqdm

workloads = [
    # 'workloads-1.0/workload-debug.csv',
] + [
    f"workloads-{i}/workload-{j}.csv"
    for i in [
        "2.0", "1.5", "realistic", "0.5",
        "1.0",
    ]
    for j in [1, 2, 3, 4, 5, 6, 7, 8]
]
policies = [
    'pollux',
    'optimus',
    'tiresias',
    'sjf',
    'fifo',
    'cfq',
]
exp_name = "0.75_MaxBsz"

python3 = "/home/yfliu/lib/miniconda3/envs/pollux/bin/python3"

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
        command = f'{python3} simulator.py --workload {workload_path} --policy {policy} 2>&1 > {log_file}'
        commands.append(command)

print(f'Total commands: {len(commands)}')


def execute_command(command):
    try:
        os.system(command)
    except Exception as e:
        print(f'Error executing command: {command}')


if __name__ == '__main__':
    pool_size = os.cpu_count()
    with Pool(pool_size) as pool:
        list(tqdm(pool.imap_unordered(execute_command, commands), total=len(commands)))
