import os
import subprocess
from multiprocessing import Pool
from tqdm import tqdm

node_num = {
    "scale_1x": 16,
    "scale_2x": 32,
    "scale_4x": 64,
    "scale_8x": 128,
    "scale_16x": 256,
    "scale_32x": 512,
}
workloads = [
    f"{name}/workload-{j}.csv"
    for name in [
        "scale_32x",
        "scale_16x",
        "scale_8x",
        "scale_4x",
        "scale_2x",
        "scale_1x",
    ]
    for j in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
]
policies = [
    # 'pollux',
    # 'optimus',
    # 'tiresias',
    'ares',
]
exp_name = "Simulation-scale"

python3 = "/home/cchen/miniconda3/envs/yfliu/bin/python3"

# 构建要执行的命令列表
commands = []
for policy in policies:
    for workload in workloads:
        workload_path = f'./workload/scale/{workload}'
        wl_set, wl_name = workload.split("/")[0], workload.split("/")[1].split(".")[0]
        log_dir = f'./simulator_logs/{exp_name}/{wl_set}/{wl_name}'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_file = f'{log_dir}/{policy}.txt'
        json_file = f'{log_dir}/{policy}.json'
        command = (f'{python3} simulator.py'
                   f' --workload {workload_path} --policy {policy} --output {json_file}'
                   f' --nodes "{" ".join([str(i) for i in range(node_num[wl_set])])}"'
                   f' --early_exit {60 if policy == "pollux" else 480}'
                   f' --interval {480 if policy == "pollux" else 60}'
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
    # nohup python3 start_for_scale_simulation.py > start_for_scale_simulation.log 2>&1 &
    pool_size = os.cpu_count() // 4 * 3
    with Pool(pool_size) as pool:
        list(tqdm(pool.imap_unordered(execute_command, commands), total=len(commands)))
