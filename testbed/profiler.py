import os
import subprocess
import time

import zerorpc

from models.env import get_checkpoint_path
from models.tool import get_cmd
from policy.applications import APPLICATIONS


def get_local_bsz_dict():
    return {
        "bert": [2, 3, 4, 6, 8, 12],
        "cifar10": [32, 64, 128, 256, 512, 1024],
        "ncf": [256, 512, 1024, 2048, 4096, 8192, 16394, 32768],
        "imagenet": [16, 25, 32, 50, 64, 100],
        "deepspeech2": [5, 10, 20, 40, 80],
        "yolov3": [2, 4, 8, 16],
    }


def get_placements(scalability=False):
    if not scalability:
        placements = sorted(list(set(tuple(sorted([i, j, m, n]))
                                     for i in range(5)
                                     for j in range(5)
                                     for m in range(5)
                                     for n in range(5))))
        placements.remove((0, 0, 0, 0))
        # print(f"len(placements) = {len(placements)}, placements = {placements}")
        return placements
    else:
        placements = []
        num_nodes = 8
        for i in range(5, num_nodes + 1):
            for j in range(1, 5):
                placement = tuple([0] * (num_nodes - i) + [j] * i)
                placements.append(placement)
        return placements


ckp_path = "/home/cchen/yfliu/ares/profile/"


def get_placement_configs(application, scalability=False):
    max_local_bsz = application.max_local_bsz
    max_global_bsz = application.max_batch_size
    return [
        {
            "job_name": f"profile-{application.name}-{''.join(map(str, placement))}-{local_bsz}",
            "app_name": application.name,
            "placement": placement,
            "local_bsz": local_bsz,
        }
        for placement in get_placements(scalability)
        for local_bsz in get_local_bsz_dict().get(application.name, [])
        if sum(placement) * local_bsz <= max_global_bsz
    ]


def performance_profiling(available_node, configs):
    assert len(available_node) == len(configs[0]["placement"])

    # 过滤已测量的
    profiled_job_name = os.listdir(ckp_path)
    configs = [config for config in configs if config["job_name"] not in profiled_job_name]
    # 过滤现在可用机器无法测量的
    configs = [config for config in configs if
               sum(map(lambda x: x > 0, config["placement"])) <= sum(map(lambda x: x is not None, available_node))]
    print(
        f"num_configs = {len(configs)}, all_configs = {configs[:3]}"
    )

    # input("press any key to start profiling...")

    connects = {}
    for node in available_node:
        if node is not None:
            connects[node] = zerorpc.Client()
            connects[node].connect(f"tcp://{node}:4242")
    # 构造命令
    for config in configs:
        job_name = config["job_name"]
        app_name = config["app_name"]
        placement = config["placement"]
        local_bsz = config["local_bsz"]
        allocation_ng = [(available_node[i], j)
                         for i in range(len(available_node)) if available_node[i] is not None
                         for j in range(placement[i])]
        allocation_n = [available_node[i]
                        for i in range(len(available_node)) if available_node[i] is not None
                        for j in range(placement[i])]
        print(
            f"job_name = {job_name}, allocation = {allocation_ng}"
        )
        for rank, (node_ip, gpu_id) in enumerate(allocation_ng):
            proc_name = f"{job_name}:{rank}"
            cmd = get_cmd(job_name=job_name, allocation=allocation_n, rank=rank,
                          acc_bsz=local_bsz, acc_step=1, model_name=app_name) + " --profile"
            out_file = f"{get_checkpoint_path(job_name, return_dir=True)}/rank_{rank}.log"
            print(f"connects[{node_ip}].run_proc({proc_name}, {cmd}, {gpu_id}, {out_file})")
            # input()
            connects[node_ip].run_proc(proc_name, cmd, gpu_id, out_file)
        # input()
        for rank, (node_ip, gpu_id) in enumerate(allocation_ng):
            proc_name = f"{job_name}:{rank}"
            time_out, start_time = 120, time.time()
            while True:
                res = connects[node_ip].stats_proc(proc_name)
                print(f"connects[{node_ip}].stats_proc({proc_name})")
                if res.get("returncode", None) is not None:
                    print(f"Process {proc_name}({node_ip}:{gpu_id}) terminated with returncode {res['returncode']}")
                    break
                if time.time() - start_time > time_out:
                    raise TimeoutError(f"Process {proc_name}({node_ip}:{gpu_id}) terminated due to timeout")
                time.sleep(1)


if __name__ == '__main__':
    # nohup python3 -u profiler.py > /home/cchen/yfliu/ares/logs/profiler_scalability.log 2>&1 &

    scalability = False

    if not scalability:
        # profile placement.csv
        available_node = ("10.0.0.23", "10.0.0.24", "10.0.0.25", "10.0.0.26")
        all_configs = []
        for app_name, app in APPLICATIONS.items():
            configs = get_placement_configs(app, scalability=False)
            all_configs += configs

            print(
                f"app_name = {app_name}, num_configs = {len(configs)}"
                # f", all_configs = {configs}"
            )
        # do profiling
        performance_profiling(available_node, all_configs)
        # subprocess.Popen(f"mv /home/cchen/yfliu/ares/checkpoints/profile-* {ckp_path}", shell=True)
    else:
        # profile scalability.csv
        available_node = ("10.0.0.12", "10.0.0.13",
                          "10.0.0.15", "10.0.0.16", "10.0.0.17",
                          "10.0.0.20", "10.0.0.21", "10.0.0.22")
        all_configs = []
        for app_name, app in APPLICATIONS.items():
            configs = get_placement_configs(app, scalability=True)
            all_configs += configs

            print(
                f"app_name = {app_name}, num_configs = {len(configs)}"
                # f", all_configs = {configs}"
            )
        # do profiling
        performance_profiling(available_node, all_configs)
        # subprocess.Popen(f"mv /home/cchen/yfliu/ares/checkpoints/profile-* {ckp_path}", shell=True)


