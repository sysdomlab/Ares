import os
import re
import csv

from policy.applications import APPLICATIONS
from profiler import get_placement_configs, ckp_path


def get_time_from_log(file_path):
    assert os.path.exists(file_path), f"Cannot find file {file_path}"
    pattern = r'batch=.*?\((.*?)\).*?sync=.*?\((.*?)\).*?gpu=.*?\((.*?)\).*?data=.*?\((.*?)\).*?other=.*?\((.*?)\)'
    with open(file_path, "r") as f:
        content = [i for i in f.readlines() if "[Epoch 0]" in i][0]
        matches = re.search(pattern, content)
        assert matches is not None, f"Cannot find profiling results in {file_path}: {content}"
        # print(content)
        # print(matches.groups())
        # input()
        batch_time, sync_time, gpu_time, data_time, other_time = matches.groups()
        print(f"{file_path}: batch_time = {batch_time}, sync_time = {sync_time}, "
              f"gpu_time = {gpu_time}, data_time = {data_time}, other_time = {other_time}")
        return {
            "batch_time": float(batch_time),
            "sync_time": float(sync_time),
            "gpu_time": float(gpu_time),
            "data_time": float(data_time),
            "other_time": float(other_time),
        }


def save_to_csv(head, data, file_name):
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(head)
        writer.writerows(data)


if __name__ == '__main__':
    scalability = False

    if not scalability:
        # profile placements.csv
        for app_name, app in APPLICATIONS.items():
            csv_data = []
            configs = get_placement_configs(app, scalability=False)

            print(f"app_name = {app_name}, num_configs = {len(configs)}, all_configs = {configs[:3]}")
            for config in configs:
                file = f"{ckp_path}{config['job_name']}/rank_0.log"
                res = get_time_from_log(file)

                placement_id = ''.join(map(str, config['placement']))
                local_bsz = config['local_bsz']

                csv_data.append([placement_id, local_bsz, res["batch_time"], res["sync_time"],
                                 res["gpu_time"], res["data_time"], res["other_time"]])

            # Save data to CSV
            out_path = f"traces/2080ti/{app_name}/placements.csv"
            if not os.path.exists(os.path.dirname(out_path)):
                os.makedirs(os.path.dirname(out_path))
            head = ["placement", "local_bsz", "step_time", "sync_time", "gpu_time", "data_time", "other_time"]
            save_to_csv(head, csv_data, out_path)
    else:
        # profile scalability.csv
        for app_name, app in APPLICATIONS.items():
            csv_data = []
            configs = get_placement_configs(app, scalability=True)

            print(f"app_name = {app_name}, num_configs = {len(configs)}, all_configs = {configs[:3]}")
            for config in configs:
                file = f"{ckp_path}{config['job_name']}/rank_0.log"
                res = get_time_from_log(file)

                placement = config['placement']
                local_bsz = config['local_bsz']
                num_nodes = sum([i != 0 for i in placement])
                num_replicas = sum(placement)

                csv_data.append([num_nodes, num_replicas, local_bsz, res["batch_time"], res["sync_time"],
                                 res["gpu_time"], res["data_time"], res["other_time"]])

            # Save data to CSV
            out_path = f"traces/2080ti/{app_name}/scalability.csv"
            if not os.path.exists(os.path.dirname(out_path)):
                os.makedirs(os.path.dirname(out_path))
            head = ["num_nodes", "num_replicas", "local_bsz", "step_time", "sync_time",
                    "gpu_time", "data_time", "other_time"]
            save_to_csv(head, csv_data, out_path)
