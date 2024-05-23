import os

from policy.applications import APPLICATIONS


def get_best_placement(num_replicas):
    placement = ()
    while sum(placement) < num_replicas:
        placement = (*placement, min(num_replicas - sum(placement), 4))
    placement = tuple(filter(None, placement))
    placement = min(placement[i:] + placement[:i] for i in range(len(placement)))
    placement_id = int("".join(map(str, placement)))
    return placement, placement_id


local_bsz_dict = {
    "bert": [2, 4, 6, 8, 12],
    "cifar10": [32, 64, 128, 256, 512, 1024],
    "ncf": [1024, 2048, 4096, 8192, 16394, 32768],
    "imagenet": [16, 25, 32, 50, 64, 100],
    "deepspeech2": [5, 10, 20, 40, 80],
    "yolov3": [2, 4, 8, 16],
}

placements = sorted(list(set(tuple(sorted([i, j, m, n]))
                             for i in range(5)
                             for j in range(5)
                             for m in range(5)
                             for n in range(5))))
placements.remove((0, 0, 0, 0))
# print(f"len(placements) = {len(placements)}, placements = {placements}")

ckp_path = "/home/cchen/yfliu/ares/profile/"


def get_all_configs(application):
    max_local_bsz = application.max_local_bsz
    max_global_bsz = application.max_batch_size
    return [
        {
            "job_name": f"{application.name}-{''.join(map(str, placement))}-{local_bsz}",
            "app_name": application.name,
            "placement": placement,
            "local_bsz": local_bsz,
        }
        for placement in placements
        for local_bsz in local_bsz_dict[application.name]
        if sum(placement) * local_bsz <= max_global_bsz
    ]


def get_num_node(placement):
    return sum(map(lambda x: x > 0, placement))


def get_unprofiled_placements():
    pass


if __name__ == '__main__':
    available_node = (None, None, None, "10.0.0.23")
    all_configs = []
    for app_name, app in APPLICATIONS.items():

        configs = get_all_configs(app)
        all_configs += configs

        print(f"app_name = {app_name}, num_configs = {len(configs)}, all_configs = {configs}")

    # 过滤已测量的
    profiled_name = os.listdir(ckp_path)
    all_configs = [config for config in all_configs if config["job_name"] not in profiled_name]
    # 过滤现在可用机器无法测量的
    all_configs = [config for config in all_configs
                   if get_num_node(config["placement"]) <= sum(map(lambda x: x is not None, available_node))]
    # 构造命令
    for config in all_configs:
        job_name = config["job_name"]
        app_name = config["app_name"]
        placement = config["placement"]
        local_bsz = config["local_bsz"]
        allocation =

    # 循环执行并检测是否完成

    # 聚合profile
