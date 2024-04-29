import collections
import math
import requests

import logging

from typing import Dict, Tuple, List

from policy.utils import JobInfo, NodeInfo

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)

class AthenaPolicy(object):
    def __init__(self):
        LOG.info("Initializing AthenaPolicy")
        self.name = "AthenaPolicy"
        # self.gpu_good = ["ip-172-31-18-41"]
        # self.gpu_bad = ["ip-172-31-29-153"]
        self.gpu_good = [f"{i}" for i in range(0, 8)]
        self.gpu_bad = [f"{i}" for i in range(8, 16)]

    def adjust_placement(self, nodes, allocations):
        # allocations: {('default', 'cifar10-0'): ['ip-172-31-36-212']}
        # data: {"cifar10": 0.1, "imagenet": 0.9}

        # 1. get dataset status from juicefs
        url = 'http://localhost:8082/dataset'
        data: Dict[str, float] = {}
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                LOG.info(data)
            else:
                LOG.warning(f"Error: {response.status_code}")
        except Exception as e:
            LOG.warning(f"Error: {e}")

        data = {"deepspeech2": 0.1, "imagenet": 0.9, "yolov3": 0.7}

        # 2. 按照 busy 程度降序排序 allocations
        busy_jobs = [k for k in allocations.keys() if data.get(k[1].split('-')[0], 0) > 0.7]
        busy_jobs = sorted(busy_jobs, key=lambda k: data[k[1].split('-')[0]])
        LOG.info(f"busy_jobs: {busy_jobs}")
        need_adjust = False
        allocated_bad_gpu_num = 0
        for k in busy_jobs:
            for node_id in allocations[k]:
                if node_id in self.gpu_bad:
                    allocated_bad_gpu_num += 1
            # 如果 k 所在的节点存在 good gpu，则需要调整
            if any(node_id in self.gpu_good for node_id in allocations[k]):
                need_adjust = True
        if not need_adjust or allocated_bad_gpu_num == len(self.gpu_bad):
            return allocations
        LOG.info(f"start adjust")

        # 3. pop good job on bad gpu
        pop_list = []
        for k, alloc in allocations.items():
            # 如果 k 不在 busy_jobs 中且 alloc 中存在 bad gpu，则将其从 allocations 中移除
            if k not in busy_jobs and any(node_id in self.gpu_bad for node_id in alloc):
                LOG.info(f"Athena: pop job {k} as good job has bad gpu")
                pop_list.append(k)
        for k in pop_list:
            allocations.pop(k)

        # 4. reschedule jobs with IO bound to bad GPU node
        total_gpus = {idx: int(node.resources['nvidia.com/gpu']) for idx, node in nodes.items()}
        free_gpus = collections.Counter(total_gpus) - collections.Counter(sum(allocations.values(), []))
        free_bad_gpus = {node_id: count for node_id, count in free_gpus.items() if node_id in self.gpu_bad}
        # free_bad_gpus: {'ip-172-31-29-153': 4}
        for k in busy_jobs:
            # 从 free_bad_gpus 中计算剩余 bad gpu 总数
            free_bad_gpu_count = sum(free_bad_gpus.values())
            alloc = allocations.get(k)
            num_replicas = len(alloc)
            if num_replicas > free_bad_gpu_count or all(node_id in self.gpu_bad for node_id in alloc):
                continue
            new_alloc = []
            for node_id in alloc:
                if node_id in self.gpu_bad:
                    new_alloc.append(node_id)
                else:
                    # if len(free_bad_gpus) == 0:
                    #     break
                    # 拿到第一个 bad gpu 节点
                    new_node_id = next(iter(free_bad_gpus))
                    LOG.info(f"Athena: reschedule job {k} from {node_id} to {new_node_id}")
                    new_alloc.append(new_node_id)
                    free_bad_gpus[new_node_id] -= 1
                    if free_bad_gpus[new_node_id] == 0:
                        free_bad_gpus.pop(new_node_id)
            allocations[k] = new_alloc

        return allocations

    def optimize(self,
                 jobs: Dict[Tuple[str, str], JobInfo],
                 nodes: Dict[str, NodeInfo],
                 prev_allocations: Dict[Tuple[str, str], List[str]],
                 node_template):
        LOG.info(f">>> jobs: {jobs}")
        LOG.info(f">>> nodes: {nodes}")
        LOG.info(f">>> prev_allocations: {prev_allocations}")
        # keep previous allocations if job still exits
        allocations = {k: v for k, v in prev_allocations.items() if k in jobs}
        # AthenaPolicy: aware of the IO statue
        allocations = self.adjust_placement(nodes, allocations)

        # 1. allocate specified number of gpu for each job in queue
        num_gpus = sum(node.resources["nvidia.com/gpu"] for node in nodes.values())
        num_replicas = {}
        for key, job in sorted(jobs.items(), key=lambda item: item[1].creation_timestamp):  # fifo queue
            # todo: move desire_replicas to trace file
            desire_replicas = math.ceil(job.target_batch_size / job.application.max_local_bsz)
            # desire_replicas = 1
            desire_replicas = min(desire_replicas, num_gpus)
            num_replicas[key] = desire_replicas
            num_gpus -= desire_replicas

        # 2. Placements.
        # keep the same allocation if the number of replicas is not changed
        allocations = {k: v for k, v in allocations.items() if len(v) == num_replicas[k]}

        total_gpus = {idx: int(node.resources['nvidia.com/gpu']) for idx, node in nodes.items()}
        free_gpus = collections.Counter(total_gpus) - collections.Counter(sum(allocations.values(), []))
        LOG.info(f">>> total_gpus: {total_gpus}")
        LOG.info(f">>> free_gpus: {free_gpus}")
        for key in sorted(jobs, key=lambda k: num_replicas[k]):
            if num_replicas[key] > 0 and not allocations.get(key):
                # Allocate resources.
                allocations[key] = []
                while len(allocations[key]) < num_replicas[key]:
                    node_idx, count = free_gpus.most_common(1)[0]
                    num = min(count, num_replicas[key] - len(allocations[key]))
                    allocations[key].extend([node_idx] * num)
                    free_gpus[node_idx] -= num

        LOG.info(f">>> allocations: {allocations}")
        return allocations, len(nodes)
