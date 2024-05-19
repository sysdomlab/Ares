import collections
import math
from typing import Dict, Tuple, List

from policy.utils import JobInfo, NodeInfo


class FIFOPolicy(object):
    def __init__(self):
        pass

    def optimize(self,
                 jobs: Dict[Tuple[str, str], JobInfo],
                 nodes: Dict[str, NodeInfo],
                 prev_allocations: Dict[Tuple[str, str], List[str]],
                 node_template):
        print(f">>> jobs: {jobs}")
        print(f">>> nodes: {nodes}")
        print(f">>> prev_allocations: {prev_allocations}")

        # 1. allocate specified number of gpu for each job in queue
        num_gpus = sum(node.resources["nvidia.com/gpu"] for node in nodes.values())
        num_replicas = {}
        for key, job in sorted(jobs.items(), key=lambda item: item[1].creation_timestamp):  # fifo queue
            desire_replicas = math.ceil(job.target_batch_size / job.application.max_local_bsz)
            if desire_replicas > num_gpus:
                break
            desire_replicas = min(desire_replicas, num_gpus)
            num_replicas[key] = desire_replicas
            num_gpus -= desire_replicas

        # 2. Placements.
        # keep previous allocations if job still exits
        allocations = {k: v for k, v in prev_allocations.items() if k in jobs}
        # keep the same allocation if the number of replicas is not changed
        allocations = {k: v for k, v in allocations.items() if len(v) == num_replicas.get(k, 0)}

        total_gpus = {idx: int(node.resources['nvidia.com/gpu']) for idx, node in nodes.items()}
        free_gpus = collections.Counter(total_gpus) - collections.Counter(sum(allocations.values(), []))
        print(f">>> total_gpus: {total_gpus}")
        print(f">>> free_gpus: {free_gpus}")
        for key in sorted(jobs, key=lambda k: num_replicas.get(k, 0)):
            if num_replicas.get(key, 0) > 0 and not allocations.get(key):
                # Allocate resources.
                allocations[key] = []
                while len(allocations[key]) < num_replicas.get(key, 0):
                    node_idx, count = free_gpus.most_common(1)[0]
                    num = min(count, num_replicas.get(key, 0) - len(allocations[key]))
                    allocations[key].extend([node_idx] * num)
                    free_gpus[node_idx] -= num

        print(f">>> allocations: {allocations}")
        return allocations, len(nodes)
