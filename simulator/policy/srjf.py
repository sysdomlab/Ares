import collections
import math
from typing import Dict, Tuple, List

from policy.utils import JobInfo, NodeInfo


class SRJFPolicy(object):
    def __init__(self):
        pass

    def optimize(self,
                 jobs: Dict[Tuple[str, str], JobInfo],
                 nodes: Dict[str, NodeInfo],
                 prev_allocations: Dict[Tuple[str, str], List[str]],
                 node_template=None):
        print(f">>> jobs: {jobs}")
        print(f">>> nodes: {nodes}")
        print(f">>> prev_allocations: {prev_allocations}")

        # Update remaining time for each job.
        for key, job in jobs.items():
            completion_epoch = job.application.get_completion_epoch(job.target_batch_size)

            completion_progress = job.application.get_progress(job.application.max_epochs)
            scale = job.target_batch_size / job.application.init_batch_size
            completion_iter = completion_progress / scale
            job.remaining = completion_iter * (1 - job.epoch / completion_epoch)

            # if completion_epoch <= job.epoch:
            #     job.remaining = 1
            # else:
            #     job.remaining = (job.application.get_iteration(job.target_batch_size, completion_epoch) -
            #                      job.application.get_iteration(job.target_batch_size, job.epoch))
            job.remaining_time = self.predict_step_time(job, job.max_replicas) * job.remaining
            print(f">>> job: {key[1]}, remaining: {job.remaining}, remaining_time: {job.remaining_time}")

        num_gpus = sum(node.resources["nvidia.com/gpu"] for node in nodes.values())
        print(f">>> num_gpus: {num_gpus}")
        num_replicas = {}
        # for key, job in sorted(jobs.items(), key=lambda item: item[1].remaining):
        for key, job in sorted(jobs.items(), key=lambda item: item[1].remaining_time):
            # num_replicas[key] = min(num_gpus, job.max_replicas)
            desire_replicas = job.max_replicas
            if desire_replicas > num_gpus:
                break
            num_replicas[key] = min(num_gpus, desire_replicas)
            num_gpus -= num_replicas[key]
        print(f">>> num_replicas: {num_replicas}")
        # Placements.
        allocations = {k: v for k, v in prev_allocations.items() if k in jobs}
        allocations = {k: v for k, v in allocations.items() if len(v) == num_replicas.get(k, 0)}
        job_keys = sorted(jobs, key=lambda k: num_replicas.get(k, 0))
        total_gpus = {idx: int(node.resources['nvidia.com/gpu']) for idx, node in nodes.items()}
        free_gpus = collections.Counter(total_gpus) - collections.Counter(sum(allocations.values(), []))
        for key in job_keys:
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

    def predict_step_time(self, job, num_replicas):
        placement = ()
        while sum(placement) < num_replicas:
            placement = (*placement, min(num_replicas - sum(placement), 4))
        local_bsz = math.ceil(job.target_batch_size / num_replicas - 1e-8)
        accum_steps = math.ceil(local_bsz / job.application.max_local_bsz - 1e-8) - 1
        if num_replicas == 1:
            accum_steps = max(1, accum_steps)
        atomic_bsz = math.ceil(local_bsz / (accum_steps + 1) - 1e-8)
        count = num_replicas * (accum_steps + 1)
        atomic_bsz = min(atomic_bsz, int(job.application.max_batch_size / count))
        # throughput = job.speedup_fn._goodput_fn.throughput(len(placement), num_replicas, atomic_bsz, accum_steps)
        # return atomic_bsz * count / throughput
        step_time, sync_time = job.application.get_throughput(placement, atomic_bsz)
        return step_time + (step_time - sync_time) * accum_steps
