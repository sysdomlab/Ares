import collections
import copy

import math
from typing import Dict, Tuple, List

from policy.utils import JobInfo, NodeInfo


def optimize(job, interval):
    # update reference system
    update_virtual_time(interval)
    if has_new_job:
        # update the new job's virtual finish time
        add_new_job(job)
    if has_new_job or has_finished_job:
        # assign resources to prioritized jobs
        resource_allocation()

def update_virtual_time(interval):
    fair_share = total_gpus * interval / active_jobs_num
    ref_system.virtual_time += fair_share

def add_new_job(job):
    Calculate gpu_time of job
    job.virtual_finish_time = ref_system.virtual_time + gpu_time
    add job to jobs
    Sort jobs by virtual_finish_time in increasing order

def resource_allocation():
    for i in jobs:
        for j in free_gpu_nums:
            Calculate the efficiency of job i with j GPUs
            if efficiency < threshold:
                break
            Assign j GPUs to job i


class ARESPolicy(object):
    def __init__(self, time_fn, total_gpus, threshold=0.75):
        self._time_fn = time_fn
        self.gps_sys = GPSSystem(time_fn(), total_gpus)
        self.allocations = {}
        self.threshold = threshold

        # self.linear_gps_sys = LinearGPSSystem(total_gpus)



        has_new_job = self.gps_sys.check_and_add_new_job(jobs)
        has_finished_job = (collections.Counter(sum(self.allocations.values(), []))
                            != collections.Counter(sum(prev_allocations.values(), [])))



        # 3. 按所有任务的完成时间升序再分配资源，但要根据非线性伸缩系数进行控制
        num_gpus = sum(node.resources["nvidia.com/gpu"] for node in nodes.values())
        num_replicas = {}
        for key in self.gps_sys.finished_job_order:
            if key not in jobs.keys():
                continue
            desire_replicas = 0
            for i, (x, efficiency, speedup, d_speedup) in enumerate(self.gps_sys.fair_jobs[key]["scale_factor"]):
                if efficiency >= self.threshold:
                    desire_replicas = x
                else:
                    break
            num_replicas[key] = min(num_gpus, desire_replicas, jobs[key].max_replicas)
            num_gpus -= num_replicas[key]
        print(f">>> num_replicas: {num_replicas}")
        # Add remaining resources
        for key in self.gps_sys.finished_job_order:
            if key not in jobs.keys():
                continue
            desire_replicas = num_replicas[key]
            for i, (x, efficiency, speedup, d_speedup) in enumerate(self.gps_sys.fair_jobs[key]["scale_factor"]):
                if x < desire_replicas:
                    continue
                if x > jobs[key].max_replicas:
                    break
                if d_speedup > 0:
                    desire_replicas = x
                else:
                    break
            delta = desire_replicas - num_replicas[key]
            delta = min(delta, num_gpus)
            # print(f">>> num_gpus: {num_gpus}, delta: {delta}")
            num_replicas[key] = delta + num_replicas[key]
            num_gpus -= delta
        print(f">>> num_replicas: {num_replicas}")

        # Placements.
        allocations = {k: v for k, v in prev_allocations.items() if len(v) == num_replicas.get(k, 0)}
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

        # print(f">>> allocations: {allocations}")
        self.allocations = allocations
        self.gps_sys.total_gpus = sum(node.resources["nvidia.com/gpu"] for node in nodes.values())
        self.gps_sys.export_fair_state()
        return allocations, len(nodes)


class LinearGPSSystem:
    def __init__(self, total_gpus):
        self.fair_jobs: dict = {}
        self.virtual_time = 0
        self.finished_job_order = []
        self.total_gpus = total_gpus

    def _step(self):
        active_jobs = {k: j for k, j in self.fair_jobs.items() if j > self.virtual_time}
        active_jobs_num = len(active_jobs)
        fair_share = self.total_gpus * 1 / active_jobs_num if active_jobs_num > 0 else 0
        self.virtual_time += fair_share

    def _add_job(self, key, job: JobInfo):
        completion_progress = job.application.get_progress(job.application.max_epochs)
        scale = job.target_batch_size / job.application.init_batch_size
        completion_iter = completion_progress / scale
        # print(f"{key}'s completion_iter: {completion_iter}")

        step_time = predict_step_time(job, 1)
        gpu_minute = step_time * completion_iter / 60

        self.fair_jobs[key] = self.virtual_time + gpu_minute

    def check_and_add_new_job(self, jobs):
        # 检查并新增任务
        has_new_job = False
        for key, job in jobs.items():
            if key not in self.fair_jobs.keys():
                has_new_job = True
                self._add_job(key, job)
        if has_new_job:
            # self.finished_job_order = sorted(self.fair_jobs.items(), key=lambda x: self.fair_jobs[x[0]])
            self.finished_job_order = sorted(self.fair_jobs.keys(), key=lambda x: self.fair_jobs[x])
        self._step()
        # print(f">>> [LinearGPSSystem] finished_job_order: {self.finished_job_order}")
        # print(f">>> [LinearGPSSystem] virtual time: {self.virtual_time}")
        # input()
        return has_new_job
