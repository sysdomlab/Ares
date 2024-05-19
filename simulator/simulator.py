import argparse
import collections
import json
import time
from typing import List

import math
import numpy as np
import pandas

from policy.applications import APPLICATIONS
from goodput import GoodputFunction, fit_perf_params
from policy.speedup import SpeedupFunction
from policy.utils import JobInfo, NodeInfo

from policy.pollux import PolluxPolicy
from policy.optimus import OptimusPolicy
from policy.tiresias import TiresiasPolicy
from policy.fifo import FIFOPolicy
from policy.athena import AthenaPolicy
from policy.cfq import CFQPolicy
from policy.sjf import SJFPolicy


def get_all_policies():
    return ["tiresias", "optimus", "pollux", "fifo", "sjf", "athena", "cfq"]


class Job(object):
    def __init__(self, name, application, submission_time,
                 target_num_replicas=None, target_batch_size=None):
        self.name = ("default", name)
        self.application = application
        self.submission_time = submission_time
        self.target_num_replicas = target_num_replicas
        self.target_batch_size = target_batch_size
        self.completion_time = None
        self.current_time = 0
        self.rescale_time = 0
        self.placement = ()
        self.atomic_bsz = 0
        self.accum_steps = 0
        self.profile = {}
        self.perf_params = None
        self.grad_params = None
        self.best_metric = None
        self.progress = 0.0
        self.epoch = 0
        self.attained_service = 0
        self.num_restarts = None

    @property
    def max_profiled_replicas(self):
        return max((k[1] for k in self.profile), default=0)

    def get_goodput_fn(self):
        app = self.application
        return GoodputFunction(self.perf_params, self.grad_params, app.init_batch_size)

    def get_speedup_fn(self):
        if self.perf_params is None:
            return lambda n, r: r
        app = self.application
        return SpeedupFunction(self.get_goodput_fn(), app.max_batch_size,
                               (app.min_local_bsz, app.max_local_bsz),
                               accumulation=True)

    def update_local_bsz(self, placement):
        app = self.application
        placement = tuple(filter(None, placement))
        num_nodes, num_replicas = len(placement), sum(placement)
        batch_size = self.target_batch_size
        if batch_size is None and self.perf_params is None:
            batch_size = max(app.init_batch_size, app.min_local_bsz * num_replicas)
        if batch_size is None:
            goodput_fn = self.get_goodput_fn()
            _, self.atomic_bsz, self.accum_steps = goodput_fn.optimize(
                num_nodes, num_replicas, app.max_batch_size,
                (app.min_local_bsz, app.max_local_bsz), accumulation=True)
        else:
            local_bsz = math.ceil(batch_size / num_replicas - 1e-8)
            self.accum_steps = math.ceil(local_bsz / app.max_local_bsz - 1e-8) - 1
            if num_replicas == 1 and batch_size > app.init_batch_size:
                # assert self.accum_steps > 0
                self.accum_steps = max(1, self.accum_steps)
            self.atomic_bsz = math.ceil(local_bsz / (self.accum_steps + 1) - 1e-8)
        count = num_replicas * (self.accum_steps + 1)
        self.atomic_bsz = min(self.atomic_bsz, int(app.max_batch_size / count))

    def update_params(self, num_nodes, num_replicas, local_bsz,
                      step_time, sync_time, grad_sqr, grad_var):
        self.grad_params = (grad_sqr, grad_var)
        if (num_nodes, num_replicas, local_bsz) in self.profile:
            return
        self.profile[num_nodes, num_replicas, local_bsz] = step_time, sync_time
        num_nodes = np.array([key[0] for key in self.profile])
        num_replicas = np.array([key[1] for key in self.profile])
        local_bsz = np.array([key[2] for key in self.profile])
        step_time = np.array([val[0] for val in self.profile.values()])
        sync_time = np.array([val[1] for val in self.profile.values()])
        compute_time = step_time - sync_time
        self.perf_params = fit_perf_params(num_nodes, num_replicas, local_bsz, compute_time, step_time)

    def step(self, seconds=60):  # todo 检查改变的变量
        if not self.placement:
            # No resources are allocated to this job.
            self.current_time += seconds
            return
        delay = min(self.rescale_time, seconds)
        self.current_time += delay
        self.attained_service += delay * sum(self.placement)
        self.rescale_time -= delay
        seconds -= delay
        while seconds > 0 and self.completion_time is None:
            assert self.epoch < self.application.max_epochs
            # Calculate current job configurations.
            placement = tuple(filter(None, self.placement))
            num_nodes, num_replicas = len(placement), sum(placement)
            batch_size = num_replicas * self.atomic_bsz * (self.accum_steps + 1)
            scale = batch_size / self.application.init_batch_size
            # Calculate true (simulated) throughput.
            step_time, sync_time = self.application.get_throughput(placement, self.atomic_bsz)
            accum_time = step_time - sync_time
            # Calculate true (simulated) efficiency.
            grad_sqr, grad_var = self.application.get_grad_stats(batch_size, self.epoch)  # todo 更新pollux参数
            gain = (grad_var + grad_sqr) / (grad_var / scale + grad_sqr)
            # Update the estimated throughput/efficiency parameters.
            self.update_params(num_nodes, num_replicas, self.atomic_bsz,
                               step_time, sync_time, grad_sqr, grad_var)
            # Calculate true (simulated) goodput.
            total_time = step_time + accum_time * self.accum_steps  # sec per iter
            # goodput = gain / total_time  # progress per iter * iter per sec
            goodput = scale / total_time  # progress per iter * iter per sec
            # Update current epoch and progress.
            next_progress = self.application.get_progress(self.epoch + 1)  # get_progress 返回的是标准 iter 数
            # print(f"<<< job: {self.name}, self.epoch: {self.epoch}, \n"
            #       f"scale: {scale}, gain: {gain}, num_replicas: {num_replicas}, \n"
            #       f"goodput: {goodput}, "
            #       f"total_time: {total_time}, "
            #       f"next_progress: {next_progress}")
            if self.progress + goodput * seconds < next_progress:
                # Used up the entire time interval without finishing an epoch.
                self.progress += goodput * seconds
                self.current_time += seconds
                self.attained_service += seconds * sum(self.placement)
                seconds = 0
            else:
                # Crossed an epoch boundary before finishing the time interval.
                self.epoch += 1
                delta = round(float((next_progress - self.progress) / goodput))
                assert delta <= seconds
                completion_epoch = self.application.get_completion_epoch(batch_size)
                if self.epoch > completion_epoch:
                    self.completion_time = self.current_time + delta
                self.progress = next_progress
                self.best_metric = self.application.get_best_metric(batch_size, self.epoch)
                self.current_time += delta
                self.attained_service += delta * sum(self.placement)
                seconds -= delta
                # Re-scale batch size between epochs.
            self.update_local_bsz(self.placement)
        self.current_time += seconds  # Add any remaining time.

    def reallocate(self, placement):
        if placement:
            self.placement = tuple(placement)
            self.update_local_bsz(self.placement)
            self.rescale_time = 30  # Start re-scale countdown.
            if self.num_restarts is None:
                self.num_restarts = 0
            else:
                self.num_restarts += 1
        else:  # De-allocate all resources.
            self.placement = ()
            self.atomic_bsz = 0


class Cluster(object):
    def __init__(self, workload_name, policy_name, nodes, num_gpus=4, interval=60):
        assert 1 <= num_gpus <= 4
        self.workload = pandas.read_csv(workload_name)
        self.nodes = nodes.split(",")
        self.num_nodes = len(self.nodes)
        self.num_gpus = num_gpus
        self.interval = interval
        self.current_time = 0
        self.real_time = time.time()  # todo 有用吗
        self.jobs = [Job(name=row.name,
                         application=APPLICATIONS[row.application],
                         submission_time=row.time,
                         target_num_replicas=None if policy_name in [] else row.num_replicas,
                         target_batch_size=None if policy_name in [] else APPLICATIONS[row.application].max_batch_size)
                     # target_batch_size=None if policy_name in [] else row.batch_size)
                     for row in self.workload.itertuples()]
        self.policy = self.get_policy(policy_name)

        self.allocations = {}
        self.logs = []

    def get_policy(self, policy_name):
        assert policy_name in get_all_policies()
        if policy_name == "tiresias":
            return TiresiasPolicy(lambda: self.current_time)
        elif policy_name == "optimus":
            return OptimusPolicy()
        elif policy_name == "pollux":
            return PolluxPolicy()
        elif policy_name == "fifo":
            return FIFOPolicy()
        elif policy_name == "sjf":
            return SJFPolicy()
        elif policy_name == "athena":
            return AthenaPolicy()
        elif policy_name == "cfq":
            return CFQPolicy(lambda: self.current_time, self.num_gpus * self.num_nodes)

    def step(self):
        for job in self.jobs:
            job.step()
            pass  # todo 要更新每个任务的状态、current_time
        self.current_time += self.interval
        job_infos = self.get_job_infos()
        node_infos = self.get_node_infos()
        if job_infos:
            # Optimize allocations.
            self.allocations = {k: v for k, v in self.allocations.items() if k in job_infos}

            allocations, desired_nodes = self.policy.optimize(job_infos, node_infos, self.allocations)

            used_gpus = collections.Counter(sum(allocations.values(), []))
            assert all(val <= node_infos[key].resources["nvidia.com/gpu"] for key, val in used_gpus.items())

            # 1. kill jobs that are not in the allocation
            killed_jobs = [job_name for job_name in self.allocations.keys() if job_name not in allocations.keys()]
            started_jobs = [job_name for job_name in allocations.keys() if job_name not in self.allocations.keys()]
            # for job in self.jobs:
            #     pass  # todo allocate 修改bsz、杀死任务、启动任务
            # 2. wait for jobs to be killed
            # 3. allocate jobs according to the new allocation
            for job in self.jobs:
                if allocations.get(job.name) != self.allocations.get(job.name):
                    alloc = allocations.get(job.name, [])
                    placement = []
                    for i in range(len(alloc)):
                        if i == 0 or alloc[i] != alloc[i - 1]:
                            placement.append(1)
                        else:
                            placement[-1] += 1
                    job.reallocate(placement)
            self.allocations = allocations

    def get_job_infos(self):
        job_infos = {}
        for job in self.jobs:
            # todo 要更新每个任务的状态、current_time
            # todo: 为完成的任务添加 completion_time
            if self.current_time >= job.submission_time and job.completion_time is None:
                # todo 查询并更新 job 的状态
                if isinstance(self.policy, TiresiasPolicy):
                    job_infos[job.name] = self.get_tiresias_job_info(job)
                elif isinstance(self.policy, OptimusPolicy):
                    job_infos[job.name] = self.get_optimus_job_info(job)
                elif isinstance(self.policy, PolluxPolicy):
                    job_infos[job.name] = self.get_pollux_job_info(job)
                elif isinstance(self.policy, FIFOPolicy):
                    job_infos[job.name] = self.get_optimus_job_info(job)
                elif isinstance(self.policy, SJFPolicy):
                    job_infos[job.name] = self.get_optimus_job_info(job)
                elif isinstance(self.policy, AthenaPolicy):
                    job_infos[job.name] = self.get_optimus_job_info(job)
                elif isinstance(self.policy, CFQPolicy):
                    job_infos[job.name] = self.get_optimus_job_info(job)
        return job_infos

    def get_node_infos(self):
        return {
            ip: NodeInfo({"nvidia.com/gpu": self.num_gpus}, preemptible=False)
            for ip in self.nodes
        }

    def get_pollux_job_info(self, job):
        job_info = JobInfo(
            resources={"nvidia.com/gpu": 1},
            speedup_fn=job.get_speedup_fn(),
            creation_timestamp=job.submission_time,
            attained_service=job.attained_service,
            min_replicas=0,
            max_replicas=min(max(2 * job.max_profiled_replicas, 1), 64,  # simulator can't handle more.
                             job.application.max_batch_size // job.application.min_local_bsz),
        )
        job_info.num_restarts = job.num_restarts or 0
        job_info.age = self.current_time - job.submission_time
        return job_info

    def get_optimus_job_info(self, job):
        job_info = JobInfo(
            resources={"nvidia.com/gpu": 1},
            speedup_fn=job.get_speedup_fn(),
            creation_timestamp=job.submission_time,
            attained_service=job.attained_service,
            min_replicas=0,
            max_replicas=min(max(2 * job.max_profiled_replicas, 1), 64,  # simulator can't handle more.
                             job.application.max_batch_size // job.application.min_local_bsz),
            # max_replicas=(job.target_batch_size // job.application.min_local_bsz),
        )
        job_info.epoch = job.epoch
        job_info.application = job.application
        job_info.target_batch_size = job.target_batch_size
        return job_info

    def get_tiresias_job_info(self, job):
        return JobInfo(
            resources={"nvidia.com/gpu": 1},
            speedup_fn=None,
            creation_timestamp=job.submission_time,
            attained_service=job.attained_service,
            min_replicas=0,
            max_replicas=job.target_num_replicas,
        )

    def all_complete(self):
        return all(job.completion_time is not None for job in self.jobs)

    def output_logs(self, path):
        with open(path, "w") as f:
            for record in self.logs:
                json.dump(record, f)
                f.write("\n")

    def get_jcts(self):
        return {
            val["name"]: val["completion_time"] - val["submission_time"]
            for val in self.logs[-1]["submitted_jobs"]
            if val["completion_time"] is not None
        }

    def run(self):
        while not self.all_complete():
            self.step()
            self.print_logs()
            # time.sleep(self.interval)
        return self.logs, self.get_jcts()

    def print_logs(self):
        self.logs.append({
            "timestamp": self.current_time,
            "num_nodes": self.num_nodes,
            "allocations": self.allocations,
            "submitted_jobs": [
                {
                    "name": job.name,
                    "epoch": job.epoch,
                    "progress": job.progress,
                    "num_restarts": job.num_restarts,
                    "allocation": self.allocations.get(job.name, []),
                    "placement": job.placement,
                    "batch_size": job.atomic_bsz * (job.accum_steps + 1) * sum(job.placement),
                    "accum_steps": job.accum_steps,
                    "submission_time": job.submission_time,
                    "completion_time": job.completion_time,
                    "grad_params": job.grad_params,
                }
                for job in self.jobs if job.submission_time <= self.current_time
            ],
        })
        print(f"---------------- SIMULATOR TIME: {self.current_time} ----------------")
        print("Active jobs:")
        for val in self.logs[-1]["submitted_jobs"]:
            if val["submission_time"] <= self.current_time and val["completion_time"] is None:
                print(f"    {val['name']}:\t[epoch {val['epoch']}]\t[restarts {val['num_restarts']}]\t"
                      f"[batch size {val['batch_size']}]\t[placement {val['placement']}]")
        print(f"allocations: {self.logs[-1]['allocations']}")
        used_gpus = sum(map(len, self.allocations.values()))
        print("Active jobs:")
        print("GPU utilization: {}".format(used_gpus))
        jct_dict = self.get_jcts()
        print(f"Completed jobs [{len(jct_dict)}]:")
        print(jct_dict)
        print("Average JCT:", sum(jct_dict.values()) / len(jct_dict) if jct_dict else 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=str, help="path to workload csv",
                        default="./workload/workloads-1.0/workload-1.csv")
                        # default="./workload/workloads-1.0/workload-debug.csv")
    parser.add_argument("--policy", type=str, default="cfq",
                        choices=get_all_policies())
    # parser.add_argument("--nodes", type=str, default=",".join([f"10.0.0.{i}" for i in range(19, 19 + 16)]),
    #                     help="min number of nodes in the cluster")
    parser.add_argument("--nodes", type=str, default=",".join([f"{i}" for i in range(0, 16)]),
                        help="min number of nodes in the cluster")
    parser.add_argument("--interval", type=int, default=60,
                        help="scheduling interval in seconds")
    parser.add_argument("--num-gpus", type=int, default=4,
                        help="number of GPUs per node")
    args = parser.parse_args()

    cluster = Cluster(args.workload, args.policy, args.nodes, num_gpus=args.num_gpus, interval=args.interval)
    cluster.run()
