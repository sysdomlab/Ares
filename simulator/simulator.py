import argparse
import collections
import json
import os
import time

import math
import numpy as np
import pandas
import zerorpc

from framework import load_job_state
from models.tool import get_cmd
from models.env import get_checkpoint_path
from policy.applications import APPLICATIONS
from policy.goodput import GoodputFunction, fit_perf_params
from policy.speedup import SpeedupFunction
from policy.utils import JobInfo, NodeInfo

from policy.pollux import PolluxPolicy
from policy.optimus import OptimusPolicy
from policy.tiresias import TiresiasPolicy
from policy.fifo import FIFOPolicy
from policy.athena import AthenaPolicy
from policy.ares import ARESPolicy
from policy.sjf import SJFPolicy

np.random.seed(0)


def random_scale():
    return (np.random.randint(2) + 1) / 2


def get_all_policies():
    return ["tiresias", "optimus", "pollux", "fifo", "sjf", "athena", "ares"]


class Job(object):
    def __init__(self, name, application, submission_time,
                 target_num_replicas=None, target_batch_size=None):
        self.name = name
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

    def step(self, seconds=60):
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
            grad_sqr, grad_var = self.application.get_grad_stats(batch_size, self.epoch)
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
    def __init__(self, workload_name, policy_name, nodes, num_gpus=4, interval=60, out_put=None):
        assert 1 <= num_gpus <= 4
        self.nodes = nodes.split(" ")
        self.num_nodes = len(self.nodes)
        self.num_gpus = num_gpus
        self.interval = interval
        self.current_time = 0
        self.start_time = time.time()
        self.jobs = collections.OrderedDict()
        for row in pandas.read_csv(workload_name).itertuples():
            self.jobs[row.name] = Job(
                name=row.name,
                application=APPLICATIONS[row.application],
                submission_time=row.time,
                target_num_replicas=row.num_replicas,
                target_batch_size=APPLICATIONS[row.application].max_batch_size * random_scale(),
                # target_batch_size=None if policy_name in [] else row.batch_size,
            )
        self.policy = self.get_policy(policy_name)
        self.out_put = out_put

        self.running_allocations = {}
        # self.running_allocations[job_name] = [(node_ip, gpu_id)]
        self.gpu_alloc = {(ip, i): [] for ip in self.nodes for i in range(self.num_gpus)}
        # gpu_alloc[(node_ip, gpu_id)] = [proc_name]

        self.logs = []

        # self.connects = {}
        # for node_ip in self.nodes:
        #     self.connects[node_ip] = zerorpc.Client()
        #     self.connects[node_ip].connect(f"tcp://{node_ip}:4242")

    def get_free_gpu(self, node):
        for i in range(self.num_gpus):
            if len(self.gpu_alloc[(node, i)]) == 0:
                return i
        raise ValueError(f"No free GPU on node {node}: {[self.gpu_alloc[(node, i)] for i in range(self.num_gpus)]}")

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
        elif policy_name == "ares":
            return ARESPolicy(lambda: self.current_time, self.num_gpus * self.num_nodes)

    def get_job_infos(self):
        job_infos = {}
        for job in self.jobs.values():
            if self.current_time >= job.submission_time and job.completion_time is None:
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
                elif isinstance(self.policy, ARESPolicy):
                    job_infos[job.name] = self.get_ares_job_info(job)
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
            max_replicas=min(max(2 * job.max_profiled_replicas, 1), job.application.max_num_replicas,
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
            max_replicas=job.target_batch_size // job.application.min_local_bsz,
        )
        job_info.epoch = job.epoch
        job_info.application = job.application
        job_info.target_batch_size = job.target_batch_size
        return job_info

    def get_ares_job_info(self, job):
        job_info = JobInfo(
            resources={"nvidia.com/gpu": 1},
            speedup_fn=job.get_speedup_fn(),
            creation_timestamp=job.submission_time,
            attained_service=job.attained_service,
            min_replicas=0,
            max_replicas=job.application.max_num_replicas
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
        return all(job.completion_time is not None for job in self.jobs.values())

    def optimize(self, job_infos, node_infos, prev_allocations):
        # 算法选择
        new_allocations, _ = self.policy.optimize(job_infos, node_infos, prev_allocations)
        # 检查合法性
        for k in new_allocations.keys():
            new_allocations[k] = sorted(new_allocations[k])
        used_gpus = collections.Counter(sum(new_allocations.values(), []))
        assert all(val <= node_infos[key].resources["nvidia.com/gpu"] for key, val in used_gpus.items())
        # 修改bsz
        for job in self.jobs.values():
            if new_allocations.get(job.name) != prev_allocations.get(job.name):
                alloc = new_allocations.get(job.name, [])
                placement = []
                for i in range(len(alloc)):
                    if i == 0 or alloc[i] != alloc[i - 1]:
                        placement.append(1)
                    else:
                        placement[-1] += 1
                job.reallocate(placement)
        return new_allocations

    def step(self):
        self.current_time += self.interval
        for job in self.jobs.values():
            job.step()
        job_infos = self.get_job_infos()
        node_infos = self.get_node_infos()
        # 过滤已完成的任务
        finished_proc = []
        for k, v in self.running_allocations.items():
            if k not in job_infos:
                for rank, (node_ip, gpu_id) in enumerate(v):
                    proc_name = f"{k}:{rank}"
                    finished_proc.append((proc_name, (node_ip, gpu_id)))
                    # res = self.connects[node_ip].stats_proc(proc_name)
                    # print(f"self.connects[{node_ip}].stats_proc({proc_name}): {res}")
                    # assert res is not None
                    # if res.get("returncode") is None:
                    #     print(f"[WARN]Process {proc_name}({node_ip}:{gpu_id}) is still running")
                    #     self.connects[node_ip].kill_proc(proc_name, force=True)
                    # if res.get("returncode") != 0:
                    #     print(f"[WARN]Process {proc_name}({node_ip}:{gpu_id}) exited unexpectedly")
                    self.gpu_alloc[(node_ip, gpu_id)].remove(f"{k}:{rank}")
            else:
                for rank, (node_ip, gpu_id) in enumerate(v):
                    proc_name = f"{k}:{rank}"
                    # res = self.connects[node_ip].stats_proc(proc_name)
                    # print(f"self.connects[{node_ip}].stats_proc({proc_name}): {res}")
                    # if res is not None and res.get("returncode") not in [0, None]:
                    #     raise ValueError(f"Process {proc_name}({node_ip}:{gpu_id}) exited unexpectedly")
        self.running_allocations = {k: v for k, v in self.running_allocations.items() if k in job_infos}
        if not job_infos:
            return

        # Optimize allocations.
        prev_allocations = {k: [v[0] for v in v] for k, v in self.running_allocations.items()}
        new_allocations = self.optimize(job_infos, node_infos, prev_allocations)

        # 1. kill jobs that are not in the allocation
        kill_job = [k for k, v in prev_allocations.items() if v != new_allocations.get(k, [])]
        kill_proc = []
        for job_name in kill_job:
            for rank, (node_ip, gpu_id) in enumerate(self.running_allocations[job_name]):
                proc_name = f"{job_name}:{rank}"
                # print(f"self.connects[{node_ip}].kill_proc({proc_name})")
                # self.connects[node_ip].kill_proc(proc_name)  # 非阻塞发送停止命令
                kill_proc.append((proc_name, (node_ip, gpu_id)))
            del self.running_allocations[job_name]  # 将进程解除注册并需要等待确认进程停止
        # 2. wait for jobs to be killed
        for proc_name, (node_ip, gpu_id) in kill_proc:
            # time_out, start_time = 60, time.time()  # 有限时间等待进程停止，否则可能在进程管理方面出错或梯度累积过多
            # force_kill_time, force_killed = 30, False
            # while True:
            #     res = self.connects[node_ip].stats_proc(proc_name)
            #     # res = None
            #     if res is None or res.get("returncode") is not None:
            #         print(f"self.connects[{node_ip}].stats_proc({proc_name}): {res}")
            #         break
            #     if time.time() - start_time > force_kill_time and not force_killed:
            #         print(f"self.connects[{node_ip}].kill_proc({proc_name}, force=True)")
            #         self.connects[node_ip].kill_proc(proc_name, force=True)  # 超时后强制杀死进程
            #         force_killed = True
            #     if time.time() - start_time > time_out:
            #         raise ValueError(f"Timeout waiting for process {proc_name}({node_ip}:{gpu_id}) to be killed")
            #     time.sleep(1)
            self.gpu_alloc[(node_ip, gpu_id)].remove(proc_name)  # 进程已完全停止，从gpu_alloc中移除
        # 3. allocate jobs according to the new allocation
        start_proc = []
        start_job = [k for k, v in new_allocations.items() if v != prev_allocations.get(k, [])]
        for job_name in start_job:
            self.running_allocations[job_name] = []
            for rank, node_ip in enumerate(new_allocations[job_name]):
                gpu_id = self.get_free_gpu(node_ip)
                self.running_allocations[job_name].append((node_ip, gpu_id))
                proc_name = f"{job_name}:{rank}"
                start_proc.append((proc_name, (node_ip, gpu_id)))
                self.gpu_alloc[(node_ip, gpu_id)].append(proc_name)  # 进程注册到gpu_alloc
                # job = self.jobs[job_name]
                # cmd = get_cmd(job_name=job_name, allocation=new_allocations[job_name], rank=rank,
                #               acc_bsz=job.atomic_bsz, acc_step=job.accum_steps + 1)
                # out_file = (f"{get_checkpoint_path(job_name, return_dir=True)}/"
                #             f"restart_{job.num_restarts}_rank_{rank}.log")
                # print(f"self.connects[{node_ip}].run_proc({proc_name}, {cmd}, {gpu_id}, {out_file})")
                # self.connects[node_ip].run_proc(proc_name, cmd, gpu_id, out_file)  # 非阻塞发送启动命令
        print(f"finished_proc: {finished_proc}")
        print(f"kill_proc: {kill_proc}")
        print(f"start_proc: {start_proc}")
        print(f"self.gpu_alloc: {self.gpu_alloc}")

    def run(self):
        while not self.all_complete():
            self.step()
            self.print_logs()
        if self.out_put is not None:
            self.output_logs()

    def output_logs(self):
        if not os.path.exists(os.path.dirname(self.out_put)):
            os.makedirs(os.path.dirname(self.out_put))
        with open(self.out_put, "w") as f:
            json.dump(self.logs, f)

    def print_logs(self):
        entry = {
            "timestamp": self.current_time,
            "num_nodes": self.num_nodes,
            "used_gpus": sum(map(len, self.running_allocations.values())),
            "allocations": self.running_allocations,
            "submitted_jobs": [
                {
                    "name": job.name,
                    "epoch": job.epoch,
                    # "progress": job.progress,
                    "num_restarts": job.num_restarts,
                    "allocation": self.running_allocations.get(job.name, []),
                    "placement": job.placement,
                    "batch_size": job.atomic_bsz * (job.accum_steps + 1) * sum(job.placement),
                    "accum_steps": job.accum_steps,
                    "submission_time": job.submission_time,
                    "completion_time": job.completion_time,
                    "grad_params": job.grad_params,
                    "attained_service": job.attained_service,
                }
                for job in self.jobs.values()
                if job.submission_time <= self.current_time
            ],
            "jct": {
                job.name: job.completion_time - job.submission_time
                for job in self.jobs.values() if job.completion_time is not None
            },
        }
        entry["avg_jct"] = sum(entry['jct'].values()) / len(entry['jct']) if entry['jct'] else 0
        self.logs.append(entry)
        print(f"---------------- SIMULATOR TIME: {self.current_time} ----------------")
        print("Active jobs:")
        for val in entry["submitted_jobs"]:
            if val["completion_time"] is None:
                print(f"\t{val['name']}:\t[epoch {val['epoch']}]\t[restarts {val['num_restarts']}]\t"
                      f"[batch size {val['batch_size']}]\t[placement {val['placement']}]")
        print(f"allocations: {entry['allocations']}")
        print(f"GPU utilization: {entry['used_gpus']}")
        print(f"Completed jobs [{len(entry['jct'])}]:")
        print(entry['jct'])
        print("Average JCT:", entry["avg_jct"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=str, default="./workload/workloads-4h-40j/workload-3.csv",
                        help="path to workload csv")
    parser.add_argument("--policy", type=str, default="ares", choices=get_all_policies(),
                        help="scheduler policy")
    parser.add_argument("--nodes", type=str,
                        default=" ".join([f"10.0.0.{i}" for i in range(19, 19 + 4)]),
                        help="list of node IPs")
    parser.add_argument("--interval", type=int, default=60,
                        help="scheduling interval in seconds")
    parser.add_argument("--num-gpus", type=int, default=4,
                        help="number of GPUs per node")
    parser.add_argument("--output", type=str, default=None,
                        help="output all logs to a json file")
    args = parser.parse_args()

    cluster = Cluster(args.workload, args.policy, args.nodes, args.num_gpus, args.interval, args.output)
    cluster.run()
