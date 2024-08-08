import collections
import copy

import math
from typing import Dict, Tuple, List

from policy.utils import JobInfo, NodeInfo


def predict_step_time(job, num_replicas):
    placement = ()
    while sum(placement) < num_replicas:
        placement = (*placement, min(num_replicas - sum(placement), 4))
    local_bsz = math.ceil(job.target_batch_size / num_replicas - 1e-8)  # gpu扩张会导致local_bsz减少而保持target_bsz不变
    accum_steps = math.ceil(local_bsz / job.application.max_local_bsz - 1e-8) - 1  # accum_steps表示额外的累积步数
    # if num_replicas == 1 and job.target_batch_size > job.application.init_batch_size:
    #     accum_steps = max(1, accum_steps)
    atomic_bsz = math.ceil(local_bsz / (accum_steps + 1) - 1e-8)
    count = num_replicas * (accum_steps + 1)
    atomic_bsz = min(atomic_bsz, int(job.application.max_batch_size / count))
    # throughput = job.speedup_fn._goodput_fn.throughput(len(placement), num_replicas, atomic_bsz, accum_steps)
    # return atomic_bsz * count / throughput
    step_time, sync_time = job.application.get_throughput(placement, atomic_bsz)  # step_time包含了sync_time
    return step_time + (step_time - sync_time) * accum_steps


class ARESPolicy(object):
    def __init__(self, time_fn, total_gpus, threshold=0.75):
        self._time_fn = time_fn
        self.gps_sys = GPSSystem(time_fn(), total_gpus)
        self.allocations = {}
        self.threshold = threshold

        # self.linear_gps_sys = LinearGPSSystem(total_gpus)

    def optimize(self,
                 jobs: Dict[Tuple[str, str], JobInfo],
                 nodes: Dict[str, NodeInfo],
                 prev_allocations: Dict[Tuple[str, str], List[str]],
                 node_template=None):
        prev_allocations = {k: v for k, v in prev_allocations.items() if k in jobs}
        # print(f">>> jobs: {jobs}")
        # print(f">>> nodes: {nodes}")
        # print(f">>> prev_allocations: {prev_allocations}")

        # 1. 检查是否有新到达或者结束的任务，如果没有则直接返回
        has_new_job = self.gps_sys.check_and_add_new_job(jobs)
        self.gps_sys.check_and_add_new_job(jobs)
        has_finished_job = (collections.Counter(sum(self.allocations.values(), []))
                            != collections.Counter(sum(prev_allocations.values(), [])))

        # 2. 更新理想公平参考系统
        self.gps_sys.do_forward(self._time_fn(), has_new_job)  # 理想公平调度器执行到当前时间，更新理想公平分配的各任务进度

        if not has_new_job and not has_finished_job:
            # print(f">>> no new job or finished job, return prev_allocations: {prev_allocations}")
            return prev_allocations, len(nodes)  # todo 如果吞吐量、完成时间变化，也需要调度

        # assert self.gps_sys.finished_job_order == self.linear_gps_sys.finished_job_order, \
        #     f"finished_job_order not equal: {self.gps_sys.finished_job_order}, {self.linear_gps_sys.finished_job_order}"

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


class GPSSystem:
    def __init__(self, cur_time, total_gpus):
        self.fair_jobs: dict = {}
        self.time = cur_time
        self.finished_job_order = []
        self.total_gpus = total_gpus
        self.has_finished_job = False

    def export_fair_state(self):
        """
        log task status under ideal fair distribution
        """
        makespan = (max(info["completion_time"] for info in self.fair_jobs.values()) + 60) // 60 * 60
        print(f"[GPSSystem]---------------- SIMULATOR TIME: {makespan} ----------------")
        jct_dict = {key: info["completion_time"] - info["arrival_time"] for key, info in self.fair_jobs.items()}
        print(f"[GPSSystem]{jct_dict}")
        print(f"[GPSSystem]Average JCT:{sum(jct_dict.values()) / len(jct_dict) if jct_dict else 0}")
        return

    def _get_step_time_with_fair_share(self, fair_job, fair_share):
        """
        assume jobs can receive divisible number of gpus, and each job receives N / M gpus
        """
        fair_share = min(fair_share, fair_job["max_replicas"])
        if int(fair_share) == 0:
            step_time = fair_job["step_time"][1] / fair_share
        elif fair_share != int(fair_share):
            ceil, floor = int(fair_share + 1), int(fair_share)
            step_time = ((fair_job["step_time"][ceil] - fair_job["step_time"][floor]) * (fair_share - floor)
                         + fair_job["step_time"][floor])
        else:
            step_time = fair_job["step_time"][int(fair_share)]
        return step_time

    def _add_job(self, key, job: JobInfo):
        # get progress/iteration of job
        completion_progress = job.application.get_progress(job.application.max_epochs)
        scale = job.target_batch_size / job.application.init_batch_size
        completion_iter = completion_progress / scale
        # print(f"{key}'s completion_iter: {completion_iter}")
        # completion_epoch = job.application.get_completion_epoch(job.target_batch_size)
        # completion_iter = job.application.get_iteration(job.target_batch_size, completion_epoch)
        # print(f"{key}'s completion_iter: {completion_iter}")

        # get throughput model
        # max_replicas = job.max_replicas
        max_replicas = job.application.max_num_replicas
        # print(f"job.max_replicas: {job.max_replicas}, max_replicas: {max_replicas}")
        step_time = {}
        for i in range(1, max_replicas + 1):
            step_time[i] = predict_step_time(job, i)

        # get non-linear scale curve
        scale_factor = []
        for i, x in enumerate([i for i in [1, 2, 4, 8, 16, 32, 64] if i <= max_replicas]):
            tp1 = 1 / step_time[1]  # iter/s
            tpx = 1 / step_time[x]  # iter/s
            speedup = tpx / tp1
            efficiency = tpx / tp1 / x
            d_speedup = speedup - (scale_factor[-1][2] if scale_factor else 0)
            scale_factor.append([x, efficiency, speedup, d_speedup])
        # print(f">>> scale_factor of {key}: {scale_factor}")

        self.fair_jobs[key] = {
            "arrival_time": self.time,  # job.creation_timestamp
            "remaining_iter": completion_iter,
            "completion_time": None,
            "step_time": step_time,
            "max_replicas": max_replicas,
            "scale_factor": scale_factor,
        }

    def do_forward(self, cur_time, has_new_job):
        self.time += 30 if has_new_job or self.has_finished_job else 0
        self.has_finished_job = False
        run_time = cur_time - self.time
        active_jobs = {k: j for k, j in self.fair_jobs.items() if j["remaining_iter"] > 0}
        if len(active_jobs) == 0:
            return

        fair_share = self.total_gpus / len(active_jobs)
        for key, job in active_jobs.items():
            step_time = self._get_step_time_with_fair_share(job, fair_share)
            finished_iter = run_time / step_time
            virtual_finish_time = job["remaining_iter"] * step_time
            job["remaining_iter"] = max(job["remaining_iter"] - finished_iter, 0)
            if job["remaining_iter"] == 0:
                self.fair_jobs[key]["completion_time"] = self.time + virtual_finish_time
                self.has_finished_job = True
        self.time += run_time

    def _sim_forward(self):
        active_jobs = {k: copy.deepcopy(j) for k, j in self.fair_jobs.items() if j["remaining_iter"] > 0}
        virtual_time = self.time
        self.finished_job_order = [k for k, j in self.fair_jobs.items() if j["remaining_iter"] <= 0]
        self.finished_job_order = sorted(self.finished_job_order, key=lambda k: self.fair_jobs[k]["completion_time"])
        while active_jobs:
            fair_share = self.total_gpus / len(active_jobs)
            virtual_skip_time = 1 << 32
            for key, job in active_jobs.items():
                step_time = self._get_step_time_with_fair_share(job, fair_share)
                v_finish_time = job["remaining_iter"] * step_time
                virtual_skip_time = min(virtual_skip_time, v_finish_time)

            virtual_skip_time += 30
            virtual_skip_time = (virtual_skip_time // 60 + 1) * 60
            # print(f">>> virtual_time: {virtual_time}, virtual_skip_time: {virtual_skip_time}")
            to_del_key = []
            for key, job in active_jobs.items():
                step_time = self._get_step_time_with_fair_share(job, fair_share)
                finished_iter = (virtual_skip_time - 30) / step_time
                v_finish_time = job["remaining_iter"] * step_time
                job["remaining_iter"] = max(job["remaining_iter"] - finished_iter, 0)

                if job["remaining_iter"] == 0:
                    self.fair_jobs[key]["completion_time"] = virtual_time + v_finish_time
                    # print(f"append {key} to finished_job_order")
                    self.finished_job_order.append(key)
                    to_del_key.append(key)
            virtual_time = virtual_time + virtual_skip_time
            for key in to_del_key:
                del active_jobs[key]

        print(f">>> finished_job_order: {self.finished_job_order}")

    def check_and_add_new_job(self, jobs):
        # 检查并新增任务
        has_new_job = False
        for key, job in jobs.items():
            if key not in self.fair_jobs.keys():
                has_new_job = True
                self._add_job(key, job)
        if has_new_job:
            # 计算理想公平分配下的完成时间, 按照完成时间排序
            self._sim_forward()
        return has_new_job


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
