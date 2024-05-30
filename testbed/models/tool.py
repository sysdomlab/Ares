import functools
import os
import random
import time

import torch.distributed as dist
from typing import List

import numpy as np
import torch
import torch.cuda
from torch.autograd import Variable
from torch.nn.parallel import DistributedDataParallel

from models.env import get_python3_path
from policy.applications import memoize, APPLICATIONS


def set_random_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


received_signal = False


def signal_handler(sig, frame):
    global received_signal
    print('Received signal {}. Setting received_signal flag...'.format(sig))
    received_signal = True


def get_signal_received():
    global received_signal
    return received_signal


class Statistics:
    def __init__(self, metrics_names: List[str], device, acc_steps):
        self.metric_names = metrics_names
        self.device = device
        self.global_value = torch.tensor([0] * len(metrics_names), dtype=torch.float64, device=device)
        self.iteration = 0
        self.local_value = torch.tensor([0] * len(metrics_names), dtype=torch.float64, device=device)
        self.acc_num = 0
        self.acc_steps = acc_steps
        self.current_batch_value = self.local_value.clone()

    def accumulate_in_batch(self, values: List[float]):
        self.local_value += torch.tensor(values, dtype=torch.float64, device=self.device)
        self.acc_num += 1

    def update_global(self):
        if self.acc_num < self.acc_steps:
            print("warn: help to update local value")
            self.local_value = self.local_value / self.acc_num * self.acc_steps
        self.acc_num = 0

        dist.barrier()
        dist.all_reduce(self.local_value, op=dist.ReduceOp.SUM)
        self.local_value /= dist.get_world_size()
        # dist.all_reduce(self.local_value, op=dist.ReduceOp.MAX)

        self.global_value = (self.global_value * self.iteration + self.local_value) / (self.iteration + 1)
        self.iteration += 1
        self.current_batch_value = self.local_value.clone()
        self.local_value.zero_()

    def get_data(self):
        return {name: (value1, value2) for name, value1, value2
                in zip(self.metric_names, self.global_value.tolist(), self.current_batch_value.tolist())}

    def reset(self):
        self.iteration = 0
        self.global_value.zero_()
        self.local_value.zero_()
        self.acc_num = 0
        self.current_batch_value.zero_()

    def __str__(self):
        return '\t' + '\t'.join([f'{name}={value[1]:.4f}({value[0]:.4f})' for name, value in self.get_data().items()])


candidate_port = 17000


@memoize
def get_port(job_name):
    global candidate_port
    candidate_port += 1
    return candidate_port


def get_cmd(job_name, allocation, rank, acc_bsz, acc_step, model_name=None):
    master_address = min(allocation)
    world_size = len(allocation)
    model_name = job_name.split('-')[0] if model_name is None else model_name
    python3 = get_python3_path()
    master_port = get_port(job_name)
    max_epoch = APPLICATIONS[model_name].max_epochs
    return " ".join([
        python3, "-u", "framework.py",
        "--job_name", job_name,
        "--master_address", master_address,
        "--master_port", str(master_port),
        "--world_size", str(world_size),
        "--global_rank", str(rank),
        "--acc_bsz", str(acc_bsz),
        "--acc_step", str(acc_step),
        "--model_name", model_name,
        "--max_epoch", str(max_epoch),
    ])


class AresDataParallel(DistributedDataParallel):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        for param in model.parameters():
            param.register_hook(functools.partial(self._backward_hook, param))

        self._sync_start = None
        self.sync_time = None

    def _backward_hook(self, param, grad):
        if grad.device.type.startswith("cuda"):
            self._sync_start = torch.cuda.Event(enable_timing=True)
            self._sync_start.record()
        else:
            self._sync_start = time.time()
        self._final_callback_queued = False
        Variable._execution_engine.queue_callback(self._queue_callback)

        if self.sync_time is not None:
            # print("WARNING: Gradient synchronization time already recorded. This should not happen.")
            self.sync_time = None

    def _queue_callback(self):
        if self._final_callback_queued:
            return
        self._final_callback_queued = True
        Variable._execution_engine.queue_callback(self._final_callback)

    def _final_callback(self):
        self._final_callback_queued = False
        if isinstance(self._sync_start, torch.cuda.Event):
            sync_end = torch.cuda.Event(enable_timing=True)
            sync_end.record()
            sync_end.synchronize()
            self.sync_time = self._sync_start.elapsed_time(sync_end) / 1e3
        else:
            self.sync_time = time.time() - self._sync_start

        # print(f"_final_callback: {self.sync_time}")

    def get_sync_time(self):
        """
        Returns the time it took to synchronize gradients in the last backward pass.
        sync_time will be generated only after the backward pass is complete.
        """
        sync_time = self.sync_time
        self._sync_start = None
        self.sync_time = None
        return sync_time or 0
