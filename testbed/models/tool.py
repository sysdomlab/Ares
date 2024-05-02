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
    def __init__(self, metrics: List[str], device):
        self.device = device
        self.iteration = 0
        self.metric_names = metrics
        self.metric_values = torch.tensor([0] * len(metrics), dtype=torch.float64, device=device)
        self.metric_batch = torch.tensor([0] * len(metrics), dtype=torch.float64, device=device)
        self.metric_current = None

    def accumulate_in_batch(self, values: List[float]):
        self.metric_batch += torch.tensor(values, dtype=torch.float64, device=self.device)

    def update_local(self):
        self.metric_values = (self.metric_values * self.iteration + self.metric_batch) / (self.iteration + 1)
        self.metric_current = self.metric_batch.clone()
        self.metric_batch.zero_()
        self.iteration += 1

    def synchronize(self):
        dist.barrier()
        dist.all_reduce(self.metric_values, op=dist.ReduceOp.SUM)
        self.metric_values /= dist.get_world_size()

    def get_data(self):
        return {name: (value1, value2) for name, value1, value2
                in zip(self.metric_names, self.metric_values.tolist(), self.metric_current.tolist())}

    def reset(self):
        self.iteration = 0
        self.metric_values.zero_()
        self.metric_batch.zero_()
        self.metric_current = None

    def __str__(self):
        return '\t' + '\t'.join([f'{name}={value[1]:.4f}({value[0]:.4f})' for name, value in self.get_data().items()])


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
            print("WARNING: Gradient synchronization time already recorded. This should not happen.")
        self.sync_time = 0

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

    def get_sync_time(self):
        sync_time = self.sync_time
        self._sync_start = None
        self.sync_time = None
        return sync_time
