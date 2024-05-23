import time
import zerorpc

from typing import List, Tuple, Dict

from models.env import get_python3_path, get_checkpoint_path
from policy.applications import APPLICATIONS, memoize


class Controller:
    """
    A client to manage all workers' connections and communicate with them in batch.
    responsible for all processes' status
    """

    def __init__(self, workers):
        # connection to all workers
        self.connects = {}
        for worker in workers:
            self.connects[worker] = zerorpc.Client()
            self.connects[worker].connect(f"tcp://{worker}:4242")

    def check_proc_health(self, proc_name):
        """
        Check if a process is still running.
        """
        pass

    def update_job_status(self, job_name):
        pass

    def run_job(self, all_commands: List[Dict]):
        """
        all_commands = [
            {
                'worker': '10.0.0.19',
                'job_name': 'bert-1',
                'args': {
                    'proc_name': 'bert-1:0',
                    'cmd': '/home/cchen/miniconda3/envs/yfliu/bin/python3 framework.py ...',
                    'gpu_id': 0,
                    'out_file': '/home/cchen/yfliu/ares/checkpoints/bert-1/restart_0_rank_0.log'
                }
            },
        ]
        """
        # try:
        for cmd in all_commands:
            worker = cmd['worker']
            args = cmd['args']
            self.connects[worker].run_proc(args['proc_name'], args['cmd'], args['gpu_id'], args['out_file'])
        # except Exception as e:
        #     print(f"Error: {e}")
        #     self.kill_job(job_name)

    def job_stats(self, job_name):
        """
        [
            {
                'proc_name': 'bert-1:0',
                'cmd': [...],
                'gpu_id': '0',
                'out_file': '/home/cchen/yfliu/ares/checkpoints/bert-1/restart_0_rank_0.log',
                'pid': 922097,
                'returncode': None,
                'worker': '10.0.0.19'
            },
            {
                'proc_name': 'bert-1:1',
                'cmd': [...],
                'gpu_id': '1',
                'out_file': '/home/cchen/yfliu/ares/checkpoints/bert-1/restart_0_rank_1.log',
                'pid': 922098,
                'returncode': None,
                'worker': '10.0.0.19'
            }
        ]
        """
        res = []
        for worker, conn in self.connects.items():
            res += conn.job_stats(job_name)
        return res

    def kill_job(self, job_name):
        for worker, conn in self.connects.items():
            conn.kill_job(job_name)

    def get_gpu_alloc(self):
        """
        res = {
            '10.0.0.19': {
                '0': ['bert-1:0'],
                '1': ['bert-1:1'],
                '2': [],
                '3': []
            }
        }
        """
        return {
            worker: conn.get_gpu_alloc()
            for worker, conn in self.connects.items()
        }


candidate_port = 17000


@memoize
def get_port(job_name):
    global candidate_port
    candidate_port += 1
    return candidate_port


def get_cmd(job_name, allocation, rank, acc_bsz, acc_step):
    master_address = min(allocation)
    world_size = len(allocation)
    model_name = job_name.split('-')[0]
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


def get_args(job_key: Tuple[str, str], allocation: List[str]):
    unique_workers = set(allocation)
    used_gpus = {wk: 0 for wk in unique_workers}

    def get_device(worker):
        res = used_gpus[worker]
        used_gpus[worker] += 1
        return res

    all_commands = []
    job_name = job_key[1]
    model_name = job_name.split('-')[0]
    restart = 0  # todo

    for rank, wk in enumerate(sorted(allocation)):
        proc_name = f"{job_name}:{rank}"
        cmd = get_cmd(job_name=job_name, allocation=allocation, rank=rank, acc_bsz=16, acc_step=5)
        gpu_id = get_device(wk)
        out_file = f"{get_checkpoint_path(job_name, return_dir=True)}/restart_{restart}_rank_{rank}.log"
        all_commands.append({
            "worker": wk,
            "args": {
                "proc_name": proc_name,
                "cmd": cmd,
                "gpu_id": gpu_id,
                "out_file": out_file,
            }
        })

    return all_commands


if __name__ == '__main__':
    job_allocations = {
        ('default', 'bert-1'): ['10.0.0.19', '10.0.0.19', '10.0.0.20', '10.0.0.20'],
        ('default', 'bert-2'): ['10.0.0.19', '10.0.0.19', '10.0.0.20', '10.0.0.20'],
    }
    proc_allocations = {
        ('default', 'bert-1'): {
            'bert-1:0': '10.0.0.19:0',
            'bert-1:1': '10.0.0.19:1',
            'bert-1:2': '10.0.0.20:0',
            'bert-1:3': '10.0.0.20:1',
        },
        ('default', 'bert-2'): {
            'bert-2:0': '10.0.0.19:2',
            'bert-2:1': '10.0.0.19:3',
            'bert-2:2': '10.0.0.20:2',
            'bert-2:3': '10.0.0.20:3',
        },
    }

    cmds = get_args(job_key=('default', 'bert-1'), allocation=['10.0.0.19', '10.0.0.19', '10.0.0.20', '10.0.0.20'])
    cmds = get_args(job_key=('default', 'bert-1'), allocation=['10.0.0.19', '10.0.0.19'])
    # cmd_yaml = yaml.dump(cmds)
    # print(cmd_yaml)
    print(cmds)
    # input()

    # c = ToyClient(["10.0.0.19", "10.0.0.20"])
    c = Controller(["10.0.0.19"])

    c.run_job(cmds)
    # c.kill_job("bert-1")

    for i in range(120):
        print(c.get_gpu_alloc())
        print(c.job_stats("bert-1"))
        time.sleep(1)
