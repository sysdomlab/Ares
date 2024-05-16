import json
import time
from typing import List, Tuple, Dict

import yaml
import zerorpc

from models.env import get_python3_path, get_checkpoint_path


def toy_subprocess():
    import subprocess
    import time

    python3 = "/home/cchen/miniconda3/envs/yfliu/bin/python3"
    cmd = (
        f"{python3} framework.py --master_address 10.0.0.20 --job_name bert-0 --world_size 1 --global_rank 0 --device 0 "
        f"--acc_bsz 16 --acc_step 5 --model_name bert"
        # f" 2>&1 >> out0.log"
    )

    with open("out0.log", "w") as f:
        proc = subprocess.Popen(cmd.split(),
                                env={"CUDA_VISIBLE_DEVICES": "1"},
                                stdout=f,
                                stderr=f)

    # proc.wait()
    print(f"proc.args: {proc.args}")
    print(f"proc.pid: {proc.pid}")
    print(f"proc.returncode: {proc.returncode}")
    # print(proc.stdout)
    print(f"proc.poll(): {proc.poll()}")

    time.sleep(60)

    print("Killing process...")
    proc.terminate()
    proc.wait()
    print(f"proc.returncode: {proc.returncode}")

    for i in range(100):
        print(f"proc.poll(): {proc.poll()}")
        # print(f"proc.returncode: {proc.returncode}")
        time.sleep(1)


class ToyClient:
    """
    A client to manage all workers' connections and communicate with them in batch.
    """

    def __init__(self, workers):
        self.connects = {}
        for worker in workers:
            self.connects[worker] = zerorpc.Client()
            self.connects[worker].connect(f"tcp://{worker}:4242")

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
        job_name = None
        # try:
        for cmd in all_commands:
            worker = cmd['worker']
            job_name = cmd['job_name']
            args = cmd['args']
            self.connects[worker].run_proc(args['proc_name'], args['cmd'], args['gpu_id'], args['out_file'])
        # except Exception as e:
        #     print(f"Error: {e}")
        #     self.kill_job(job_name)

    def job_stats(self, job_name):
        res = []
        for worker, conn in self.connects.items():
            res += conn.job_stats(job_name)
        return res

    def kill_job(self, job_name):
        for worker, conn in self.connects.items():
            conn.kill_job(job_name)

    def get_gpu_alloc(self):
        return {
            worker: conn.get_gpu_alloc()
            for worker, conn in self.connects.items()
        }

    def close(self):
        for worker, conn in self.connects.items():
            conn.close()


def get_cmd(job_key: Tuple[str, str], allocation: List[str]):
    unique_workers = set(allocation)
    used_gpus = {wk: 0 for wk in unique_workers}

    def get_device(worker):
        res = used_gpus[worker]
        used_gpus[worker] += 1
        return res

    all_commands = []
    job_name = job_key[1]
    restart = 0  # todo

    for rank, wk in enumerate(sorted(allocation)):
        proc_name = f"{job_name}:{rank}"
        cmd = [  # todo template
            get_python3_path(), "framework.py",
            "--job_name", job_name,
            "--master_address", min(allocation),
            "--master_port", "17001",
            "--world_size", str(len(allocation)),
            "--global_rank", str(rank),
            "--acc_bsz", "16",
            "--acc_step", "5",
            "--model_name", "bert",
            "--max_epoch", "10",
        ]
        cmd = " ".join(cmd)
        gpu_id = get_device(wk)
        out_file = f"{get_checkpoint_path(job_name, return_dir=True)}/restart_{restart}_rank_{rank}.log"
        all_commands.append({
            "worker": wk,
            "job_name": job_name,
            "args": {
                "proc_name": proc_name,
                "cmd": cmd,
                "gpu_id": gpu_id,
                "out_file": out_file,
            }
        })

    return all_commands


if __name__ == '__main__':
    # toy_subprocess()

    cmds = get_cmd(job_key=('default', 'bert-1'), allocation=['10.0.0.19', '10.0.0.19', '10.0.0.20', '10.0.0.20'])
    # cmds = get_cmd(job_key=('default', 'bert-1'), allocation=['10.0.0.19', '10.0.0.19', '10.0.0.19', '10.0.0.19'])
    # cmd_yaml = yaml.dump(cmds)
    # print(cmd_yaml)
    print(cmds)
    # input()

    c = ToyClient(["10.0.0.19", "10.0.0.20"])

    # c.run_job(cmds)
    c.kill_job("bert-1")

    for i in range(120):
        print(c.get_gpu_alloc())
        print(c.job_stats("bert-1"))
        time.sleep(1)
