import argparse
import json
import os

import zerorpc
import subprocess


class WorkerRPCServer(object):
    """
    RPC server for worker.

    This server is only responsible for running and killing processes on this machine.
    It also provides a way to get the status of processes and the GPU allocation status.

    :param gpu_num: number of GPUs available on this machine.
    :param ip_address: IP address of this machine.
    """

    def __init__(self, gpu_num=4, ip_address="0.0.0.0"):
        self.ip_address = ip_address
        self.registered_processes = {}
        self.gpu_alloc = {str(i): [] for i in range(gpu_num)}

    def run_proc(self, proc_name: str, cmd: [str, list], gpu_id: [int, str], out_file: str):
        """
        Run a process on this machine.

        note:
        - no need to redirect stdout and stderr to file, since we can use subprocess.Popen to capture the output.

        :param proc_name: format as f"{job_name}:{rank}"
        :param cmd: full command to run.
        :param gpu_id: device to use.
        :param out_file: output file path.
        """
        # print(f"get run_proc args: {proc_name} {cmd} {gpu_id} {out_file}")

        if proc_name in self.registered_processes:
            raise ValueError(f"Proc {proc_name} is already running.")
        gpu_id = str(gpu_id)
        if gpu_id not in self.gpu_alloc:
            raise ValueError(f"GPU {gpu_id} is not available.")
        cmd = cmd if isinstance(cmd, list) else cmd.split()

        print(f"Running proc {proc_name} on GPU {gpu_id}")
        out_dir = os.path.dirname(out_file)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        with open(out_file, "w") as f:
            proc = subprocess.Popen(cmd,
                                    env={"CUDA_VISIBLE_DEVICES": gpu_id},
                                    stdout=f,
                                    stderr=f)

        self.registered_processes[proc_name] = {
            "proc": proc,
            "cmd": cmd,
            "gpu_id": gpu_id,
            "out_file": out_file
        }
        self.gpu_alloc[gpu_id].append(proc_name)
        return True, f"Running proc {proc_name} on GPU {gpu_id}"

    def kill_proc(self, proc_name):
        """
        Kill a process on this machine.

        :param proc_name: format as f"{job_name}:{rank}"
        """
        if proc_name not in self.registered_processes:
            return 0, f"process don't exist"
        proc = self.registered_processes[proc_name]

        if proc["proc"].returncode is not None:
            print(f"Proc {proc_name} on GPU {proc['gpu_id']} has already finished.")
            del self.registered_processes[proc_name]
            self.gpu_alloc[proc["gpu_id"]].remove(proc_name)

            return 0, f"Proc {proc_name} on GPU {proc['gpu_id']} has already finished."

        print(f"Killing proc {proc_name} on GPU {proc['gpu_id']}")
        if proc["proc"].returncode is None:
            proc["proc"].terminate()
            # proc["proc"].wait()

        return proc["proc"].returncode, f"Killed proc {proc_name} on GPU {proc['gpu_id']}"

    def kill_job(self, job_name):
        """
        Kill all processes of a job on this machine.

        :param job_name: format as f"{job_name}*"
        """
        procs_to_kill = [k for k in self.registered_processes.keys() if k.startswith(job_name)]
        for proc_name in procs_to_kill:
            self.kill_proc(proc_name)

    def proc_stats(self, proc_name):
        """
        Get the status of a process on this machine.

        :param proc_name: format as f"{job_name}:{rank}"
        :return: a dictionary containing the status of the process.
        """
        if proc_name in self.registered_processes:
            proc = self.registered_processes[proc_name]
            res = {
                "proc_name": proc_name,
                "cmd": proc["cmd"],
                "gpu_id": proc["gpu_id"],
                "out_file": proc["out_file"],
                "pid": proc["proc"].pid,
                "returncode": proc["proc"].returncode,
                "worker": self.ip_address
            }
            if proc["proc"].poll() is not None:
                print(f"Proc {proc_name} on GPU {proc['gpu_id']} has already finished.")
                del self.registered_processes[proc_name]
                self.gpu_alloc[proc["gpu_id"]].remove(proc_name)
            return res
        return None

    def job_stats(self, job_name):
        """
        Get the status of all processes of a job on this machine.

        :param job_name: format as f"{job_name}*"
        :return: a list of dictionaries containing the status of all processes of the job.
        """
        procs_to_stats = [k for k in self.registered_processes.keys() if k.startswith(job_name)]
        return [self.proc_stats(proc_name) for proc_name in procs_to_stats]

    def get_gpu_alloc(self):
        """
        Get the GPU allocation status.

        :return: a dictionary containing the GPU allocation status.
        """
        return self.gpu_alloc


if __name__ == '__main__':
    # python3 worker.py --local_ip 10.0.0.19 --manager_ip 10.0.0.19
    # python3 worker.py --local_ip 10.0.0.20 --manager_ip 10.0.0.19
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_num', type=int, default=4)
    parser.add_argument('--local_ip', type=str)
    parser.add_argument('--local_port', type=str, default="4242")
    parser.add_argument('--manager_ip', type=str)
    args = parser.parse_args()

    print(f"start the RPC server on {args.local_ip}:{args.local_port}")
    s = zerorpc.Server(WorkerRPCServer(gpu_num=args.gpu_num, ip_address=args.local_ip))
    s.bind(f"tcp://0.0.0.0:{args.local_port}")
    s.run()
