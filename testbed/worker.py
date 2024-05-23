import argparse
import os
import signal
import time

import gevent
import zerorpc
import subprocess


class Worker(object):
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
        cmd = cmd if isinstance(cmd, list) else cmd.split()
        gpu_id = str(gpu_id)
        if gpu_id not in self.gpu_alloc:
            raise ValueError(f"GPU {gpu_id} is not available.")
        out_dir = os.path.dirname(out_file)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        print(f"Running proc {proc_name} on GPU {gpu_id}")
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

        if proc["proc"].poll() is not None:
            print(f"Proc {proc_name} on GPU {proc['gpu_id']} has already finished.")
            del self.registered_processes[proc_name]
            self.gpu_alloc[proc["gpu_id"]].remove(proc_name)

            return 0, f"Proc {proc_name} on GPU {proc['gpu_id']} has already finished."

        print(f"Killing proc {proc_name} on GPU {proc['gpu_id']}")
        if proc["proc"].poll() is None:
            proc["proc"].terminate()
            # proc["proc"].wait()

        return proc["proc"].poll(), f"Killed proc {proc_name} on GPU {proc['gpu_id']}"

    def stats_proc(self, proc_name):
        """
        Get the status of a process on this machine.

        :param proc_name: format as f"{job_name}:{rank}"
        :return: a dictionary containing the status of the process.
        """
        print(f"stats_proc {proc_name}")
        if proc_name in self.registered_processes:
            proc = self.registered_processes[proc_name]
            res = {
                "proc_name": proc_name,
                "cmd": proc["cmd"],
                "gpu_id": proc["gpu_id"],
                "out_file": proc["out_file"],
                "pid": proc["proc"].pid,
                "returncode": proc["proc"].poll(),
                "worker": self.ip_address
            }
            if proc["proc"].poll() is not None:
                print(f"Proc {proc_name} on GPU {proc['gpu_id']} finished.")
                del self.registered_processes[proc_name]
                self.gpu_alloc[proc["gpu_id"]].remove(proc_name)
            return res
        print(f"Proc {proc_name} not found.")
        return None

    def get_gpu_alloc(self):
        """
        Get the GPU allocation status.

        :return: a dictionary containing the GPU allocation status.
        """
        return self.gpu_alloc

    def cleanup(self):
        """
        cleanup all the registered processes and release the GPUs.
        """
        print("Cleaning up all the registered processes and release the GPUs.")
        for proc_name, proc in self.registered_processes.items():
            if proc["proc"].poll() is None:
                print(f"terminate proc {proc_name} on GPU {proc['gpu_id']}")
                proc["proc"].terminate()
        for proc_name, proc in self.registered_processes.items():
            timeout, start_time = 10, time.time()
            while proc["proc"].poll() is None:
                time.sleep(1)
                if time.time() - start_time > timeout:
                    print(f"kill proc {proc_name} forcefully")
                    proc["proc"].kill()
                    break

        self.registered_processes = {}
        self.gpu_alloc = {str(i): [] for i in range(len(self.gpu_alloc))}


if __name__ == '__main__':
    # python3 worker.py --local_ip 10.0.0.19
    # python3 worker.py --local_ip 10.0.0.20
    # nohup python3 worker.py --local_ip 10.0.0.20 > ./worker.log 2>&1 &
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_num', type=int, default=4)
    parser.add_argument('--local_ip', type=str)
    parser.add_argument('--local_port', type=str, default="4242")
    args = parser.parse_args()

    print(f"start the RPC server on {args.local_ip}:{args.local_port}")
    core = Worker(gpu_num=args.gpu_num, ip_address=args.local_ip)
    server = zerorpc.Server(core)
    server.bind(f"tcp://0.0.0.0:{args.local_port}")
    gevent.signal_handler(signal.SIGINT, server.stop)
    gevent.signal_handler(signal.SIGTERM, server.stop)
    server.run()
    core.cleanup()
