import zerorpc

from models.env import get_checkpoint_path
from models.tool import get_cmd

available_node = ["10.0.0.19"]
job_name = "imagenet-6-seed0"
app_name = "imagenet"
num_replica = 1
bsz = 6400
lc_bsz = bsz // num_replica

connects = {}
for node_ip in available_node:
    if node_ip is not None:
        connects[node_ip] = zerorpc.Client()
        connects[node_ip].connect(f"tcp://{node_ip}:4242")

allocation_ng = [("10.0.0.19", 0)]
allocation_n = ["10.0.0.19"]
# print(allocation_ng)
for rank, (node_ip, gpu_id) in enumerate(allocation_ng):
    proc_name = f"{job_name}:{rank}"
    cmd = get_cmd(job_name=job_name, allocation=allocation_n, rank=rank,
                  acc_bsz=75, acc_step=86, model_name=app_name, master_port=13330 + gpu_id)
    out_file = f"{get_checkpoint_path(job_name, return_dir=True)}/rank_{rank}.log"
    # res = connects[node_ip].stats_proc(proc_name)
    print(f"connects[{node_ip}].run_proc({proc_name}, {cmd}, {gpu_id}, {out_file})")
    connects[node_ip].run_proc(proc_name, cmd, gpu_id, out_file)
