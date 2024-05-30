# Ares: Fair and Efficient Scheduling of Deep Learning Training Jobs with Elastic Fair Queuing

## Start a deep learning training job

### prepare datasets

1. cifar10 
2. imagenet 
3. yolov3 
4. ncf 
5. bert 
6. deepspeech2 

### start training a job

```bash
# Modify testbed/models/env.py to set the paths of datasets and the locations where Ares saves the job statuses and checkpoints.

python3 framework.py --master_address {ip} --job_name bert-0 --world_size 1 --global_rank 0 --device 0 --acc_bsz 16 --acc_step 1 --model_name bert

or

export CUDA_VISIBLE_DEVICES=0 && nohup $python3 framework.py --master_address 10.0.0.20 --job_name imagenet-0 --world_size 4 --global_rank 0 --device 0 --acc_bsz 32 --acc_step 1 --model_name imagenet > log0.txt &
export CUDA_VISIBLE_DEVICES=1 && nohup $python3 framework.py --master_address 10.0.0.20 --job_name imagenet-0 --world_size 4 --global_rank 1 --device 0 --acc_bsz 32 --acc_step 1 --model_name imagenet > log1.txt &
export CUDA_VISIBLE_DEVICES=2 && nohup $python3 framework.py --master_address 10.0.0.20 --job_name imagenet-0 --world_size 4 --global_rank 2 --device 0 --acc_bsz 32 --acc_step 1 --model_name imagenet > log2.txt &
export CUDA_VISIBLE_DEVICES=3 && nohup $python3 framework.py --master_address 10.0.0.20 --job_name imagenet-0 --world_size 4 --global_rank 3 --device 0 --acc_bsz 32 --acc_step 1 --model_name imagenet > log3.txt &
```

## Start a scheduling in the sharing cluster

```bash
# modify testbed/start_workers.sh and testbed/clean_workers.sh to set the machines you have
bash start_workers.sh  # start the workers

nohup python3 -u scheduler.py --policy pollux --nodes "10.0.0.22 10.0.0.23 10.0.0.24 10.0.0.25" > /home/cchen/yfliu/ares/logs/scheduler.log 2>&1 &

bash clean_workers.sh  # clean the workers
```
