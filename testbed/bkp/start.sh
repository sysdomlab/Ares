python3=/home/cchen/miniconda3/envs/yfliu/bin/python3

#$python3 framework.py --master_address 10.0.0.20 --job_name cifar10-0 --world_size 1 --global_rank 0 --device 0 --acc_bsz 256 --acc_step 1
#export CUDA_VISIBLE_DEVICES=0 && nohup $python3 framework.py --master_address 10.0.0.20 --job_name cifar10-0 --world_size 4 --global_rank 0 --device 0 --acc_bsz 64 --acc_step 1 > log0.txt &
#export CUDA_VISIBLE_DEVICES=1 && nohup $python3 framework.py --master_address 10.0.0.20 --job_name cifar10-0 --world_size 4 --global_rank 1 --device 0 --acc_bsz 64 --acc_step 1 > log1.txt &
#export CUDA_VISIBLE_DEVICES=2 && nohup $python3 framework.py --master_address 10.0.0.20 --job_name cifar10-0 --world_size 4 --global_rank 2 --device 0 --acc_bsz 64 --acc_step 1 > log2.txt &
#export CUDA_VISIBLE_DEVICES=3 && nohup $python3 framework.py --master_address 10.0.0.20 --job_name cifar10-0 --world_size 4 --global_rank 3 --device 0 --acc_bsz 64 --acc_step 1 > log3.txt &

#$python3 framework.py --master_address 10.0.0.20 --job_name imagenet-0 --world_size 1 --global_rank 0 --device 0 --acc_bsz 32 --acc_step 1 --model_name imagenet
#export CUDA_VISIBLE_DEVICES=0 && nohup $python3 framework.py --master_address 10.0.0.20 --job_name imagenet-0 --world_size 4 --global_rank 0 --device 0 --acc_bsz 32 --acc_step 1 --model_name imagenet > log0.txt &
#export CUDA_VISIBLE_DEVICES=1 && nohup $python3 framework.py --master_address 10.0.0.20 --job_name imagenet-0 --world_size 4 --global_rank 1 --device 0 --acc_bsz 32 --acc_step 1 --model_name imagenet > log1.txt &
#export CUDA_VISIBLE_DEVICES=2 && nohup $python3 framework.py --master_address 10.0.0.20 --job_name imagenet-0 --world_size 4 --global_rank 2 --device 0 --acc_bsz 32 --acc_step 1 --model_name imagenet > log2.txt &
#export CUDA_VISIBLE_DEVICES=3 && nohup $python3 framework.py --master_address 10.0.0.20 --job_name imagenet-0 --world_size 4 --global_rank 3 --device 0 --acc_bsz 32 --acc_step 1 --model_name imagenet > log3.txt &

#$python3 framework.py --master_address 10.0.0.20 --job_name yolov3-0 --world_size 1 --global_rank 0 --device 0 --acc_bsz 16 --acc_step 1 --model_name yolov3


#$python3 framework.py --master_address 10.0.0.20 --job_name ncf-0 --world_size 1 --global_rank 0 --device 0 --acc_bsz 32768 --acc_step 1 --model_name ncf


#$python3 framework.py --master_address 10.0.0.20 --job_name deepspeech2-0 --world_size 1 --global_rank 0 --device 0 --acc_bsz 32 --acc_step 1 --model_name deepspeech2


$python3 framework.py --master_address 10.0.0.20 --job_name bert-0 --world_size 1 --global_rank 0 --device 0 --acc_bsz 16 --acc_step 1 --model_name bert
