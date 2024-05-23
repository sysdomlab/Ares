#!/bin/bash

python3=/home/cchen/miniconda3/envs/yfliu/bin/python3
work_dir=/home/cchen/yfliu/cluster_schedule/pollux/testbed
log_dir=/home/cchen/yfliu/ares/logs

nohup $python3 -u scheduler.py > /home/cchen/yfliu/ares/logs/scheduler.log 2>&1 &
