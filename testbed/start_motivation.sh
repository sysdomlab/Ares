#!/bin/bash

policy=fifo

python3=/home/cchen/miniconda3/envs/yfliu/bin/python3
workload_path=./workload/workload-motivation.csv

nodes="10.0.0.11 10.0.0.13 10.0.0.15 10.0.0.16 10.0.0.17 10.0.0.20 10.0.0.21 10.0.0.22 10.0.0.23 10.0.0.24 10.0.0.25 10.0.0.26 10.0.0.19"
namespace="motivation"
log_path=/home/cchen/yfliu/ares/logs/$namespace.log
json_file=/home/cchen/yfliu/ares/logs/$namespace.json
nohup $python3 -u scheduler.py \
  --workload "$workload_path" \
  --policy "$policy" \
  --nodes "$nodes" \
  --output "$json_file"  > "$log_path" 2>&1 &
