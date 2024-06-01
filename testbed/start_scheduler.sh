#!/bin/bash

python3=/home/cchen/miniconda3/envs/yfliu/bin/python3

log_path=/home/cchen/yfliu/ares/logs/scheduler.log
json_file=/home/cchen/yfliu/ares/logs/scheduler.json

workload_path=./workload/workloads-4h-40j/workload-3.csv
policy=ares
nodes="10.0.0.23 10.0.0.24 10.0.0.25 10.0.0.26"

nohup $python3 -u scheduler.py \
  --workload "$workload_path" \
  --policy "$policy" \
  --nodes "$nodes" \
  --output "$json_file" > "$log_path" 2>&1 &
