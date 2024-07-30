#!/bin/bash

policy=fifo

python3=/home/cchen/miniconda3/envs/yfliu/bin/python3
#workload_path=./workload/endo-elasticity.csv
workload_path=./workload/exo-elasticity.csv

nodes="10.0.0.17 10.0.0.20 10.0.0.21 10.0.0.22 10.0.0.23 10.0.0.24 10.0.0.25 10.0.0.26"
namespace="seed0"
log_path=/home/cchen/yfliu/ares/logs/"$namespace".log
json_file=/home/cchen/yfliu/ares/logs/"$namespace".json
nohup $python3 -u scheduler.py \
  --workload "$workload_path" \
  --policy "$policy" \
  --nodes "$nodes" \
  --output "$json_file" \
  --namespace "$namespace"  > "$log_path" 2>&1 &
