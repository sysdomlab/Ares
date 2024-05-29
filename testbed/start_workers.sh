#!/bin/bash

# Array of IP addresses
ips=("10.0.0.22" "10.0.0.23" "10.0.0.24" "10.0.0.26")

python3=/home/cchen/miniconda3/envs/yfliu/bin/python3
work_dir=/home/cchen/yfliu/cluster_schedule/pollux/testbed
log_dir=/home/cchen/yfliu/ares/logs
# Loop through the IPs and execute the command on each
for ip in "${ips[@]}"; do
    echo "Starting worker on $ip"
    ssh $ip "
        cd $work_dir
        nohup $python3 -u worker.py --local_ip $ip > /home/cchen/yfliu/ares/logs/$ip.log 2>&1 &
    "
done
