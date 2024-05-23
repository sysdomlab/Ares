#!/bin/bash

# Array of IP addresses
ips=("10.0.0.23" "10.0.0.24" "10.0.0.20")

# Loop through the IPs and execute the clean-up command on each
for ip in "${ips[@]}"; do
    echo "Cleaning up on $ip"
    ssh $ip "
        ps -ef | grep worker.py | grep -v grep
        ps -ef | grep worker.py | grep -v grep | awk '{print \$2}' | xargs kill
    "
done

echo "Clean-up completed on all specified IPs."
