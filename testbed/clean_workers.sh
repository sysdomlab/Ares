#!/bin/bash

# Array of IP addresses
ips=(
  "10.0.0.11"
  "10.0.0.12"
  "10.0.0.13"
#  "10.0.0.14"
  "10.0.0.15"
  "10.0.0.16"
  "10.0.0.17"
#  "10.0.0.18"
  "10.0.0.19"
  "10.0.0.20"
  "10.0.0.21"
  "10.0.0.22"
  "10.0.0.23"
  "10.0.0.24"
  "10.0.0.25"
  "10.0.0.26"
)

# Loop through the IPs and execute the clean-up command on each
for ip in "${ips[@]}"; do
    echo "Cleaning up on $ip"
    ssh $ip "
        ps -ef | grep worker.py | grep -v grep
        ps -ef | grep worker.py | grep -v grep | awk '{print \$2}' | xargs kill
    "
done

echo "Clean-up completed on all specified IPs."
