# Ares: Fair and Efficient Scheduling of Deep Learning Jobs with Elastic Fair Queuing

This repository contains the artifact for the INFOCOM'25 paper "Ares: Fair and Efficient Scheduling of Deep Learning Jobs with Elastic Fair Queuing". We would like to thank the Pollux authors for open-sourcing their implementation!

## Ares Implementation

Key files and modules:

- testbed/framework.py: Implementation of the distributed elastic training framework.
- testbed/scheduler.py: Implementation of the physical cluster scheduler.
- testbed/worker.py: Implementation of the physical cluster worker.
- simulator/simulator.py: Implementation of the cluster simulator.
- simulator/policy : Scheduling algorithms for Ares and baselines.
- simulator/traces: Throughput data from the physical cluster.
- simulator/workload: Traces used for job submission replay.
- simulator/plot: Script for generating the figures in the paper.

## Reproducing Experiments

### Enviroment

We run our testbed experiments in a 16-GPU (4-node) cluster, where each node is equipped with 4 NVIDIA GeForce RTX 2080 Ti GPUs, 2 Intel Xeon Gold 6230 CPUs and 512GB DDR4 RAM. 

`pip3 install simulator/requirements.txt`

**Please read the code before running the scripts and modify the corresponding file storage paths (it’s recommended to search for all paths with the `/home` prefix).**

### Testbed (Fig. 5)

Run a trace from Philly with 40 jobs in a 4-hour submission window.

1. Run Ares and the six baseline simulations in one go: `cd simulator && python3 start_for_testbed_simulation.py`
2. Review the script and modify the policy for each scheduler: `cd tesbed && bash clean_workers.sh && bash clean_scheduler.sh && bash start_workers.sh && bash start_scheduler.sh`
3. Plot Fig. 5：`cd simulator/plot && python3 collect_result_v2.py`

### Simulation (Fig. 6-10)

1. Collect data for Figs. 6-8: `cd simulator && python3 start_for_full_simulation.py`
2. Collect data for Fig. 9: `cd simulator && python3 start_for_sensitivity_simulation.py`
3. Collect data for Fig. 10: `cd simulator && python3 start_for_scale_simulation.py`
4. Plot Figs. 6-10: `cd simulator/plot && python3 collect_result_v2.py`

