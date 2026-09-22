# Ares: Fair and Efficient Scheduling of Deep Learning Jobs with Elastic Fair Queuing

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

[Paper](https://doi.org/10.1145/3766896) | [Project structure](#project-structure) | [Installation](#installation) | [Running Ares](#running-ares) | [Citation](#citation)

## About

Ares is a scheduler for elastic deep learning training jobs. It uses virtual finish time to identify jobs that can finish earlier and temporarily gives them more resources, while keeping each job's global batch size unchanged and bounding the loss in resource utilization. This elastic fair queuing policy improves completion time while preserving a theoretical fairness guarantee.

The repository contains the testbed and simulator used in the paper. In the reported experiments, Ares reduces average job completion time by more than 20% and the number of unfairly served jobs by more than 40%.

## Project structure

```text
.
├── simulator/
│   ├── simulator.py                  # Trace-driven cluster simulator
│   ├── policy/                       # Ares and baseline scheduling policies
│   ├── traces/                       # Measured training throughput profiles
│   ├── workload/                     # Philly, Saturn, synthetic, and scale traces
│   ├── plot/                         # Figure generation scripts
│   └── start_for_*_simulation.py     # Paper experiment launchers
└── testbed/
    ├── framework.py                  # Elastic distributed training runtime
    ├── scheduler.py                  # Physical-cluster scheduler
    ├── worker.py                     # Worker daemon
    ├── models/                       # Training workloads
    ├── traces/                       # Testbed throughput profiles
    ├── workload/                     # Testbed job traces
    └── start_*.sh                    # Cluster launch scripts
```

## Installation

The simulator is CPU-only. The testbed additionally requires Linux hosts with NVIDIA GPUs, a working CUDA and PyTorch installation, passwordless SSH between the controller and workers, and the datasets used by the selected models.

The paper's testbed used four nodes. Each node had four NVIDIA GeForce RTX 2080 Ti GPUs, two Intel Xeon Gold 6230 CPUs, and 512 GB of DDR4 memory.

```bash
git clone https://github.com/NephrenCake/Ares.git
cd Ares

python3.8 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r simulator/requirements.txt
```

The repository does not provide a unified dependency file for the testbed. Install a PyTorch build compatible with the CUDA version on every worker, followed by the packages required by the selected model. Set dataset and checkpoint paths in `testbed/models/env.py` before launching jobs.

## Running Ares

### Run one simulation

The following command replays one Philly trace on a simulated 16-node cluster with four GPUs per node:

```bash
cd simulator
python simulator.py \
  --policy ares \
  --workload workload/philly/workload-1.csv \
  --nodes "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15" \
  --num-gpus 4 \
  --interval 60 \
  --output simulator_logs/quickstart/ares.json
```

Use `python simulator.py --help` to list the available policies and parameters.

### Reproduce the paper experiments

The launchers encode the workload sets and policy selections used for the paper:

```bash
cd simulator

# Figure 5 simulator results for the testbed workload
python start_for_testbed_simulation.py

# Figures 6 through 8
python start_for_full_simulation.py

# Figure 9
python start_for_sensitivity_simulation.py

# Figure 10
python start_for_scale_simulation.py
```

Before running these scripts, replace the hard-coded Python interpreter path and review the enabled entries in each `policies` list. Results are written below `simulator/simulator_logs/`. The plotting entry point is:

```bash
cd simulator/plot
python collect_result_v2.py
```

### Run the physical testbed

Edit `testbed/start_workers.sh` and `testbed/start_scheduler.sh` for the worker addresses, repository location, Python interpreter, output directory, workload, and policy on your cluster. Then run:

```bash
cd testbed
bash start_workers.sh
bash start_scheduler.sh
```

The worker daemon listens on port `4242` by default. Stop the processes with `clean_scheduler.sh` and `clean_workers.sh`. See [`testbed/README.md`](testbed/README.md) for a standalone training command and model-specific notes.

## Citation

If you use Ares, please cite:

```bibtex
@article{liu2025ares,
  author  = {Yifei Liu and Chen Chen and Qiang Wang and Yu Feng and Weihao Cui and Quan Chen and Minyi Guo},
  title   = {Ares: Fair and Efficient Scheduling of Deep Learning Jobs with Elastic Fair Queuing},
  journal = {ACM Transactions on Architecture and Code Optimization},
  volume  = {22},
  number  = {4},
  pages   = {1--21},
  year    = {2025},
  doi     = {10.1145/3766896}
}
```

## License

Ares is released under the [Apache License 2.0](LICENSE). Parts of the implementation are adapted from [Pollux/AdaptDL](https://github.com/petuum/adaptdl/tree/osdi21-artifact); their copyright and license notices remain in the corresponding source files.

## Contact

Questions and feedback are welcome at [nephrencake@sjtu.edu.cn](mailto:nephrencake@sjtu.edu.cn).
