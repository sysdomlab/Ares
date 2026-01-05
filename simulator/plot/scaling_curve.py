import os

import math
import numpy as np
from matplotlib import pyplot as plt

from policy.applications import APPLICATIONS

plt.rcParams['font.family'] = 'Heiti TC'


def predict_step_time(app, num_replicas, global_bsz):
    placement = ()
    while sum(placement) < num_replicas:
        placement = (*placement, min(num_replicas - sum(placement), 4))
    local_bsz = math.ceil(global_bsz / num_replicas - 1e-8)  # gpu扩张会导致local_bsz减少而保持target_bsz不变
    accum_steps = math.ceil(local_bsz / app.max_local_bsz - 1e-8) - 1  # accum_steps表示额外的累积步数
    atomic_bsz = math.ceil(local_bsz / (accum_steps + 1) - 1e-8)
    # count = num_replicas * (accum_steps + 1)  # todo 检查差异
    # atomic_bsz = min(atomic_bsz, int(app.max_batch_size / count))
    step_time, sync_time = app.get_throughput(placement, atomic_bsz)  # step_time包含了sync_time
    print(f"  > num_replicas {num_replicas}, "
          # f"placement {placement}, "
          f"global_bsz {global_bsz}, "
          f"local_bsz {local_bsz}/{app.max_local_bsz}, "
          f"atomic_bsz {atomic_bsz}, "
          f"accum_steps {accum_steps}")
    return step_time + (step_time - sync_time) * accum_steps, sync_time


if __name__ == '__main__':
    # prepare data
    all_step_times, all_sync_times, all_efficiency, all_speedup = {}, {}, {}, {}
    for app_name, app in APPLICATIONS.items():
        step_times, sync_times, efficiency, speedup = {}, {}, {}, {}

        print(f"{app_name}:")
        for num_replicas in range(1, 1 + 32):  # [1, 2, 4, 8, 16, 32]:  # range(1, 1 + 32):
            global_bsz = app.max_batch_size / 4

            (step_times[num_replicas], sync_times[num_replicas]) = predict_step_time(app, num_replicas, global_bsz)

            tp1 = global_bsz / step_times[1]  # 1并行单卡吞吐 sample/s
            tpx = global_bsz / step_times[num_replicas]  # x并行单卡吞吐 sample/s
            speedup[num_replicas] = tpx / tp1
            efficiency[num_replicas] = speedup[num_replicas] / num_replicas

            print(f"    > step_time: {step_times[num_replicas]:.2f} s, "
                  f"sync_time_rate: {sync_times[num_replicas] / step_times[num_replicas]:.2f}, "
                  f"efficiency: {efficiency[num_replicas]:.2f}, "
                  f"speedup: {speedup[num_replicas]:.2f}")

        all_step_times[app_name], all_sync_times[app_name], all_efficiency[app_name], all_speedup[app_name] \
            = step_times, sync_times, efficiency, speedup

    label_dict = {
        "bert": "BERT",
        "cifar10": "ResNet18",
        "ncf": "NeuMF",
        "imagenet": "ResNet50",
        "deepspeech2": "DeepSpeech2",
        "yolov3": "YOLOv3",
    }
    fontsize = 21
    legend_fontsize = 21
    left, right, top, bottom = 0.17, 1, 1, 0.14
    linewidth = 2
    markersize = 10

    # 1. iso-efficiency
    plt.figure(figsize=(14, 7))
    plt.style.use('ggplot')

    plt.gcf().patch.set_facecolor('white')
    ax = plt.gca()
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_color('black')
    ax.grid(color='lightgray', linestyle='--', linewidth=0.8)

    num_replicas_list = [1, 2, 4, 8, 16, 32]
    x = np.log2(num_replicas_list)

    plt.plot(x, [1, 1, 1, 1, 1, 1], color='black', linestyle='--', linewidth=linewidth, label="Linear", markersize=markersize)
    for app_name, app in APPLICATIONS.items():
        efficiency = all_efficiency[app_name]
        y = [efficiency[num_replicas] for num_replicas in num_replicas_list]
        plt.plot(x, y, marker='8', linestyle='-', linewidth=linewidth, label=label_dict[app_name], markersize=markersize)

    plt.xticks(x, num_replicas_list, fontsize=fontsize, color='black')
    plt.yticks(fontsize=fontsize, color='black')

    # plt.title("Efficiency", fontsize=fontsize)
    plt.xlabel("GPU 数量", fontsize=fontsize, color='black')
    plt.ylabel("归一化吞吐量", fontsize=fontsize, color='black')
    plt.legend(fontsize=legend_fontsize)
    plt.tight_layout()
    # plt.subplots_adjust(left=left, right=right, top=top, bottom=bottom)
    plt.savefig(os.path.join(os.path.abspath(os.path.dirname(__file__)), f"Efficiency.pdf"))
    plt.savefig(os.path.join(os.path.abspath(os.path.dirname(__file__)), f"Efficiency.png"))
    plt.show()

    # 2. speedup
    plt.figure(figsize=(14, 7))
    print(plt.style.available)
    plt.style.use('ggplot')

    plt.gcf().patch.set_facecolor('white')
    ax = plt.gca()
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_color('black')
    ax.grid(color='lightgray', linestyle='--', linewidth=0.8)

    num_replicas_list = [1, 2, 4, 8, 16, 32]
    x = [0] + num_replicas_list
    plt.plot(x, x, color='black', linestyle='--', linewidth=linewidth, label="Linear", markersize=markersize)
    for app_name, app in APPLICATIONS.items():
        speedup = all_speedup[app_name]
        y = [0] + [speedup[num_replicas] for num_replicas in num_replicas_list]
        plt.plot(x, y, marker='8', linestyle='-', linewidth=linewidth, label=label_dict[app_name], markersize=markersize)

    xticks = list(range(0, 33, 8))
    plt.xticks(xticks, xticks, fontsize=fontsize, color='black')
    plt.yticks(xticks, xticks, fontsize=fontsize, color='black')

    # plt.title("Speedup", fontsize=fontsize)
    plt.xlabel("GPU 数量", fontsize=fontsize, color='black')
    plt.ylabel("归一化吞吐量", fontsize=fontsize, color='black')
    plt.legend(fontsize=legend_fontsize)
    plt.tight_layout()
    # plt.subplots_adjust(left=left, right=right, top=top, bottom=bottom)
    plt.savefig(os.path.join(os.path.abspath(os.path.dirname(__file__)), f"Speedup.pdf"))
    plt.savefig(os.path.join(os.path.abspath(os.path.dirname(__file__)), f"Speedup.png"))
    plt.show()
