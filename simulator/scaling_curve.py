import math
import numpy as np
from matplotlib import pyplot as plt

from policy.applications import APPLICATIONS


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

    # iso-efficiency
    plt.style.use('ggplot')
    for app_name, app in APPLICATIONS.items():
        efficiency = all_efficiency[app_name]

        # 数据存储列表
        num_replicas_list, values = [], []
        for num_replicas in [1, 2, 4, 8, 16, 32]:  # [1, 2, 4, 8, 16, 32]:  # range(1, 1 + 32):
            # 将数据添加到列表中
            num_replicas_list.append(num_replicas)
            values.append(efficiency[num_replicas])

        # 绘制折线图
        num_replicas_log2 = np.log2(num_replicas_list)
        plt.plot(num_replicas_log2, values, label=app_name)

        power_of_2 = [2 ** i for i in range(int(math.log2(max(num_replicas_list))) + 1)]
        power_of_2_log2 = np.log2(power_of_2)
        plt.xticks(power_of_2_log2, power_of_2)

    plt.title("Iso-Efficiency")
    plt.xlabel("Num of GPUs")
    plt.ylabel("Normalized Throughput")
    plt.legend()
    plt.show()

    # speedup
    plt.style.use('ggplot')
    for app_name, app in APPLICATIONS.items():
        speedup = all_speedup[app_name]

        # 数据存储列表
        num_replicas_list, values = [0], [0]
        for num_replicas in [1, 2, 4, 8, 16, 32]:  # [1, 2, 4, 8, 16, 32]:  # range(1, 1 + 32):
            # 将数据添加到列表中
            num_replicas_list.append(num_replicas)
            values.append(speedup[num_replicas])

        # 绘制折线图
        # num_replicas_log2 = np.log2(num_replicas_list)
        plt.plot(num_replicas_list, values, label=app_name)

        # power_of_2 = [2 ** i for i in range(int(math.log2(max(num_replicas_list))) + 1)]
        # power_of_2_log2 = np.log2(power_of_2)
        # plt.xticks(power_of_2_log2, power_of_2)
        xticks = list(range(0, 33, 4))
        plt.xticks(xticks, xticks)

    plt.title("Speedup")
    plt.xlabel("Num of GPUs")
    plt.ylabel("Normalized Throughput")
    plt.legend()
    plt.show()

