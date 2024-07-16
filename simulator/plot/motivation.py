import os
import re

import matplotlib.pyplot as plt
import matplotlib.patches as patches

fontsize = 32
legend_fontsize = 21
left, right, top, bottom = 0.17, 1, 1, 0.14
linewidth = 2
markersize = 10
figsize = (6, 3)


def fig1():
    # 创建一个新的图形
    fig, ax = plt.subplots(figsize=figsize)

    # 添加Job-1的矩形
    job1 = patches.Rectangle((0, 1), 3, 1, linewidth=0, edgecolor='black', facecolor='navy', label='Job-1')
    ax.add_patch(job1)

    # 添加Job-2的矩形
    job2 = patches.Rectangle((0, 0), 3, 1, linewidth=0, edgecolor='black', facecolor='orange', label='Job-2')
    ax.add_patch(job2)
    job2_2 = patches.Rectangle((3, 0), 2, 2, linewidth=0, edgecolor='black', facecolor='orange')
    ax.add_patch(job2_2)

    # 设置图的范围和标签
    plt.xlim(0, 5)
    plt.ylim(0, 2)
    plt.xlabel('Time', fontsize=fontsize, color='black')
    plt.ylabel('GPU #', fontsize=fontsize, color='black')
    plt.xticks(range(8), fontsize=fontsize, color='black')
    plt.yticks([0.5, 1.5], [1, 2], fontsize=fontsize, color='black')

    # 在横轴上添加箭头
    ax.annotate('', xy=(7, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color='black', lw=4))

    # 去掉画布的边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(left=False, bottom=False)  # 隐藏刻度

    # 添加文字
    ax.text(1.5, 1.5, 'Job-1', horizontalalignment='center', verticalalignment='center', fontsize=fontsize,
            color='white')
    ax.text(2.5, 0.5, 'Job-2', horizontalalignment='center', verticalalignment='center', fontsize=fontsize,
            color='black')
    # ax.text(4, 1, 'Job-2', horizontalalignment='center', verticalalignment='center', fontsize=fontsize, color='black')

    # 添加图例
    # plt.legend(handles=[job1, job2])
    plt.tight_layout()  # 调整布局以防止标签被裁剪

    # 显示图形
    plt.savefig(os.path.join(os.path.abspath(os.path.dirname(__file__)), f"motivation1.pdf"))
    plt.savefig(os.path.join(os.path.abspath(os.path.dirname(__file__)), f"motivation1.png"))
    plt.show()


def fig2():
    # 创建一个新的图形
    fig, ax = plt.subplots(figsize=figsize)

    # 添加Job-1的矩形
    job1 = patches.Rectangle((0, 0), 2, 2, linewidth=0, edgecolor='black', facecolor='navy', label='Job-1')
    ax.add_patch(job1)

    # 添加Job-2的矩形
    job2 = patches.Rectangle((2, 0), 4, 2, linewidth=0, edgecolor='black', facecolor='orange', label='Job-2')
    ax.add_patch(job2)

    # 设置图的范围和标签
    plt.xlim(0, 5)
    plt.ylim(0, 2)
    plt.xlabel('Time', fontsize=fontsize, color='black')
    plt.ylabel('GPU #', fontsize=fontsize, color='black')
    plt.xticks(range(8), fontsize=fontsize, color='black')
    plt.yticks([0.5, 1.5], [1, 2], fontsize=fontsize, color='black')

    # 在横轴上添加箭头
    ax.annotate('', xy=(7, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color='black', lw=4))

    # 去掉画布的边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(left=False, bottom=False)  # 隐藏刻度

    # 添加文字
    ax.text(1, 1, 'Job-1', horizontalalignment='center', verticalalignment='center', fontsize=fontsize,
            color='white')
    ax.text(4, 1, 'Job-2', horizontalalignment='center', verticalalignment='center', fontsize=fontsize,
            color='black')
    # ax.text(4, 1, 'Job-2', horizontalalignment='center', verticalalignment='center', fontsize=fontsize, color='black')

    # 添加图例
    # plt.legend(handles=[job1, job2])
    plt.tight_layout()  # 调整布局以防止标签被裁剪

    # 显示图形
    plt.savefig(os.path.join(os.path.abspath(os.path.dirname(__file__)), f"motivation2.pdf"))
    plt.savefig(os.path.join(os.path.abspath(os.path.dirname(__file__)), f"motivation2.png"))
    plt.show()


def get_metric(exp, metric="cer"):
    for i in ["-seed0", "-seed1234", "-seed6666"]:
        try:
            file_path = f"/home/cchen/yfliu/ares/bkp/motivation/{exp}{i}/restart_0_rank_0.log"
            with open(file_path, "r") as f:
                text = f.read()
        except FileNotFoundError:
            file_path = f"/home/cchen/yfliu/ares/bkp/motivation/{exp}{i}/rank_0.log"
            with open(file_path, "r") as f:
                text = f.read()
        pattern = r"Finish Epoch (\d+).*?" + metric + r"=([\d.]+)"

        matches = re.findall(pattern, text)

        epochs = []
        cers = []

        for match in matches:
            epoch, cer = match
            epochs.append(int(epoch))
            cers.append(float(cer) / 100)
            # print(f"Epoch: {epoch}, cer: {cer}")

        print(cers)
        input()

    return epochs, cers


def fig3():
    app = {
        "deepspeech2-1": "32-GPU",
        "deepspeech2-2": "16-GPU",
        "deepspeech2-3": "8-GPU",
        "deepspeech2-4": "4-GPU",
    }

    fontsize = 32
    legend_fontsize = 19
    linewidth = 2
    markersize = 10
    plt.figure(figsize=(8, 7))
    plt.style.use('ggplot')
    for i in app:
        # file_path = f"/home/cchen/yfliu/ares/bkp/motivation/{i}/restart_0_rank_0.log"
        epochs, metrics = get_metric(i)
        print(f"{i} {app[i]}, Epoch: {epochs[-1]}, cer: {metrics[-1]}")
        plt.plot(epochs, metrics, marker='o', label=app[i])

    plt.xlabel('Epoch', fontsize=fontsize, color='black')
    plt.ylabel('Character Error Rate', fontsize=fontsize, color='black')
    plt.xticks(fontsize=fontsize, color='black')
    plt.yticks(fontsize=fontsize, color='black')
    plt.legend(fontsize=legend_fontsize)
    plt.tight_layout()
    # plt.savefig("cer_vs_epoch.png")
    plt.show()


    app = {
        "imagenet-4": "32-GPU",
        "imagenet-3": "16-GPU",
        "imagenet-2": "8-GPU",
        "imagenet-1": "4-GPU",
    }

    fontsize = 32
    legend_fontsize = 19
    linewidth = 2
    markersize = 10
    plt.figure(figsize=(8, 7))
    plt.style.use('ggplot')
    for i in app:
        # file_path = f"/home/cchen/yfliu/ares/bkp/motivation/{i}/restart_0_rank_0.log"
        epochs, metrics = get_metric(i, "accuracy5")
        print(f"{i} {app[i]}, Epoch: {epochs[-1]}, accuracy5: {metrics[-1]}")
        plt.plot(epochs, metrics, marker='o', label=app[i])

    plt.xlabel('Epoch', fontsize=fontsize, color='black')
    plt.ylabel('Top-5 Accuracy', fontsize=fontsize, color='black')
    plt.xticks(fontsize=fontsize, color='black')
    plt.yticks(fontsize=fontsize, color='black')
    plt.legend(fontsize=legend_fontsize)
    plt.tight_layout()
    # plt.savefig("cer_vs_epoch.png")
    plt.show()


if __name__ == '__main__':
    # 调度示意图
    # fig1()
    # fig2()
    # 内生弹性精度
    fig3()
