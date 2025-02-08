import os
import re

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.patches import Patch

fontsize = 28
legend_fontsize = 15
linewidth = 2
markersize = 10
figsize = (7, 4)
colors = ['#e24933', '#348abd', '#fbc15e']


def fig1():
    # 创建一个新的图形
    fig, ax = plt.subplots(figsize=figsize)

    # 1
    job1 = patches.Rectangle((0, 4), 6, 2, linewidth=0, edgecolor='black', facecolor=colors[0])
    ax.add_patch(job1)

    # 2
    job2 = patches.Rectangle((0, 0), 6, 2, linewidth=0, edgecolor='black', facecolor=colors[1])
    ax.add_patch(job2)
    job2 = patches.Rectangle((6, 0), 2, 6, linewidth=0, edgecolor='black', facecolor=colors[1])
    ax.add_patch(job2)

    # 3
    job3 = patches.Rectangle((0, 2), 6, 2, linewidth=0, edgecolor='black', facecolor=colors[2])
    ax.add_patch(job3)

    # 设置图的范围和标签
    plt.xlim(0, 8.5)
    plt.ylim(0, 6)
    plt.xlabel('Time (hrs)', fontsize=fontsize, color='black')
    plt.ylabel('GPU #', fontsize=fontsize, color='black')
    plt.xticks(range(9), fontsize=fontsize, color='black')
    plt.yticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], [1, 2, 3, 4, 5, 6], fontsize=fontsize, color='black')

    # 在横轴上添加箭头
    ax.annotate('', xy=(8.5, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color='black', lw=4))
    ax.annotate('', xy=(0, 6), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-", color='black', lw=4))

    # 去掉画布的边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(left=False, bottom=False)  # 隐藏刻度

    # 添加文字
    ax.text(3, 5, 'Job-1', horizontalalignment='center', verticalalignment='center', fontsize=fontsize,
            color='white')
    ax.text(3, 3, 'Job-3', horizontalalignment='center', verticalalignment='center', fontsize=fontsize,
            color='white')
    ax.text(4, 1, 'Job-2', horizontalalignment='center', verticalalignment='center', fontsize=fontsize,
            color='white')

    # 添加图例
    # legend_patches = [Patch(color=color, label=label)
    #                   for color, label in zip(colors, [f"Job-{i}" for i in range(1, 4)])]
    # fig.legend(handles=legend_patches,
    #            loc='upper center',
    #            bbox_to_anchor=(0.5, 1.025),
    #            ncol=3,
    #            fontsize=legend_fontsize,
    #            frameon=False)
    # plt.tight_layout(rect=[0, 0, 1, 0.9])  # 调整布局以防止标签被裁剪
    plt.tight_layout()  # 调整布局以防止标签被裁剪

    # 显示图形
    plt.savefig(os.path.join(os.path.abspath(os.path.dirname(__file__)), f"motivation1.pdf"))
    plt.savefig(os.path.join(os.path.abspath(os.path.dirname(__file__)), f"motivation1.png"))
    plt.show()

    print(os.path.join(os.path.abspath(os.path.dirname(__file__)), f"motivation1.png"))


def fig2():
    # 创建一个新的图形
    fig, ax = plt.subplots(figsize=figsize)

    # 添加Job-1的矩形
    job1 = patches.Rectangle((0, 0), 2, 6, linewidth=0, edgecolor='black', facecolor=colors[0])
    ax.add_patch(job1)

    # 添加Job-2的矩形
    job2 = patches.Rectangle((2, 0), 2, 6, linewidth=0, edgecolor='black', facecolor=colors[2])
    ax.add_patch(job2)

    # 添加Job-2的矩形
    job2 = patches.Rectangle((4, 0), 4, 6, linewidth=0, edgecolor='black', facecolor=colors[1])
    ax.add_patch(job2)

    # 设置图的范围和标签
    plt.xlim(0, 8.5)
    plt.ylim(0, 6)
    plt.xlabel('Time (hrs)', fontsize=fontsize, color='black')
    plt.ylabel('GPU #', fontsize=fontsize, color='black')
    plt.xticks(range(9), fontsize=fontsize, color='black')
    plt.yticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], [1, 2, 3, 4, 5, 6], fontsize=fontsize, color='black')

    # 在横轴上添加箭头
    ax.annotate('', xy=(8.5, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color='black', lw=4))
    ax.annotate('', xy=(0, 6), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-", color='black', lw=4))

    # 去掉画布的边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(left=False, bottom=False)  # 隐藏刻度

    # 添加文字
    ax.text(1, 3, 'Job-1', horizontalalignment='center', verticalalignment='center', fontsize=fontsize,
            color='white')
    ax.text(3, 3, 'Job-3', horizontalalignment='center', verticalalignment='center', fontsize=fontsize,
            color='white')
    ax.text(6, 3, 'Job-2', horizontalalignment='center', verticalalignment='center', fontsize=fontsize,
            color='white')

    # 添加图例
    # legend_patches = [Patch(color=color, label=label)
    #                   for color, label in zip(colors, [f"Job-{i}" for i in range(1, 4)])]
    # fig.legend(handles=legend_patches,
    #            loc='upper center',
    #            bbox_to_anchor=(0.5, 1.025),
    #            ncol=3,
    #            fontsize=legend_fontsize,
    #            frameon=False)
    # plt.tight_layout(rect=[0, 0, 1, 0.9])  # 调整布局以防止标签被裁剪
    plt.tight_layout()  # 调整布局以防止标签被裁剪

    # 显示图形
    plt.savefig(os.path.join(os.path.abspath(os.path.dirname(__file__)), f"motivation2.pdf"))
    plt.savefig(os.path.join(os.path.abspath(os.path.dirname(__file__)), f"motivation2.png"))
    plt.show()

    print(os.path.join(os.path.abspath(os.path.dirname(__file__)), f"motivation2.png"))


def fig1_():
    # 创建一个新的图形
    fig, ax = plt.subplots(figsize=figsize)

    # 1
    job1 = patches.Rectangle((0, 3), 2, 3, linewidth=0, edgecolor='black', facecolor=colors[0])
    ax.add_patch(job1)
    job1 = patches.Rectangle((0, 0), 2, 3, linewidth=0, edgecolor='black', facecolor=colors[1])
    ax.add_patch(job1)

    # 2
    job2 = patches.Rectangle((2, 4), 3, 2, linewidth=0, edgecolor='black', facecolor=colors[0])
    ax.add_patch(job2)
    job2 = patches.Rectangle((2, 2), 3, 2, linewidth=0, edgecolor='black', facecolor=colors[2])
    ax.add_patch(job2)
    job2 = patches.Rectangle((2, 0), 3, 2, linewidth=0, edgecolor='black', facecolor=colors[1])
    ax.add_patch(job2)

    # 3
    job1 = patches.Rectangle((5, 3), 2, 3, linewidth=0, edgecolor='black', facecolor=colors[2])
    ax.add_patch(job1)
    job1 = patches.Rectangle((5, 0), 2, 3, linewidth=0, edgecolor='black', facecolor=colors[1])
    ax.add_patch(job1)

    # 4
    job1 = patches.Rectangle((7, 0), 1, 6, linewidth=0, edgecolor='black', facecolor=colors[1])
    ax.add_patch(job1)

    # 设置图的范围和标签
    plt.xlim(0, 8.5)
    plt.ylim(0, 6)
    plt.xlabel('Time (hrs)', fontsize=fontsize, color='black')
    plt.ylabel('GPU #', fontsize=fontsize, color='black')
    plt.xticks(range(9), fontsize=fontsize, color='black')
    plt.yticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], [1, 2, 3, 4, 5, 6], fontsize=fontsize, color='black')

    # 在横轴上添加箭头
    ax.annotate('', xy=(8.5, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color='black', lw=4))
    ax.annotate('', xy=(0, 6), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-", color='black', lw=4))

    # 去掉画布的边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(left=False, bottom=False)  # 隐藏刻度

    # 添加文字
    ax.text(2.5, 5, 'Job-1', horizontalalignment='center', verticalalignment='center', fontsize=fontsize,
            color='white')
    ax.text(3.5, 3, 'Job-3', horizontalalignment='center', verticalalignment='center', fontsize=fontsize,
            color='white')
    ax.text(4.5, 1, 'Job-2', horizontalalignment='center', verticalalignment='center', fontsize=fontsize,
            color='white')

    # 添加图例
    # legend_patches = [Patch(color=color, label=label)
    #                   for color, label in zip(colors, [f"Job-{i}" for i in range(1, 4)])]
    # fig.legend(handles=legend_patches,
    #            loc='upper center',
    #            bbox_to_anchor=(0.5, 1.025),
    #            ncol=3,
    #            fontsize=legend_fontsize,
    #            frameon=False)
    # plt.tight_layout(rect=[0, 0, 1, 0.9])  # 调整布局以防止标签被裁剪
    plt.tight_layout()  # 调整布局以防止标签被裁剪

    # 显示图形
    plt.savefig(os.path.join(os.path.abspath(os.path.dirname(__file__)), f"motivation3.pdf"))
    plt.savefig(os.path.join(os.path.abspath(os.path.dirname(__file__)), f"motivation3.png"))
    plt.show()

    print(os.path.join(os.path.abspath(os.path.dirname(__file__)), f"motivation3.png"))


def fig2_():
    # 创建一个新的图形
    fig, ax = plt.subplots(figsize=figsize)

    # 添加Job-1的矩形
    job1 = patches.Rectangle((0, 2), 3, 4, linewidth=0, edgecolor='black', facecolor=colors[0])
    ax.add_patch(job1)

    # 添加Job-2的矩形
    job2 = patches.Rectangle((2, 0), 1, 2, linewidth=0, edgecolor='black', facecolor=colors[2])
    ax.add_patch(job2)
    job2 = patches.Rectangle((3, 0), 2.5, 4, linewidth=0, edgecolor='black', facecolor=colors[2])
    ax.add_patch(job2)

    # 添加Job-2的矩形
    job2 = patches.Rectangle((0, 0), 2, 2, linewidth=0, edgecolor='black', facecolor=colors[1])
    ax.add_patch(job2)
    job2 = patches.Rectangle((3, 4), 5, 2, linewidth=0, edgecolor='black', facecolor=colors[1])
    ax.add_patch(job2)
    job2 = patches.Rectangle((5.5, 0), 2.5, 4, linewidth=0, edgecolor='black', facecolor=colors[1])
    ax.add_patch(job2)

    # 设置图的范围和标签
    plt.xlim(0, 8.5)
    plt.ylim(0, 6)
    plt.xlabel('Time (hrs)', fontsize=fontsize, color='black')
    plt.ylabel('GPU #', fontsize=fontsize, color='black')
    plt.xticks(range(9), fontsize=fontsize, color='black')
    plt.yticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], [1, 2, 3, 4, 5, 6], fontsize=fontsize, color='black')

    # 在横轴上添加箭头
    ax.annotate('', xy=(8.5, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color='black', lw=4))
    ax.annotate('', xy=(0, 6), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-", color='black', lw=4))

    # 去掉画布的边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(left=False, bottom=False)  # 隐藏刻度

    # 添加文字
    ax.text(1.5, 4, 'Job-1', horizontalalignment='center', verticalalignment='center', fontsize=fontsize,
            color='white')
    ax.text(4, 2, 'Job-3', horizontalalignment='center', verticalalignment='center', fontsize=fontsize,
            color='white')
    ax.text(6.75, 3, 'Job-2', horizontalalignment='center', verticalalignment='center', fontsize=fontsize,
            color='white')

    # 添加图例
    # legend_patches = [Patch(color=color, label=label)
    #                   for color, label in zip(colors, [f"Job-{i}" for i in range(1, 4)])]
    # fig.legend(handles=legend_patches,
    #            loc='upper center',
    #            bbox_to_anchor=(0.5, 1.025),
    #            ncol=3,
    #            fontsize=legend_fontsize,
    #            frameon=False)
    # plt.tight_layout(rect=[0, 0, 1, 0.9])  # 调整布局以防止标签被裁剪
    plt.tight_layout()  # 调整布局以防止标签被裁剪

    # 显示图形
    plt.savefig(os.path.join(os.path.abspath(os.path.dirname(__file__)), f"motivation4.pdf"))
    plt.savefig(os.path.join(os.path.abspath(os.path.dirname(__file__)), f"motivation4.png"))
    plt.show()

    print(os.path.join(os.path.abspath(os.path.dirname(__file__)), f"motivation4.png"))


def get_metric(exp, metric="cer", t="endo_elasticity"):
    all_epochs = []
    all_cers = []

    for seed in ["-seed0", "-seed1234", "-seed6666"] if t == "endo_elasticity" else ["-seed0"] * 3:
        try:
            file_path = f"/home/cchen/yfliu/ares/bkp/motivation/{t}/{exp}{seed}/restart_0_rank_0.log"
            with open(file_path, "r") as f:
                text = f.read()
        except FileNotFoundError:
            file_path = f"/home/cchen/yfliu/ares/bkp/motivation/{t}/{exp}{seed}/rank_0.log"
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

        all_epochs.append(epochs)
        all_cers.append(cers)

    return all_epochs, all_cers


def plot_with_range(ax, epochs, metrics, label, t="endo_elasticity", color=None, marker='o'):
    min_metrics = np.min(metrics, axis=0)
    max_metrics = np.max(metrics, axis=0)
    mean_metrics = np.mean(metrics, axis=0)

    ax.plot(epochs[0], mean_metrics, marker=marker, label=label, color=color)
    ax.fill_between(epochs[0], min_metrics, max_metrics, alpha=0.3, color=color)


def fig3():
    fontsize = 35
    legend_fontsize = 22
    linewidth = 2
    markersize = 10

    colors = {
        "16-GPU": '#e24a33',
        "8-GPU": '#348abd',
        "4-GPU": '#988ed5',
        "2-GPU": '#777777',
    }
    markers = {'endo_elasticity': 'o', 'exo_elasticity': 'v'}
    custom_lines = [
        plt.Line2D([0], [0], color=colors["16-GPU"]),
        plt.Line2D([0], [0], color=colors["8-GPU"]),
        plt.Line2D([0], [0], color=colors["4-GPU"]),
        plt.Line2D([0], [0], color=colors["2-GPU"]),
        plt.Line2D([0], [0], color='black', marker='o'),
        plt.Line2D([0], [0], color='black', marker='v'),
    ]

    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(8, 7))

    app = {
        "deepspeech2-5": "2-GPU",
        "deepspeech2-4": "4-GPU",
        "deepspeech2-3": "8-GPU",
        "deepspeech2-2": "16-GPU",
    }

    for i, (exp, label) in enumerate(app.items()):
        if label == "16-GPU":
            continue
        epochs, metrics = get_metric(exp, t="exo_elasticity")
        print(f"{exp} {label}, accuracy5: {[i[-1] for i in metrics]}, avg: {np.average([i[-1] for i in metrics])}, "
              f"diff: {np.max([i[-1] for i in metrics]) - np.min([i[-1] for i in metrics])}")
        plot_with_range(ax, epochs, metrics, label, t="exo_elasticity", color=colors[label],
                        marker=markers['exo_elasticity'])

    for i, (exp, label) in enumerate(app.items()):
        epochs, metrics = get_metric(exp)
        print(f"{exp} {label}, accuracy5: {[i[-1] for i in metrics]}, avg: {np.average([i[-1] for i in metrics])}, "
              f"diff: {np.max([i[-1] for i in metrics]) - np.min([i[-1] for i in metrics])}")
        plot_with_range(ax, epochs, metrics, label, color=colors[label], marker=markers['endo_elasticity'])

    ax.set_xlabel('Epoch', fontsize=fontsize, color='black')
    ax.set_ylabel('Character Error Rate', fontsize=fontsize, color='black')
    ax.tick_params(axis='both', which='major', labelsize=fontsize, colors='black')

    ax.legend(custom_lines, ['16-GPU', '8-GPU', '4-GPU', '2-GPU', 'endo-elasticity', 'exo-elasticity'],
              fontsize=legend_fontsize)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.abspath(os.path.dirname(__file__)), f"endo_elasticity1.pdf"))
    plt.savefig(os.path.join(os.path.abspath(os.path.dirname(__file__)), f"endo_elasticity1.png"))
    plt.show()
    plt.clf()

    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(8, 7))

    app = {
        "imagenet-5": "2-GPU",
        "imagenet-1": "4-GPU",
        "imagenet-2": "8-GPU",
        "imagenet-3": "16-GPU",
    }

    for i, (exp, label) in enumerate(app.items()):
        if label == "16-GPU":
            continue
        epochs, metrics = get_metric(exp, "accuracy5", t="exo_elasticity")
        print(f"{exp} {label}, accuracy5: {[i[-1] for i in metrics]}, avg: {np.average([i[-1] for i in metrics])}, "
              f"diff: {np.max([i[-1] for i in metrics]) - np.min([i[-1] for i in metrics])}")
        plot_with_range(ax, epochs, metrics, label, t="exo_elasticity", color=colors[label],
                        marker=markers['exo_elasticity'])

    for i, (exp, label) in enumerate(app.items()):
        epochs, metrics = get_metric(exp, "accuracy5")
        print(f"{exp} {label}, accuracy5: {[i[-1] for i in metrics]}, avg: {np.average([i[-1] for i in metrics])}, "
              f"diff: {np.max([i[-1] for i in metrics]) - np.min([i[-1] for i in metrics])}")
        plot_with_range(ax, epochs, metrics, label, color=colors[label], marker=markers['endo_elasticity'])

    ax.set_xlabel('Epoch', fontsize=fontsize, color='black')
    ax.set_ylabel('Top-5 Accuracy', fontsize=fontsize, color='black')
    ax.tick_params(axis='both', which='major', labelsize=fontsize, colors='black')

    ax.legend(custom_lines, ['16-GPU', '8-GPU', '4-GPU', '2-GPU', 'endo-elasticity', 'exo-elasticity'],
              fontsize=legend_fontsize)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.abspath(os.path.dirname(__file__)), f"endo_elasticity2.pdf"))
    plt.savefig(os.path.join(os.path.abspath(os.path.dirname(__file__)), f"endo_elasticity2.png"))
    plt.show()
    plt.clf()


if __name__ == '__main__':
    # 调度示意图
    fig1()
    fig2()
    fig1_()
    fig2_()
    # 内生弹性精度
    # fig3()
