import os

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


fig1()
fig2()
