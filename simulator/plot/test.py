import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Heiti TC'

font = {'family': 'Heiti TC', 'size': 16}
plt.rc('font', **font)

# 1. 整理数据
tasks = ['ResNet-50', 'Vgg-19', 'MobileNetV3', 'PPO']  # 任务名称
k80_data = [1.0, 1.0, 1.0, 1.0]          # K80吞吐量
v100_data = [6.5, 6.3, 4.7, 1.6]         # V100吞吐量
rtx3090_data = [8.9, 8.2, 6.8, 1.8]     # RTX 3090吞吐量

# 2. 配置柱状图参数
width = 0.25  # 柱子宽度
x = np.arange(len(tasks))  # 任务对应的x轴位置

# 3. 绘制分组柱状图
plt.figure(figsize=(8, 6))  # 设置画布大小
plt.bar(x - width, k80_data, width=width, label='K80', color='#1f77b4')  # K80（蓝色）
plt.bar(x, v100_data, width=width, label='V100', color='#ff7f0e')  # V100（橙色）
plt.bar(x + width, rtx3090_data, width=width, label='RTX 3090', color='#2ca02c')  # RTX 3090（绿色）

# 4. 添加图表元素
plt.xlabel('任务')
plt.ylabel('吞吐量（归一化于K80）')
plt.xticks(x, tasks)  # 将x轴刻度替换为任务名称
plt.ylim(0, 10)  # 匹配原图y轴范围
plt.legend()  # 显示图例

# 5. 显示/保存图表
plt.tight_layout()  # 自动调整布局
plt.savefig('c1.pdf')
plt.show()
