# -*- coding: gbk -*-
import matplotlib.pyplot as plt
import numpy as np


def plot_runtime_comparison():
    # 设置全局字体大小和粗细，进一步增大字号
    # plt.rcParams.update({
    #     'font.family': 'Times New Roman',  # 设置Times New Roman字体
    #     'font.size': 38,  # 增大基础字体大小 (从32增至38)
    #     'font.weight': 'bold',
    #     'axes.titlesize': 42,  # 增大标题字体大小 (从36增至42)
    #     'axes.labelsize': 40,  # 增大轴标签字体大小 (从34增至40)
    #     'xtick.labelsize': 38,  # 增大x轴刻度标签大小 (从32增至38)
    #     'ytick.labelsize': 38,  # 增大y轴刻度标签大小 (从32增至38)
    #     'legend.fontsize': 38,  # 增大图例字体大小 (从32增至38)
    #     'axes.titleweight': 'bold',
    #     'axes.labelweight': 'bold',
    # })

    plt.rcParams.update({
        'font.family': 'Times New Roman',  # 设置Times New Roman字体
        'font.size': 42,  # 增大基础字体大小
        'font.weight': 'bold',
        'axes.titlesize': 50,  # 增大标题字体大小
        'axes.labelsize': 50,  # 增大轴标签字体大小
        'xtick.labelsize': 45,  # 增大x轴刻度标签大小
        'ytick.labelsize': 45,  # 增大y轴刻度标签大小
        'legend.fontsize': 42,  # 增大图例字体大小
        'axes.titleweight': 'bold',
        'axes.labelweight': 'bold',
    })
    
    # 问题规模数据
    problem_sizes = ['200', '300', '400', '500', '1000']

    # 运行时间数据
    runtime_1 = [765, 871.56, 1106.44, 1529.61, 1918.53]  # MoH数据
    runtime_2 = [1033, 1529.61, 4002.78, 11505.23, 19723]  # ReEvo数据
    runtime_3 = [3135, 7562.85, 19723, 36228, 115505.23]  # EoH数据

    # 创建更大尺寸的图表
    fig, ax = plt.subplots(figsize=(16, 12))  # 增大图形尺寸 (从14,10增至16,12)
    
    # 设置背景色
    ax.set_facecolor('#ffffff')
    fig.patch.set_facecolor('white')

    # 设置柱状图的位置
    x = np.arange(len(problem_sizes))
    width = 0.25

    # 设置专业的配色方案
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    edge_colors = ['#2980b9', '#c0392b', '#27ae60']

    # 绘制三组柱状图，增加柱子宽度和边框粗细
    bars1 = ax.bar(x - width, runtime_1, width, label='NeRM',
                   color=colors[0], alpha=0.8, edgecolor=edge_colors[0], 
                   linewidth=2.5, zorder=2)  # 增加边框粗细
    bars2 = ax.bar(x, runtime_2, width, label='ReEvo',
                   color=colors[1], alpha=0.8, edgecolor=edge_colors[1], 
                   linewidth=2.5, zorder=2)
    bars3 = ax.bar(x + width, runtime_3, width, label='EoH',
                   color=colors[2], alpha=0.8, edgecolor=edge_colors[2], 
                   linewidth=2.5, zorder=2)

    # # 添加趋势线
    # ax.plot(x - width, runtime_1, '-', color=edge_colors[0], linewidth=2,
    #        marker='o', markersize=6, markerfacecolor='white', markeredgewidth=1.5,
    #        markeredgecolor=edge_colors[0], zorder=3)
    # ax.plot(x, runtime_2, '-', color=edge_colors[1], linewidth=2,
    #        marker='s', markersize=6, markerfacecolor='white', markeredgewidth=1.5,
    #        markeredgecolor=edge_colors[1], zorder=3)
    # ax.plot(x + width, runtime_3, '-', color=edge_colors[2], linewidth=2,
    #        marker='^', markersize=6, markerfacecolor='white', markeredgewidth=1.5,
    #        markeredgecolor=edge_colors[2], zorder=3)

    # 在创建图表后，设置y轴上限
    ax.set_ylim(top=60000)

    # 设置x轴标签，使用更大字号
    ax.set_xticks(x)
    ax.set_xticklabels(problem_sizes, fontsize=38, fontweight='bold')  # 从32增至38
    ax.tick_params(axis='both', which='major', labelsize=38, width=3.0, length=12)

    # 设置标题和标签，使用更大字号
    # ax.set_title('Time consumed for different problem sizes',
    #              fontsize=42, pad=25, fontweight='bold')  # 从36增至42
    ax.set_xlabel('Problem Size of TSP instances', fontsize=50, labelpad=20, fontweight='bold')  # 从34增至40
    ax.set_ylabel('Running Time (seconds)', fontsize=50, labelpad=20, fontweight='bold')  # 修改y轴标签

    # 设置y轴为科学计数法
    ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    # 加粗网格线
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='gray', zorder=1, linewidth=2.5)  # 增加网格线宽度
    ax.set_axisbelow(True)

    # 加粗图例，减小边距和背景区域
    legend = ax.legend(loc='upper left', fontsize=38, frameon=True,
                      facecolor='white', edgecolor='black',
                      bbox_to_anchor=(0.02, 0.98),
                      borderpad=0.8,  # 减小边框padding (从2.0减至0.8)
                      handlelength=3,  # 减小图例标记长度 (从4减至3)
                      handletextpad=0.5,  # 减小文本间距 (从1.0减至0.5)
                      borderaxespad=0.5,  # 减小轴边距
                      labelspacing=0.5)  # 减小标签间距
    legend.get_frame().set_alpha(0.9)

    # 加粗坐标轴
    for spine in ax.spines.values():
        spine.set_linewidth(3.0)  # 从2.5增至3.0
        spine.set_color('#333333')

    # 调整布局
    plt.tight_layout()

    # 保存高质量图表
    plt.savefig('single_run_time.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


if __name__ == "__main__":
    plot_runtime_comparison()

