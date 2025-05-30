# -*- coding: gbk -*-
import matplotlib.pyplot as plt
import numpy as np

def plot_runtime_comparison():
    # 设置全局字体和字号
    plt.rcParams.update({
        'font.family': 'Times New Roman',  # 设置Times New Roman字体
        'font.size': 42,  # 基础字体大小
        'font.weight': 'bold',
        'axes.titlesize': 50,  # 标题字体大小
        'axes.labelsize': 50,  # 轴标签字体大小
        'xtick.labelsize': 45,  # x轴刻度标签大小
        'ytick.labelsize': 45,  # y轴刻度标签大小
        'legend.fontsize': 42,  # 图例字体大小
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
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # 设置背景色
    ax.set_facecolor('#ffffff')
    fig.patch.set_facecolor('white')

    # 设置专业的配色方案
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    edge_colors = ['#2980b9', '#c0392b', '#27ae60']
    
    # 绘制三条线
    ax.plot(range(len(problem_sizes)), runtime_1, 'o-', label='NeRM', 
           color=colors[0], linewidth=5, markersize=16,
           markeredgewidth=3, markerfacecolor='white')
    ax.plot(range(len(problem_sizes)), runtime_2, 's-', label='ReEvo', 
           color=colors[1], linewidth=5, markersize=16,
           markeredgewidth=3, markerfacecolor='white')
    ax.plot(range(len(problem_sizes)), runtime_3, '^-', label='EoH', 
           color=colors[2], linewidth=5, markersize=16,
           markeredgewidth=3, markerfacecolor='white')
    
    # 设置x轴标签
    ax.set_xticks(range(len(problem_sizes)))
    ax.set_xticklabels(problem_sizes, fontsize=45, fontweight='bold')
    ax.tick_params(axis='both', which='major', width=3.0, length=12)
    
    # 设置标签
    ax.set_xlabel('Problem Size of TSP instances', fontsize=50, labelpad=20, fontweight='bold')
    ax.set_ylabel('Running Time (seconds)', fontsize=50, labelpad=20, fontweight='bold')
    
    # 设置y轴为科学计数法
    ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    # 加粗网格线
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='gray', linewidth=2.5)
    ax.set_axisbelow(True)
    
    # 加粗图例
    legend = ax.legend(loc='upper left', fontsize=42, frameon=True,
                      facecolor='white', edgecolor='black',
                      bbox_to_anchor=(0.02, 0.98),
                      borderpad=0.8,
                      handlelength=3,
                      handletextpad=0.5,
                      borderaxespad=0.5,
                      labelspacing=0.5)
    legend.get_frame().set_alpha(0.9)
    
    # 加粗坐标轴
    for spine in ax.spines.values():
        spine.set_linewidth(3.0)
        spine.set_color('#333333')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存高质量图表
    plt.savefig('single_evaluation_time.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

if __name__ == "__main__":
    plot_runtime_comparison()

