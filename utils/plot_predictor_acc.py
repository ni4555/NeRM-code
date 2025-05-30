# -*- coding: gbk -*-
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 设置全局字体和样式
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 42,
        'font.weight': 'bold',
        'axes.titlesize': 50,
        'axes.labelsize': 50,
        'xtick.labelsize': 45,
        'ytick.labelsize': 45,
        'legend.fontsize': 42,
        'axes.titleweight': 'bold',
        'axes.labelweight': 'bold',
    })

    # Data points
    sample_sizes = [1000, 3000, 6000, 9000, 12000, 15000]
    accuracies = [0.71, 0.76, 0.78, 0.80, 0.81, 0.81]

    # 创建更大尺寸的图表
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # 设置背景色
    ax.set_facecolor('#ffffff')
    fig.patch.set_facecolor('white')

    # 创建专业的线图
    line = ax.plot(sample_sizes, accuracies, 
                  marker='o', 
                  linewidth=3.5,
                  markersize=12,
                  color='#3498db',
                  markerfacecolor='white',
                  markeredgecolor='#2980b9',
                  markeredgewidth=3.0,
                  zorder=3)

    # 设置坐标轴范围和刻度
    ax.set_ylim(0.65, 0.85)
    ax.set_yticks([i/100 for i in range(65, 86, 5)])
    
    # 设置标签
    ax.set_xlabel('Number of Training Samples', fontsize=50, labelpad=20, fontweight='bold')
    ax.set_ylabel('Prediction Accuracy', fontsize=50, labelpad=20, fontweight='bold')

    # 加粗坐标轴和刻度
    ax.tick_params(axis='both', which='major', labelsize=45, width=3.0, length=12)
    for spine in ax.spines.values():
        spine.set_linewidth(3.0)
        spine.set_color('#333333')

    # 添加网格线
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='gray', zorder=1, linewidth=2.5)
    ax.set_axisbelow(True)

    # 添加数据点标签
    for x, y in zip(sample_sizes, accuracies):
        ax.annotate(f'{y:.2f}',
                   (x, y),
                   textcoords="offset points",
                   xytext=(0, 25),
                   ha='center',
                   fontsize=42,
                   fontweight='bold')

    # 调整布局
    plt.tight_layout()

    # 保存高质量图表
    plt.savefig('accuracy_vs_samples.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
