# -*- coding: gbk -*-
import matplotlib.pyplot as plt
import numpy as np

def plot_diff_llm():
    # 数据
    # models = ['gpt-3.5', 'gpt-4', 'llama', 'gemini-pro', 'deepseek', 'glm-4-flash']
    models = ['GPT-3.5-Turbo', 'GPT-4-Turbo', 'Gemini-1.5-pro', 'DeepSeek-V3', 'GLM-4-Flash']
    tsp_sizes = ['TSP200', 'TSP300', 'TSP400', 'TSP500']
    gaps = {
        'GPT-3.5-Turbo':    [0.20802939, 0.634700546, 0.897561601, 1.097829191],
        'GPT-4-Turbo':      [0.17545784, 0.402285099, 0.551276518, 0.716249319],
        # 'llama':      [0.21999217, 0.286583115, 0.547156325, 0.721671743],
        'Gemini-1.5-pro': [0.22445056, 0.334845456, 0.571702159, 0.770362236],
        'DeepSeek-V3':   [0.20484482, 0.417568814, 0.518931312, 0.69465501],
        'GLM-4-Flash':[0.19100032, 0.348531447, 0.65993904, 0.770018112]
    }

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建图表
    plt.figure(figsize=(12, 8))

    # 颜色方案 - 使用更鲜明的颜色
    colors = ['#FF0000', '#0066CC', '#00CC00', '#FF6600', '#9933CC']  # 红、蓝、绿、橙、紫
    markers = ['o', 's', '^', 'D', 'v']

    # 绘制每个模型的折线
    for i, (model, color, marker) in enumerate(zip(models, colors, markers)):
        plt.plot(tsp_sizes, gaps[model],
                 label=model,
                 color=color,
                 marker=marker,
                 linewidth=2.5,        # 增加线条粗细
                 markersize=10,        # 增大标记大小
                 linestyle='-')

    # 设置图表样式
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Problem Sizes', fontsize=16)      # 进一步增大字号
    plt.ylabel('Gap (%)', fontsize=16)            # 进一步增大字号
    plt.title('Gap Comparison of Different LLMs on TSP Problems', fontsize=18, pad=15)  # 进一步增大标题字号

    # 设置图例位置到左上角，调整透明度
    plt.legend(loc='upper left', fontsize=14, framealpha=0.7, edgecolor='gray')

    # 设置刻度字号
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # 调整布局
    plt.tight_layout()

    # 保存图表
    plt.savefig('llm_tsp_comp.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    plot_diff_llm()
