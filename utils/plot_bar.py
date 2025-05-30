# -*- coding: gbk -*-
import matplotlib.pyplot as plt
import numpy as np


def plot_runtime_comparison():
    # ����ȫ�������С�ʹ�ϸ����һ�������ֺ�
    # plt.rcParams.update({
    #     'font.family': 'Times New Roman',  # ����Times New Roman����
    #     'font.size': 38,  # ������������С (��32����38)
    #     'font.weight': 'bold',
    #     'axes.titlesize': 42,  # ������������С (��36����42)
    #     'axes.labelsize': 40,  # �������ǩ�����С (��34����40)
    #     'xtick.labelsize': 38,  # ����x��̶ȱ�ǩ��С (��32����38)
    #     'ytick.labelsize': 38,  # ����y��̶ȱ�ǩ��С (��32����38)
    #     'legend.fontsize': 38,  # ����ͼ�������С (��32����38)
    #     'axes.titleweight': 'bold',
    #     'axes.labelweight': 'bold',
    # })

    plt.rcParams.update({
        'font.family': 'Times New Roman',  # ����Times New Roman����
        'font.size': 42,  # ������������С
        'font.weight': 'bold',
        'axes.titlesize': 50,  # ������������С
        'axes.labelsize': 50,  # �������ǩ�����С
        'xtick.labelsize': 45,  # ����x��̶ȱ�ǩ��С
        'ytick.labelsize': 45,  # ����y��̶ȱ�ǩ��С
        'legend.fontsize': 42,  # ����ͼ�������С
        'axes.titleweight': 'bold',
        'axes.labelweight': 'bold',
    })
    
    # �����ģ����
    problem_sizes = ['200', '300', '400', '500', '1000']

    # ����ʱ������
    runtime_1 = [765, 871.56, 1106.44, 1529.61, 1918.53]  # MoH����
    runtime_2 = [1033, 1529.61, 4002.78, 11505.23, 19723]  # ReEvo����
    runtime_3 = [3135, 7562.85, 19723, 36228, 115505.23]  # EoH����

    # ��������ߴ��ͼ��
    fig, ax = plt.subplots(figsize=(16, 12))  # ����ͼ�γߴ� (��14,10����16,12)
    
    # ���ñ���ɫ
    ax.set_facecolor('#ffffff')
    fig.patch.set_facecolor('white')

    # ������״ͼ��λ��
    x = np.arange(len(problem_sizes))
    width = 0.25

    # ����רҵ����ɫ����
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    edge_colors = ['#2980b9', '#c0392b', '#27ae60']

    # ����������״ͼ���������ӿ�Ⱥͱ߿��ϸ
    bars1 = ax.bar(x - width, runtime_1, width, label='NeRM',
                   color=colors[0], alpha=0.8, edgecolor=edge_colors[0], 
                   linewidth=2.5, zorder=2)  # ���ӱ߿��ϸ
    bars2 = ax.bar(x, runtime_2, width, label='ReEvo',
                   color=colors[1], alpha=0.8, edgecolor=edge_colors[1], 
                   linewidth=2.5, zorder=2)
    bars3 = ax.bar(x + width, runtime_3, width, label='EoH',
                   color=colors[2], alpha=0.8, edgecolor=edge_colors[2], 
                   linewidth=2.5, zorder=2)

    # # ���������
    # ax.plot(x - width, runtime_1, '-', color=edge_colors[0], linewidth=2,
    #        marker='o', markersize=6, markerfacecolor='white', markeredgewidth=1.5,
    #        markeredgecolor=edge_colors[0], zorder=3)
    # ax.plot(x, runtime_2, '-', color=edge_colors[1], linewidth=2,
    #        marker='s', markersize=6, markerfacecolor='white', markeredgewidth=1.5,
    #        markeredgecolor=edge_colors[1], zorder=3)
    # ax.plot(x + width, runtime_3, '-', color=edge_colors[2], linewidth=2,
    #        marker='^', markersize=6, markerfacecolor='white', markeredgewidth=1.5,
    #        markeredgecolor=edge_colors[2], zorder=3)

    # �ڴ���ͼ�������y������
    ax.set_ylim(top=60000)

    # ����x���ǩ��ʹ�ø����ֺ�
    ax.set_xticks(x)
    ax.set_xticklabels(problem_sizes, fontsize=38, fontweight='bold')  # ��32����38
    ax.tick_params(axis='both', which='major', labelsize=38, width=3.0, length=12)

    # ���ñ���ͱ�ǩ��ʹ�ø����ֺ�
    # ax.set_title('Time consumed for different problem sizes',
    #              fontsize=42, pad=25, fontweight='bold')  # ��36����42
    ax.set_xlabel('Problem Size of TSP instances', fontsize=50, labelpad=20, fontweight='bold')  # ��34����40
    ax.set_ylabel('Running Time (seconds)', fontsize=50, labelpad=20, fontweight='bold')  # �޸�y���ǩ

    # ����y��Ϊ��ѧ������
    ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    # �Ӵ�������
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='gray', zorder=1, linewidth=2.5)  # ���������߿��
    ax.set_axisbelow(True)

    # �Ӵ�ͼ������С�߾�ͱ�������
    legend = ax.legend(loc='upper left', fontsize=38, frameon=True,
                      facecolor='white', edgecolor='black',
                      bbox_to_anchor=(0.02, 0.98),
                      borderpad=0.8,  # ��С�߿�padding (��2.0����0.8)
                      handlelength=3,  # ��Сͼ����ǳ��� (��4����3)
                      handletextpad=0.5,  # ��С�ı���� (��1.0����0.5)
                      borderaxespad=0.5,  # ��С��߾�
                      labelspacing=0.5)  # ��С��ǩ���
    legend.get_frame().set_alpha(0.9)

    # �Ӵ�������
    for spine in ax.spines.values():
        spine.set_linewidth(3.0)  # ��2.5����3.0
        spine.set_color('#333333')

    # ��������
    plt.tight_layout()

    # ���������ͼ��
    plt.savefig('single_run_time.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


if __name__ == "__main__":
    plot_runtime_comparison()

