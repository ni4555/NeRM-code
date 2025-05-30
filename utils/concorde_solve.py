from concorde.tsp import TSPSolver
from typing import *
import numpy as np

def tsp(dots: List[Tuple[float, float]]) -> Tuple[List[int], float]:
    """
    TSP�����㷨��concorde��
    :param dots: һϵ�еĵ�����꣬��֮��ľ����ʾ����
    :return: (tour, tour_length) ����tour�ǵ�ı�����У�tour_length��·���ܳ���
    """
    xs = []
    ys = []
    for (x, y) in dots:
        xs.append(int(x * 1000))
        ys.append(int(y * 1000))
    solver = TSPSolver.from_data(xs, ys, norm="EUC_2D")
    solution = solver.solve()
    tour = solution.tour.tolist()
    
    # ����ʵ�ʵ�·�����ȣ�ʹ��ԭʼ���������꣩
    tour_length = 0.0
    for i in range(len(tour)):
        j = (i + 1) % len(tour)
        x1, y1 = dots[tour[i]]
        x2, y2 = dots[tour[j]]
        tour_length += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    return tour, tour_length

def process_dataset(dataset_path: str, output_path: str):
    """
    �������ݼ��е�����TSPʵ���������д���ļ�
    :param dataset_path: .npy�ļ�·��
    :param output_path: ��������txt�ļ�·��
    """
    # �������ݼ�
    instances = np.load(dataset_path)
    
    total_length = 0.0
    n_instances = len(instances)
    
    with open(output_path, 'w') as f:
        # Ϊ������ϢԤ����һ��
        f.write("placeholder for summary\n")
        
        # ����ÿ��ʵ��
        for i, instance in enumerate(instances):
            # ��numpy����ת��Ϊ����Ԫ���б�
            dots = [(float(x), float(y)) for x, y in instance]
            
            # ���TSP
            tour, length = tsp(dots)
            
            total_length += length
            
            # д��ÿ��ʵ���Ľ��
            f.write(f"Instance {i}: Length={length:.4f}\n")
            
        # ����ƽ������
        avg_length = total_length / n_instances
        
        # �����ļ���ͷд�������Ϣ
        f.seek(0)
        f.write(f"Average tour length across {n_instances} instances: {avg_length:.4f}\n")

if __name__ == "__main__":
    dataset_path = "tsp200_dataset.npy"  # �����ʵ��·���޸�
    output_path = "concorde_results.txt"  # �����ʵ�������޸�
    process_dataset(dataset_path, output_path)


