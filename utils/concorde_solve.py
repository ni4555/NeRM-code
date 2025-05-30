from concorde.tsp import TSPSolver
from typing import *
import numpy as np

def tsp(dots: List[Tuple[float, float]]) -> Tuple[List[int], float]:
    """
    TSP最优算法（concorde）
    :param dots: 一系列的点的坐标，点之间的距离表示代价
    :return: (tour, tour_length) 其中tour是点的编号序列，tour_length是路径总长度
    """
    xs = []
    ys = []
    for (x, y) in dots:
        xs.append(int(x * 1000))
        ys.append(int(y * 1000))
    solver = TSPSolver.from_data(xs, ys, norm="EUC_2D")
    solution = solver.solve()
    tour = solution.tour.tolist()
    
    # 计算实际的路径长度（使用原始浮点数坐标）
    tour_length = 0.0
    for i in range(len(tour)):
        j = (i + 1) % len(tour)
        x1, y1 = dots[tour[i]]
        x2, y2 = dots[tour[j]]
        tour_length += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    return tour, tour_length

def process_dataset(dataset_path: str, output_path: str):
    """
    处理数据集中的所有TSP实例并将结果写入文件
    :param dataset_path: .npy文件路径
    :param output_path: 输出结果的txt文件路径
    """
    # 加载数据集
    instances = np.load(dataset_path)
    
    total_length = 0.0
    n_instances = len(instances)
    
    with open(output_path, 'w') as f:
        # 为汇总信息预留第一行
        f.write("placeholder for summary\n")
        
        # 处理每个实例
        for i, instance in enumerate(instances):
            # 将numpy数组转换为坐标元组列表
            dots = [(float(x), float(y)) for x, y in instance]
            
            # 求解TSP
            tour, length = tsp(dots)
            
            total_length += length
            
            # 写入每个实例的结果
            f.write(f"Instance {i}: Length={length:.4f}\n")
            
        # 计算平均长度
        avg_length = total_length / n_instances
        
        # 返回文件开头写入汇总信息
        f.seek(0)
        f.write(f"Average tour length across {n_instances} instances: {avg_length:.4f}\n")

if __name__ == "__main__":
    dataset_path = "tsp200_dataset.npy"  # 请根据实际路径修改
    output_path = "concorde_results.txt"  # 请根据实际需求修改
    process_dataset(dataset_path, output_path)


