# -*- coding: gbk -*-
import time
import glob
import importlib.util
import os
import sys

import numpy as np
import logging
from gen_inst import TSPInstance, load_dataset, dataset_conf, load_dataset_with_optimal
from gls import guided_local_search
from tqdm import tqdm

def calculate_cost(inst: TSPInstance, path: np.ndarray) -> float:
    return inst.distmat[path, np.roll(path, 1)].sum().item()

def load_heuristic_from_file(file_path):
    """动态加载Python文件中的heuristics函数"""
    spec = importlib.util.spec_from_file_location("module", file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["module"] = module
    spec.loader.exec_module(module)
    return module.heuristics_v2

def solve_with_heuristic(inst: TSPInstance, heuristic_func) -> float:
    """使用指定的启发式函数求解TSP实例"""
    heu = heuristic_func(inst.distmat.copy())
    assert tuple(heu.shape) == (inst.n, inst.n)
    result = guided_local_search(inst.distmat, heu, perturbation_moves, iter_limit)
    return calculate_cost(inst, result)

if __name__ == "__main__":
    print("[*] Running ...")

    # 配置参数
    test_inst_num = 64
    # tsp_test_list = ['TSP20', 'TSP50', 'TSP100', 'TSP200', 'TSP500', 'TSP1000']
    tsp_test_list = ['TSP200', ]
    perturbation_moves_list = {'TSP20': 5, 'TSP50': 30, 'TSP100': 40, 'TSP200': 40, 'TSP500': 80, 'TSP1000': 80}
    iter_max_dict = {'TSP20': 200, 'TSP50': 500, 'TSP100': 2000, 'TSP200': 800, 'TSP500': 800, 'TSP1000': 800}



    basepath = os.path.dirname(__file__)
    
    # 获取/final/目录下的所有Python文件
    final_dir = os.path.join(basepath, "final")
    python_files = glob.glob(os.path.join(final_dir, "*.py"))



    # 对每个问题规模进行测试
    for problem_size in tsp_test_list:

        # 对每个Python文件进行测试
        for py_file in python_files:
            heuristic_name = os.path.basename(py_file)[:-3]  # 移除.py后缀
            print(f"\n[*] Testing heuristic: {heuristic_name}")

            try:
                heuristic_func = load_heuristic_from_file(py_file)
            except Exception as e:
                print(f"Error loading {py_file}: {str(e)}")
                continue


            iter_limit = iter_max_dict[problem_size]
            perturbation_moves = perturbation_moves_list[problem_size]

            dataset_path = os.path.join(basepath, f"dataset/{problem_size}.pkl")
            try:
                dataset = load_dataset_with_optimal(dataset_path, test_inst_num)
            except Exception as e:
                print(f"Error loading dataset {dataset_path}: {str(e)}")
                continue

            logging.info(f"[*] Evaluating {dataset_path}")

            objs = np.zeros(len(dataset))
            gaps = np.zeros(len(dataset))
            global_objs = np.zeros(len(dataset))
            total_solve_time = 0
            start_t = time.time()

            for i, instance in enumerate(tqdm(dataset)):
                try:
                    pure_solve_time = time.time()
                    obj = solve_with_heuristic(instance, heuristic_func)
                    pure_end_time = time.time()
                    total_solve_time += pure_end_time - pure_solve_time
                    global_objs[i] = dataset[i].opt_cost
                    objs[i] = obj
                    gaps[i] = round(((obj / instance.opt_cost) - 1) * 100, 5)
                except Exception as e:
                    print(f"Error solving instance {i}: {str(e)}")
                    continue

            time_spend = time.time() - start_t
            result = (
                f"Average gap is: {np.mean(gaps):7.5f} timecost: {time_spend:7.3f}, "
                f"pure solve time cost: {total_solve_time: 7.3f}; "
                f"Average global distance: {np.mean(global_objs):.5f}, "
                f"local distance: {np.mean(objs):.5f}")

            # 使用启发式名称和问题规模来命名结果文件
            result_file_name = os.path.join(
                basepath,
                f'results_gls_{problem_size}_{heuristic_name}.txt'
            )

            with open(result_file_name, "w") as file:
                file.write(result + "\n")
                for gap_index in range(len(gaps)):
                    file.write(f"{gaps[gap_index]}\t{global_objs[gap_index]}\t{objs[gap_index]}\n")

            print(f"[*] Average for {problem_size} using {heuristic_name}: {np.mean(gaps)}")
