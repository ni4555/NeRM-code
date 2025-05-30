
import importlib.util
import time

from aco import ACO
import glob
import numpy as np
import torch
import logging
import sys
sys.path.insert(0, "../../../")

import gpt
from utils.utils import get_heuristic_name


possible_func_names = ["heuristics", "heuristics_v1", "heuristics_v2", "heuristics_v3"]

heuristic_name = get_heuristic_name(gpt, possible_func_names)
heuristics = getattr(gpt, heuristic_name)


N_ITERATIONS = 50
N_ANTS = 10

def solve(prize: np.ndarray, weight: np.ndarray):
    n, m = weight.shape
    heu = heuristics(prize.copy(), weight.copy()) + 1e-9
    assert heu.shape == (n,)
    heu[heu < 1e-9] = 1e-9
    aco = ACO(torch.from_numpy(prize), torch.from_numpy(weight), torch.from_numpy(heu), N_ANTS)
    obj, _ = aco.run(N_ITERATIONS)
    return obj


# if __name__ == "__main__":
#     import sys
#     import os
#
#     print("[*] Running ...")
#
#     problem_size = int(sys.argv[1])
#     root_dir = sys.argv[2]
#     mood = sys.argv[3]
#     assert mood in ['train', 'val']
#
#     basepath = os.path.dirname(__file__)
#     # automacially generate dataset if nonexists
#     if not os.path.isfile(os.path.join(basepath, f"dataset/train50_dataset.npz")):
#         from gen_inst import generate_datasets
#         generate_datasets()
#
#     if mood == 'train':
#         dataset_path = os.path.join(basepath, f"dataset/{mood}{problem_size}_dataset.npz")
#         dataset = np.load(dataset_path)
#         prizes, weights = dataset['prizes'], dataset['weights']
#         n_instances = prizes.shape[0]
#
#         print(f"[*] Dataset loaded: {dataset_path} with {n_instances} instances.")
#
#         objs = []
#         for i, (prize, weight) in enumerate(zip(prizes, weights)):
#             obj = solve(prize, weight)
#             print(f"[*] Instance {i}: {obj}")
#             objs.append(obj.item())
#
#         print("[*] Average:")
#         print(np.mean(objs))
#
#     else: # mood == 'val'
#         for problem_size in [100, 300, 500]:
#             dataset_path = os.path.join(basepath, f"dataset/{mood}{problem_size}_dataset.npz")
#             dataset = np.load(dataset_path)
#             prizes, weights = dataset['prizes'], dataset['weights']
#             n_instances = prizes.shape[0]
#             logging.info(f"[*] Evaluating {dataset_path}")
#
#             objs = []
#             for i, (prize, weight) in enumerate(zip(prizes, weights)):
#                 obj = solve(prize, weight)
#                 objs.append(obj.item())
#
#             print(f"[*] Average for {problem_size}: {np.mean(objs)}")
def load_heuristic_from_file(file_path):
    """动态加载Python文件中的heuristics函数"""
    spec = importlib.util.spec_from_file_location("module", file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["module"] = module
    spec.loader.exec_module(module)
    return module.heuristics_v2


if __name__ == "__main__":
    import sys
    import os

    print("[*] Running ...")

    problem_size_list = [100, 300, 500]
    mood = 'val'
    assert mood in ['train', 'val']

    # 获取/final/目录下的所有Python文件
    basepath = os.path.dirname(__file__)
    final_dir = os.path.join(basepath, "final")
    python_files = glob.glob(os.path.join(final_dir, "*.py"))

    # 对每个Python文件进行测试
    for py_file in python_files:
        heuristic_name = os.path.basename(py_file)[:-3]  # 移除.py后缀
        print(f"\n[*] Testing heuristic: {heuristic_name}")
        results = {}

        try:
            heuristic_func = load_heuristic_from_file(py_file)
        except Exception as e:
            print(f"Error loading {py_file}: {str(e)}")
            continue


        for problem_size in problem_size_list:
            dataset_path = os.path.join(basepath, f"dataset/{mood}{problem_size}_dataset.npz")
            dataset = np.load(dataset_path)
            prizes, weights = dataset['prizes'], dataset['weights']
            n_instances = prizes.shape[0]
            logging.info(f"[*] Evaluating {dataset_path}")


            start_t = time.time()
            objs = []
            for i, (prize, weight) in enumerate(zip(prizes, weights)):
                obj = solve(prize, weight)
                objs.append(obj.item())

            time_spend = time.time() - start_t

            results[problem_size] = f"Average obj is: {np.mean(objs):7.5f} timecost: {time_spend:7.3f}, "

            print(f"[*] Average for {problem_size}: {np.mean(objs)}")

        # 使用启发式名称和问题规模来命名结果文件
        result_file_name = os.path.join(
            basepath,
            f'results_aco_{heuristic_name}.txt'
        )

        with open(result_file_name, "w") as file:
            for problem_size, sentence in results.items():
                file.write(f"{problem_size}: {sentence}\n")


