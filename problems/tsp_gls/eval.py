import time

import numpy as np
import logging
from gen_inst import TSPInstance, load_dataset, dataset_conf, load_dataset_with_optimal
from gls import guided_local_search
from tqdm import tqdm

try:
    from gpt import heuristics_v2 as heuristics
except:
    try:
        from gpt import heuristics_v1 as heuristics
    except:
        from gpt import heuristics

perturbation_moves = 30
iter_limit = 1200

def calculate_cost(inst: TSPInstance, path: np.ndarray) -> float:
    return inst.distmat[path, np.roll(path, 1)].sum().item()

def solve(inst: TSPInstance) -> float:
    heu = heuristics(inst.distmat.copy())
    assert tuple(heu.shape) == (inst.n, inst.n)
    result = guided_local_search(inst.distmat, heu, perturbation_moves, iter_limit)
    # print(result)
    return calculate_cost(inst, result)

if __name__ == "__main__":
    import sys
    import os

    print("[*] Running ...")

    problem_size = int(sys.argv[1])
    mood = sys.argv[3]
    assert mood in ['train', 'val', "test"]

    basepath = os.path.dirname(__file__)
    # automacially generate dataset if nonexists
    if not os.path.isfile(os.path.join(basepath, f"dataset/train{dataset_conf['train'][0]}_dataset.npy")):
        from gen_inst import generate_datasets
        generate_datasets()

    if mood == 'train':
        dataset_path = os.path.join(basepath, f"dataset/{mood}{problem_size}_dataset.npy")
        dataset = load_dataset(dataset_path)

        print(f"[*] Dataset loaded: {dataset_path} with {len(dataset)} instances.")

        objs = []
        for i, instance in enumerate(dataset):
            obj = solve(instance)
            print(f"[*] Instance {i}: {obj}")
            objs.append(obj)

        print("[*] Average:")
        print(np.mean(objs))

    else: # mood == 'val'
        for problem_size in dataset_conf['val']:
            dataset_path = os.path.join(basepath, f"dataset/{mood}{problem_size}_dataset.npy")
            dataset = load_dataset(dataset_path)
            logging.info(f"[*] Evaluating {dataset_path}")

            objs = []
            for i, instance in enumerate(tqdm(dataset)):
                obj = solve(instance)
                objs.append(obj)

            print(f"[*] Average for {problem_size}: {np.mean(objs)}")

# if __name__ == "__main__":
#     import os
#
#     print("[*] Running ...")
#
#     tsp_test_list = ['TSP20', 'TSP50', 'TSP100']
#
#
#     basepath = os.path.dirname(__file__)
#
#     for problem_size in tsp_test_list:
#         dataset_path = os.path.join(basepath, f"dataset/{problem_size}.pkl")
#         dataset = load_dataset_with_optimal(dataset_path)
#         logging.info(f"[*] Evaluating {dataset_path}")
#
#
#
#         objs = np.zeros(len(dataset))
#         gaps = np.zeros(len(dataset))
#         global_objs = np.zeros(len(dataset))
#         start_t = time.time()
#         for i, instance in enumerate(tqdm(dataset)):
#             obj = solve(instance)
#             global_objs[i] = dataset[i].opt_cost
#             objs[i] = obj
#             gaps[i] = round(((obj / instance.opt_cost) - 1) * 100, 5)
#
#         time_spend = time.time() - start_t
#         result = (
#             f"Average gap is: {np.mean(gaps):7.5f} timecost: {time_spend:7.3f}; Average global distance: {np.mean(global_objs):.5f}, local distance: {np.mean(objs):.5f}")
#
#         result_file_name = basepath + 'results_gls_' + problem_size + '_3' + '.txt'
#         with open(result_file_name, "w") as file:
#             file.write(result + "\n")
#             for gap_index in range(len(gaps)):
#                 file.write(str(gaps[gap_index]) + "\t" + str(global_objs[gap_index]) + "\t" + str(
#                     objs[gap_index]) + "\n")
#
#         print(f"[*] Average for {problem_size}: {np.mean(gaps)}")
