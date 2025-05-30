import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n, m = prize.shape
    heuristics = np.zeros(n)
    
    # Calculate a simple heuristic based on the ratio of prize to weight
    for i in range(n):
        item_ratio = prize[i] / np.sum(weight[i])
        heuristics[i] = item_ratio
    
    # Introduce adaptive dynamic knapsack weight partitioning
    total_weight = np.sum(weight, axis=1)
    for i in range(n):
        heuristics[i] *= (1 - (total_weight[i] / np.sum(total_weight)))
    
    # Apply intelligent heuristic-based sampling
    sorted_indices = np.argsort(heuristics)[::-1]
    sample_indices = sorted_indices[:np.sum(heuristics > 0)]
    
    heuristics[sample_indices] *= 2
    
    return heuristics
