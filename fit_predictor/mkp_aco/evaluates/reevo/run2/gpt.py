import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    value_per_weight = prize / np.sum(weight, axis=1)
    diversity_factor = np.sum(weight, axis=1) / (np.linalg.norm(weight, axis=1) + 1e-8)
    normalized_prize = prize / np.sum(prize)
    balance_factor = np.mean(weight, axis=1) / np.max(weight, axis=1)
    sparsity = 1 / (np.linalg.norm(weight, axis=1) + 1e-8)
    heuristics = (value_per_weight * diversity_factor * normalized_prize * balance_factor * sparsity)
    sparsity_threshold = 0.1
    heuristics[heuristics < sparsity_threshold] = 0
    heuristics = heuristics / np.max(heuristics)
    return heuristics
