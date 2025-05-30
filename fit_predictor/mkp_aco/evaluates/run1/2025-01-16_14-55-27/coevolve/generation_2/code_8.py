import numpy as np
import numpy as np
import random

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n = prize.shape[0]
    m = weight.shape[1]
    heuristic_values = np.zeros(n)

    # Function to calculate the weighted ratio for each item
    def weighted_ratio(item_index):
        total_weight = weight[item_index].sum()
        return prize[item_index] / total_weight if total_weight > 0 else 0

    # Adaptive stochastic sampling to estimate the heuristics
    num_samples = max(1, n // 10)  # Adjust the number of samples based on problem scale
    for _ in range(num_samples):
        # Sample items randomly
        sampled_indices = np.random.choice(n, n, replace=False)
        sampled_prizes = prize[sampled_indices]
        sampled_weights = weight[sampled_indices]

        # Calculate weighted ratio for each sampled item
        sampled_ratios = np.apply_along_axis(weighted_ratio, 1, sampled_indices)

        # Calculate heuristic for each item based on sampled ratios
        for i in range(n):
            if i in sampled_indices:
                heuristic_values[i] += np.mean(sampled_ratios[sampled_indices == i])
            else:
                heuristic_values[i] += np.mean(sampled_ratios[sampled_indices])

    # Normalize heuristics to sum to 1
    heuristic_sum = heuristic_values.sum()
    heuristic_values /= heuristic_sum

    return heuristic_values