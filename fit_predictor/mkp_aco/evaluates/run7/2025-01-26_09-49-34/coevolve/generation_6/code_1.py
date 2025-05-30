import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n, m = prize.shape
    # Calculate normalized value-to-weight ratio for each item
    value_to_weight = prize / weight
    # Normalize this ratio to make it suitable for probabilistic selection
    normalized_ratio = (value_to_weight - np.min(value_to_weight)) / (np.max(value_to_weight) - np.min(value_to_weight))
    # Initialize heuristic array with normalized ratios
    heuristics = normalized_ratio.copy()
    # Adjust heuristics based on feasibility considering remaining knapsack capacities
    remaining_capacities = np.ones_like(weight)  # Assuming each knapsack has infinite capacity for simplicity
    for i in range(n):
        for j in range(m):
            remaining_capacities[j] -= weight[i, j]
        feasibility = np.sum(remaining_capacities > 0) == m
        heuristics[i] *= feasibility
    return heuristics
