import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n, m = prize.shape[0], weight.shape[1]
    value_to_weight_ratio = prize / weight
    heuristic = np.zeros(n)
    
    for i in range(n):
        # Start with the maximum value-to-weight ratio
        max_ratio = value_to_weight_ratio[i]
        # Initialize the sum of the best ratios for each dimension
        sum_best_ratios = 0
        # Iterate over each dimension
        for j in range(m):
            # Calculate the best ratio for the current dimension
            best_ratio = max(value_to_weight_ratio[i, :])
            # Update the sum of the best ratios
            sum_best_ratios += best_ratio
            # Normalize the ratio by the sum of the best ratios
            value_to_weight_ratio[i, j] /= best_ratio
        # Calculate the heuristic for the item by taking the mean of the normalized ratios
        heuristic[i] = sum_best_ratios / m
    
    return heuristic