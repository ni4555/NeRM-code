import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the weighted ratio for each item
    weighted_ratio = prize / weight.sum(axis=1)
    
    # Calculate the cumulative sum for sorting
    cumulative_sum = np.cumsum(weighted_ratio)
    
    # Calculate the multi-dimensional weighted ratio metric
    multi_dimensional_weighted_ratio = weighted_ratio / cumulative_sum
    
    # Apply the dynamic sorting mechanism based on the multi-dimensional weighted ratio metric
    sorted_indices = np.argsort(-multi_dimensional_weighted_ratio)
    
    # Calculate the heuristics based on the sorted indices
    heuristics = np.zeros_like(prize)
    for i, index in enumerate(sorted_indices):
        heuristics[index] = i + 1
    
    return heuristics