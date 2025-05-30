import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the multi-dimensional weighted ratio metric
    # Since constraint of each dimension is fixed to 1, the metric is simply the ratio of prize to weight
    weighted_ratio = prize / weight
    
    # Dynamic sorting mechanism based on the multi-dimensional weighted ratio metric
    # We use the cumulative sum of the sorted indices to achieve the dynamic sorting
    sorted_indices = np.argsort(weighted_ratio)
    cumulative_sum = np.cumsum(np.ones(len(sorted_indices)))
    
    # Calculate the heuristic for each item based on the cumulative sum
    heuristics = np.array([cumulative_sum[idx] for idx in sorted_indices])
    
    return heuristics