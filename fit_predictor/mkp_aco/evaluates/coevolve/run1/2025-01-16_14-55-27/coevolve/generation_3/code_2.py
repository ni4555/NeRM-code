import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the multi-dimensional weighted ratio metric
    weighted_ratio = np.sum(prize * weight, axis=1)
    
    # Normalize the weighted ratio by the sum of weights for each item
    normalized_weighted_ratio = weighted_ratio / np.sum(weight, axis=1)
    
    # Use cumulative sum analysis to assess item contribution
    cumulative_sum = np.cumsum(normalized_weighted_ratio)
    
    # Dynamic sorting mechanism based on the multi-dimensional weighted ratio metric
    sorted_indices = np.argsort(-cumulative_sum)
    
    # Create the heuristics array based on the sorted indices
    heuristics = np.zeros_like(prize)
    heuristics[sorted_indices] = cumulative_sum[sorted_indices]
    
    return heuristics