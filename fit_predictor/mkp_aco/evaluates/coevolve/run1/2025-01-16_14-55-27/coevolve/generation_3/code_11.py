import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the multi-dimensional weighted ratio for each item
    weighted_prize = np.sum(prize * weight, axis=1)
    total_weight = np.sum(weight, axis=1)
    
    # Ensure that the total weight of each item is 1 by normalizing
    normalized_weighted_prize = weighted_prize / total_weight
    
    # Use cumulative sum analysis to assess the contribution of each item
    cumulative_sum = np.cumsum(normalized_weighted_prize)
    
    # Calculate the multi-dimensional weighted ratio metric
    multi_dimensional_weighted_ratio = cumulative_sum / np.arange(1, len(cumulative_sum) + 1)
    
    # Dynamic sorting mechanism based on the multi-dimensional weighted ratio metric
    sorted_indices = np.argsort(multi_dimensional_weighted_ratio)
    
    # Apply the sorting indices to the prize array to create the heuristics
    heuristics = np.zeros_like(prize)
    heuristics[sorted_indices] = multi_dimensional_weighted_ratio[sorted_indices]
    
    return heuristics