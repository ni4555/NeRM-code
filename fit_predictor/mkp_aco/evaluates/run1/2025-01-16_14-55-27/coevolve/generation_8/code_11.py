import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n, m = prize.shape[0], weight.shape[1]
    
    # Calculate the total weight for each item
    total_weight = np.sum(weight, axis=1)
    
    # Calculate the weighted ratio for each item
    weighted_ratio = prize / total_weight
    
    # Apply a dynamic multi-criteria sorting
    # We will sort by weighted ratio in descending order
    # Since each dimension's constraint is fixed to 1, we can sort by any dimension's ratio
    sorted_indices = np.argsort(weighted_ratio)[::-1]
    
    # Normalize the heuristics based on the sorted indices
    # We will normalize by dividing by the sum of the sorted ratios
    sorted_ratios = weighted_ratio[sorted_indices]
    max_ratio = np.max(sorted_ratios)
    min_ratio = np.min(sorted_ratios)
    # Normalize to a range between 0 and 1
    normalized_ratios = (sorted_ratios - min_ratio) / (max_ratio - min_ratio)
    
    # The normalized ratios are now the heuristics
    heuristics = normalized_ratios[sorted_indices]
    
    return heuristics