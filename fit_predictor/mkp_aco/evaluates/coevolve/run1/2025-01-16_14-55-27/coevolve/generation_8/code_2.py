import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the average weight across all dimensions for each item
    avg_weight = np.mean(weight, axis=1)
    
    # Calculate the weighted ratio for each item
    weighted_ratio = prize / avg_weight
    
    # Sort the weighted ratios in descending order to prioritize items with higher ratios
    sorted_indices = np.argsort(weighted_ratio)[::-1]
    
    # Normalize the sorted weighted ratios to ensure they are on a comparable scale
    max_ratio = np.max(weighted_ratio)
    normalized_ratios = weighted_ratio / max_ratio
    
    # Return the normalized ratios, which represent the heuristics for each item
    return normalized_ratios[sorted_indices]