import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight.sum(axis=1)
    
    # Normalize the ratios to a 0-1 scale
    normalized_ratios = (value_to_weight_ratio - value_to_weight_ratio.min()) / (value_to_weight_ratio.max() - value_to_weight_ratio.min())
    
    # Rank the items based on the normalized value-to-weight ratio
    ranked_indices = np.argsort(normalized_ratios)[::-1]
    
    # Create the heuristics array where the rank is used as a heuristic value
    heuristics = np.zeros(len(prize))
    heuristics[ranked_indices] = np.arange(1, len(ranked_indices) + 1)
    
    return heuristics