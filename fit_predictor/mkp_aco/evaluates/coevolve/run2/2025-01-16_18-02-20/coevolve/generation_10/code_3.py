import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight
    
    # Sort items based on their value-to-weight ratio in descending order
    sorted_indices = np.argsort(value_to_weight_ratio)[::-1]
    
    # Create the heuristics array where the higher the value-to-weight ratio, the higher the score
    heuristics = np.zeros_like(prize)
    heuristics[sorted_indices] = value_to_weight_ratio[sorted_indices]
    
    return heuristics