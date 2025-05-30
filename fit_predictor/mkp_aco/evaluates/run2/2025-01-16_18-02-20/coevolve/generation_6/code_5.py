import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate value-to-weight ratio for each item
    value_to_weight = prize / weight.sum(axis=1)
    
    # Normalize the value-to-weight ratio using a dynamic approach
    normalized_value_to_weight = value_to_weight / value_to_weight.sum()
    
    # Initialize the heuristic array
    heuristics = np.zeros_like(prize)
    
    # Prioritize items with the highest normalized value-to-weight ratio
    sorted_indices = np.argsort(normalized_value_to_weight)[::-1]
    
    # Assign heuristic values based on sorted order
    for i in sorted_indices:
        heuristics[i] = 1.0
    
    return heuristics