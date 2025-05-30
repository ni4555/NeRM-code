import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight.sum(axis=1)
    
    # Sort items based on their value-to-weight ratio in descending order
    sorted_indices = np.argsort(value_to_weight_ratio)[::-1]
    
    # Initialize the heuristics array with zeros
    heuristics = np.zeros_like(prize)
    
    # Iterate over the sorted indices and assign a higher heuristic value to the top items
    for index in sorted_indices:
        heuristics[index] = 1  # Set the heuristic to 1 for the top item, indicating its high priority
    
    return heuristics