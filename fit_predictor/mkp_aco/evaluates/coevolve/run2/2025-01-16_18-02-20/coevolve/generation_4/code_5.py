import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight
    
    # Sort items based on their value-to-weight ratio in descending order
    sorted_indices = np.argsort(value_to_weight_ratio)[::-1]
    
    # Initialize the heuristics array with zeros
    heuristics = np.zeros_like(prize)
    
    # Iterate over sorted items and assign a higher heuristic value to high-value items
    for i, sorted_index in enumerate(sorted_indices):
        heuristics[sorted_index] = 1.0 / (i + 1)  # Use inverse rank as heuristic value
    
    return heuristics