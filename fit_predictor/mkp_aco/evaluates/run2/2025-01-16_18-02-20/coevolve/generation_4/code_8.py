import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight
    
    # Sort items by value-to-weight ratio in descending order
    sorted_indices = np.argsort(value_to_weight_ratio)[::-1]
    
    # Normalize the sorted indices by dividing by the number of items
    normalized_indices = sorted_indices / len(sorted_indices)
    
    # Generate a random permutation of the normalized indices
    permutation = np.random.permutation(normalized_indices)
    
    # Create a binary heuristic array where higher values indicate higher priority
    heuristics = np.zeros(len(prize))
    heuristics[permutation > 0.5] = 1
    
    return heuristics