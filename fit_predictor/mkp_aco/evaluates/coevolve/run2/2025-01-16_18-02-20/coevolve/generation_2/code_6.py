import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the normalized value for each item
    normalized_value = prize / weight.sum(axis=1, keepdims=True)
    
    # Initialize an empty array for the heuristics
    heuristics = np.zeros_like(prize)
    
    # Loop through each item to calculate its heuristic
    for i in range(prize.shape[0]):
        # Normalize the item's weight by summing all item weights in the same dimension
        normalized_weight = weight[i] / weight.sum(axis=0)
        
        # Calculate the heuristic by taking the dot product of normalized value and normalized weight
        heuristics[i] = np.dot(normalized_value[i], normalized_weight)
    
    return heuristics