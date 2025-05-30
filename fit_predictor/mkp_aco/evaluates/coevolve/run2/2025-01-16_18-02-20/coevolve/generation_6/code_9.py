import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Initialize an array to store the heuristic values
    heuristics = np.zeros_like(prize)
    
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight.sum(axis=1, keepdims=True)
    
    # Normalize the value-to-weight ratio to sum to 1
    normalized_ratio = value_to_weight_ratio / value_to_weight_ratio.sum(axis=0, keepdims=True)
    
    # Apply an adaptive stochastic sampling algorithm to select items based on the normalized ratio
    # Here, we use a simple random selection as a placeholder for the adaptive algorithm
    np.random.shuffle(normalized_ratio)
    
    # Update the heuristic values based on the selected items
    heuristics = normalized_ratio
    
    # Prioritize items by weight by multiplying the heuristic values by the inverse of the weight
    heuristics *= 1 / weight.sum(axis=1, keepdims=True)
    
    return heuristics