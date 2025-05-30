import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate value-to-weight ratio for each item
    value_to_weight = prize / weight.sum(axis=1, keepdims=True)
    
    # Normalize the value-to-weight ratio to get a probability
    probabilities = value_to_weight / value_to_weight.sum(axis=0, keepdims=True)
    
    # Apply a state-of-the-art prioritization framework
    # For simplicity, we'll use the sum of probabilities as a heuristic
    heuristics = probabilities.sum(axis=1)
    
    return heuristics