import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Normalize the weights by taking the average across dimensions
    avg_weight = np.mean(weight, axis=1, keepdims=True)
    
    # Calculate the heuristic value for each item by combining the prize value and the normalized weight
    heuristics = prize / avg_weight
    
    # Optionally, you can further adjust the heuristic to make it more or less aggressive
    # For example, here we add a constant to avoid division by zero
    # heuristics += 1 / avg_weight
    
    return heuristics