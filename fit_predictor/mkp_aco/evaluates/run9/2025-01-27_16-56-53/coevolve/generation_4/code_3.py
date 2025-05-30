import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Normalize weights for each dimension
    norm_weights = weight / weight.sum(axis=1, keepdims=True)
    
    # Calculate heuristic values for each item
    heuristics = (prize * norm_weights).sum(axis=1)
    
    # Normalize heuristic values to scale between 0 and 1
    heuristics /= heuristics.max()
    
    return heuristics
