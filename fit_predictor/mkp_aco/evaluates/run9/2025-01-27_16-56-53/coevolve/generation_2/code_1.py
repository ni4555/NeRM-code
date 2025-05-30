import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    m = weight.shape[1]
    if m != 1:
        raise ValueError("Dimension of weights must be 1 for this heuristic")
    
    # Calculate the normalized weights based on the prize
    normalized_weights = prize / np.sum(prize)
    
    # Calculate the heuristic for each item
    heuristics = normalized_weights * np.sum(weight, axis=1)
    
    return heuristics
