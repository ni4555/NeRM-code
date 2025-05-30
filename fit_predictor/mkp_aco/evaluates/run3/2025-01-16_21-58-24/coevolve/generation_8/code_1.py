import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n, m = weight.shape
    heuristics = np.zeros(n)
    
    # Calculate heuristic based on the sum of weights in each dimension
    for i in range(n):
        weight_sum = np.sum(weight[i])
        # Normalize by the dimension constraint (fixed to 1)
        heuristics[i] = prize[i] / weight_sum
    
    # Normalize heuristics to be between 0 and 1
    heuristics = heuristics / np.sum(heuristics)
    
    return heuristics