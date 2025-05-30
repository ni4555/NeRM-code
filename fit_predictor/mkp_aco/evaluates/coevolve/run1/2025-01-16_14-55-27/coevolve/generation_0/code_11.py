import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Initialize the heuristics array with zeros
    heuristics = np.zeros(prize.shape[0])
    
    # Assuming the "promise" of an item is proportional to its prize-to-weight ratio
    # and that the weights are the same across all dimensions (fixed to 1),
    # the promise can be directly calculated as the prize value of each item.
    heuristics = prize / weight.sum(axis=1)
    
    # Return the calculated heuristics
    return heuristics