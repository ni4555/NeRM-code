import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Initialize an array to store heuristic values
    heuristics = np.zeros_like(prize)
    
    # Calculate the heuristic value for each item
    # Since the weight constraint is fixed to 1 for each dimension,
    # the heuristic can be the ratio of prize to the sum of weights across dimensions
    heuristics = prize / np.sum(weight, axis=1)
    
    return heuristics