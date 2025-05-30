import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Ensure the input arrays are NumPy arrays
    prize = np.asarray(prize)
    weight = np.asarray(weight)
    
    # Calculate the weighted ratio for each item
    weighted_ratio = prize / weight.sum(axis=1)
    
    # Calculate the heuristic score based on the weighted ratio
    # Assuming that the heuristic is the weighted ratio itself, but this can be modified
    # based on the specific heuristic requirements of the problem
    heuristics = weighted_ratio
    
    return heuristics