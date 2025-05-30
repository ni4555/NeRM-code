import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristics array with the same shape as the distance matrix
    heuristics = np.zeros_like(distance_matrix)
    
    # Fill the heuristics array with the inverse of the distance matrix elements
    # We use the np.finfo to get the smallest positive normal number
    # to avoid dividing by zero when the distance is very small
    epsilon = np.finfo(float).eps
    heuristics = 1 / (distance_matrix + epsilon)
    
    return heuristics