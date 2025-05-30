import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize a matrix of the same shape as the distance matrix to store heuristics
    heuristics = np.zeros_like(distance_matrix)
    
    # Manhattan distance heuristic
    heuristics += np.abs(np.diff(distance_matrix, axis=0)).sum(axis=1)
    heuristics += np.abs(np.diff(distance_matrix, axis=1)).sum(axis=0)
    
    # Direct distance matrix usage as a heuristic
    heuristics += np.sum(distance_matrix, axis=0)
    heuristics += np.sum(distance_matrix, axis=1)
    
    # Apply adaptive parameter tuning to adjust the heuristic values
    # This is a placeholder for any adaptive parameter tuning logic
    # For example, we could normalize the heuristics to a common scale
    # heuristics /= np.max(heuristics)
    
    return heuristics