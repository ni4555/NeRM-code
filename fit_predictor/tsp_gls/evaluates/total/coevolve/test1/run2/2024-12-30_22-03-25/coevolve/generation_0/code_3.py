import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristics array with zeros
    heuristics = np.zeros_like(distance_matrix)
    
    # Calculate the minimum distance from each node to all other nodes
    for i in range(distance_matrix.shape[0]):
        min_dist = np.min(distance_matrix[i])
        heuristics[i] = distance_matrix[i] - min_dist
    
    return heuristics