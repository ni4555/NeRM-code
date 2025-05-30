import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristics array with the same shape as the distance matrix
    heuristics = np.zeros_like(distance_matrix)
    
    # Compute the heuristics using a simple approach: the sum of distances from each node to all other nodes
    for i in range(distance_matrix.shape[0]):
        heuristics[i] = np.sum(distance_matrix[i])
    
    return heuristics