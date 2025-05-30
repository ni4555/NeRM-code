import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize a matrix with the same shape as the input distance matrix to store heuristics
    heuristics_matrix = np.zeros_like(distance_matrix, dtype=float)
    
    # Example heuristic: Use the average distance from each node to all other nodes
    for i in range(distance_matrix.shape[0]):
        average_distance = np.mean(distance_matrix[i, :])
        heuristics_matrix[i, :] = average_distance
    
    return heuristics_matrix