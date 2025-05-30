import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the distance matrix is symmetric
    assert np.allclose(distance_matrix, distance_matrix.T), "Distance matrix must be symmetric"
    
    # Initialize a matrix of the same shape as the input with zeros
    heuristics_matrix = np.zeros_like(distance_matrix)
    
    # Compute the heuristic values by taking the minimum distance from each node to all others
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:
                heuristics_matrix[i, j] = min(distance_matrix[i, :]) + min(distance_matrix[:, j])
            else:
                heuristics_matrix[i, j] = float('inf')  # No heuristic for the same node
    
    return heuristics_matrix