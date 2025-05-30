import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize a matrix of the same shape as distance_matrix with zeros
    heuristics_matrix = np.zeros_like(distance_matrix)
    
    # Loop through each pair of nodes (i, j) where i < j
    for i in range(distance_matrix.shape[0]):
        for j in range(i + 1, distance_matrix.shape[1]):
            # Calculate the heuristic value as the negative distance to include edge (i, j)
            heuristics_matrix[i, j] = -distance_matrix[i, j]
    
    # Add self-loops with a very high penalty to prevent revisiting the same node
    self_loop_penalty = float('inf')
    heuristics_matrix[:, i] = self_loop_penalty
    heuristics_matrix[j, :] = self_loop_penalty
    
    return heuristics_matrix