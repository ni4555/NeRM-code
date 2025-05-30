import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming that the distance matrix is symmetric, we can use only half of it
    # and fill the rest to avoid redundant calculations.
    half_matrix = np.tril(distance_matrix, k=-1)
    half_matrix += np.tril(distance_matrix, k=-1).T
    
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # Compute the heuristic for each edge
    for i in range(len(distance_matrix)):
        for j in range(i + 1, len(distance_matrix)):
            # The heuristic is a function of the pairwise distances
            # For example, here we can use the simple average of the distances
            # to the current node's neighbors as the heuristic
            neighbors = np.delete(half_matrix[i], i, axis=0)
            heuristic = np.mean(neighbors)
            heuristic_matrix[i, j] = heuristic
            heuristic_matrix[j, i] = heuristic  # Since the matrix is symmetric
    
    return heuristic_matrix