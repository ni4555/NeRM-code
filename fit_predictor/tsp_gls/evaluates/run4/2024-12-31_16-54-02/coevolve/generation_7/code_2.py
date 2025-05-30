import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristics matrix with zeros
    heuristics = np.zeros_like(distance_matrix)
    
    # Compute the heuristics for each edge
    for i in range(len(distance_matrix)):
        for j in range(i+1, len(distance_matrix)):  # Only compute for unique edges
            # Use the shortest path algorithm to find the shortest path between node i and j
            # without looping back to the origin
            heuristics[i][j] = np.sum(distance_matrix[i]) + distance_matrix[i][j]
            heuristics[j][i] = np.sum(distance_matrix[j]) + distance_matrix[j][i]
    
    return heuristics