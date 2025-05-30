import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming Manhattan distance heuristic is used
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = np.zeros_like(distance_matrix, dtype=float)
    
    # Iterate over the distance matrix to calculate Manhattan distance
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix[i])):
            if i != j:
                # Calculate Manhattan distance for the edge (i, j)
                heuristic_matrix[i][j] = abs(i - j) * (len(distance_matrix) - 1)
            else:
                # Distance to itself is zero
                heuristic_matrix[i][j] = 0
    
    return heuristic_matrix