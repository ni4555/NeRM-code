import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristic array with zeros
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # Compute pairwise distances and use them to calculate heuristics
    for i in range(len(distance_matrix)):
        for j in range(i+1, len(distance_matrix)):
            # Calculate the heuristic value as the negative of the distance
            # This assumes that a smaller distance is better (lower heuristic value)
            heuristic_matrix[i][j] = -distance_matrix[i][j]
            heuristic_matrix[j][i] = -distance_matrix[j][i]  # Symmetry
    
    return heuristic_matrix