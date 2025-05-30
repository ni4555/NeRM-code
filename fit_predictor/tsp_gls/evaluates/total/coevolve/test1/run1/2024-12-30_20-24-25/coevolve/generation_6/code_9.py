import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # Calculate Manhattan distances for each edge
    for i in range(len(distance_matrix)):
        for j in range(i + 1, len(distance_matrix)):  # Avoid duplicate calculations
            heuristic_matrix[i][j] = heuristic_matrix[j][i] = np.abs(i - j).sum()
    
    return heuristic_matrix