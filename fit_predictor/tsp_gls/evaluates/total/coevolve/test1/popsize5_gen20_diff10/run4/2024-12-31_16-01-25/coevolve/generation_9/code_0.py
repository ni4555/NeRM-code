import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # Calculate Manhattan distance heuristic
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix)):
            if i != j:
                # Compute Manhattan distance
                heuristic = np.abs(i - j)
                # Assign to the heuristic matrix
                heuristic_matrix[i][j] = heuristic
    
    return heuristic_matrix