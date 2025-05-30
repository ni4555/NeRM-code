import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # Calculate Manhattan distance heuristics
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix[i])):
            if i != j:
                # For each city i, compute the Manhattan distance to city j
                heuristic_matrix[i][j] = abs(i - j)
    
    # Incorporate the direct use of the distance matrix as a heuristic
    # This is essentially the same as the Manhattan distance for a complete graph
    # However, we will add a small constant to avoid zero values which can cause division by zero
    small_constant = 1e-6
    heuristic_matrix += distance_matrix + small_constant
    
    return heuristic_matrix