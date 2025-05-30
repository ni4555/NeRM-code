import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize a matrix with zeros of the same shape as the distance matrix
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # Calculate the prior indicators for each edge
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix[i])):
            if i != j:
                # Use a simple heuristic: the larger the distance, the "worse" the edge
                heuristic_matrix[i][j] = distance_matrix[i][j]
            else:
                # No edge to itself, set the heuristic to a very large number
                heuristic_matrix[i][j] = float('inf')
    
    return heuristic_matrix