import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Simple distance-based heuristic: the further the distance, the worse the heuristic value
    # We use the maximum distance between any two nodes to create a simple negative heuristic
    max_distance = np.max(distance_matrix)
    
    # Initialize a result matrix with the same shape as the distance matrix
    # and fill it with negative of the maximum distance to reflect the heuristic value
    heuristic_matrix = -np.ones_like(distance_matrix)
    
    # Update the heuristic value for each edge to be the negative of its distance
    # We do not modify the diagonal elements (distance to itself is zero)
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix[i])):
            if i != j:  # Exclude diagonal elements
                heuristic_matrix[i, j] = -distance_matrix[i, j]
    
    return heuristic_matrix