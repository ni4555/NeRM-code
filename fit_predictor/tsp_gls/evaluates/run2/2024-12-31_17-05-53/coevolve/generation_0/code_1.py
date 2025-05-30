import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristic array with zeros
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # Loop through each pair of nodes to compute the heuristic
    for i in range(len(distance_matrix)):
        for j in range(i+1, len(distance_matrix)):
            # The heuristic is a rough estimate of the cost to include this edge
            # In this example, we simply use the inverse of the distance
            # This is a very basic heuristic that might not be very effective
            # in all cases, but it's a starting point
            heuristic_matrix[i, j] = 1.0 / distance_matrix[i, j]
            heuristic_matrix[j, i] = 1.0 / distance_matrix[j, i]
    
    return heuristic_matrix