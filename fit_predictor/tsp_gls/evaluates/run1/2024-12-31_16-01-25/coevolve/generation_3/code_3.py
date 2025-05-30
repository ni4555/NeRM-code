import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming Manhattan distance heuristic is being used here
    # Manhattan distance is the sum of the absolute differences of their Cartesian coordinates.
    # For a TSP, it can be thought of as the sum of the horizontal and vertical distances
    # needed to move from one city to the next in the distance matrix.
    
    # Create a new matrix for heuristics, initialized to zero
    heuristics = np.zeros_like(distance_matrix)
    
    # For each city (i), calculate the Manhattan distance to all other cities (j)
    # and store it in the heuristics matrix.
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:
                # Sum of the absolute differences for each dimension
                heuristics[i, j] = np.abs(distance_matrix[i, j] - distance_matrix[i, i])
    
    return heuristics