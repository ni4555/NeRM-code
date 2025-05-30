import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristics matrix with zeros
    heuristics = np.zeros_like(distance_matrix, dtype=float)
    
    # Fill the diagonal with a very high number to represent the start/finish of the tour
    np.fill_diagonal(heuristics, np.inf)
    
    # Iterate over the distance matrix to calculate heuristics
    for i in range(len(distance_matrix)):
        for j in range(i + 1, len(distance_matrix)):
            # Use some heuristic function to estimate the cost of this edge
            # For demonstration, we'll just use the negative distance (the lower, the better)
            heuristics[i, j] = -distance_matrix[i, j]
            heuristics[j, i] = -distance_matrix[j, i]
    
    return heuristics