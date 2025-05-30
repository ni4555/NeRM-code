import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # This is a placeholder implementation for the heuristics_v2 function.
    # The actual implementation would involve the advanced Manhattan distance heuristic
    # as well as other components of the described hybrid solution, but since the details
    # of those are not provided, the following is a simple Manhattan distance heuristic.
    
    # Initialize an array of the same shape as distance_matrix with default values
    heuristic_values = np.full(distance_matrix.shape, np.inf)
    
    # For simplicity, assume that the first row and first column should not be considered as edges
    for i in range(1, distance_matrix.shape[0]):
        for j in range(1, distance_matrix.shape[1]):
            # Compute Manhattan distance between points i and j
            heuristic_value = abs(i - j)  # Manhattan distance is simply the absolute difference in indices
            
            # Update the heuristic value if it's less than the current
            if heuristic_value < heuristic_values[i][j]:
                heuristic_values[i][j] = heuristic_value
    
    return heuristic_values