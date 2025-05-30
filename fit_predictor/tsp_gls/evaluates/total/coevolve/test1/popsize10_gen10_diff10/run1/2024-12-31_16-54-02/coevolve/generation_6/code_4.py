import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Create a copy of the distance matrix to avoid modifying the original data
    heuristics_matrix = np.copy(distance_matrix)
    
    # Initialize a matrix to hold the heuristic values, which will be the negative of the distances
    # because we want to minimize the value in the heuristic
    for i in range(len(heuristics_matrix)):
        for j in range(len(heuristics_matrix)):
            if i != j:
                # Set the heuristic value to be the negative of the distance, except for the diagonal
                heuristics_matrix[i][j] = -distance_matrix[i][j]
            else:
                # The diagonal elements represent the distance from a node to itself, which is 0
                heuristics_matrix[i][j] = 0
    
    return heuristics_matrix