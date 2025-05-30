import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the distance_matrix is square and symmetric
    n = distance_matrix.shape[0]
    
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # Calculate the heuristic for each edge
    for i in range(n):
        for j in range(i + 1, n):
            # Example heuristic: the sum of the distances to the nearest nodes
            # excluding the endpoints themselves
            heuristic = np.sum(distance_matrix[i, :i] + distance_matrix[i, i+1:])
            heuristic_matrix[i, j] = heuristic
            heuristic_matrix[j, i] = heuristic
    
    return heuristic_matrix