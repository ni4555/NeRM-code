import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming distance_matrix is square and symmetric
    n = distance_matrix.shape[0]
    # Initialize an array with the same shape as the distance matrix to store heuristics
    heuristics = np.zeros_like(distance_matrix)
    
    # Calculate the heuristic for each edge
    for i in range(n):
        for j in range(n):
            if i != j:
                # For simplicity, we use the Manhattan distance to the nearest city
                min_distance = np.min(distance_matrix[i, :])
                heuristics[i, j] = distance_matrix[i, j] - min_distance
    
    return heuristics