import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristic matrix with zeros
    heuristics = np.zeros_like(distance_matrix)
    
    # Calculate the minimum distance from each node to all others
    min_distances = np.min(distance_matrix, axis=1)
    
    # Calculate the sum of the longest edges in each node pair
    max_distances = np.max(distance_matrix, axis=1)
    
    # Compute the heuristic value for each edge
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix)):
            if i != j:
                heuristics[i][j] = max_distances[i] - min_distances[j]
    
    return heuristics