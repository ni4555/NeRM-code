import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristic array with the same shape as the distance matrix
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # Calculate the minimum pairwise distances
    min_distances = np.min(distance_matrix, axis=1)
    
    # Adjust the minimum distances dynamically based on the distance matrix
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix)):
            if i != j:
                # Adjust based on the ratio of the minimum distance to the current edge distance
                heuristic_matrix[i, j] = min_distances[i] / distance_matrix[i, j]
    
    return heuristic_matrix