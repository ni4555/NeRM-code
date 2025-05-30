import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming distance_matrix is square and symmetric (since it's a distance matrix)
    num_cities = distance_matrix.shape[0]
    
    # Initialize the heuristic matrix with zeros
    heuristics = np.zeros_like(distance_matrix)
    
    # Compute distance-weighted normalization and resilient minimum spanning tree (RMST) heuristic
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                # Calculate distance-weighted normalization
                weight = (distance_matrix[i, j] / np.sum(distance_matrix[i])) * (np.sum(distance_matrix[i]) / np.sum(distance_matrix[j]))
                
                # Calculate RMST for edge i-j
                rmst = np.sum(distance_matrix[i]) + np.sum(distance_matrix[j]) - distance_matrix[i, j]
                
                # Combine the two to get the heuristic value
                heuristics[i, j] = weight + rmst
    
    return heuristics