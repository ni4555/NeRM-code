import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the sum of the distances for each edge in the matrix
    edge_sums = np.sum(distance_matrix, axis=1) + np.sum(distance_matrix, axis=0) - np.diag(distance_matrix)
    
    # Normalize the sums by the total sum of the matrix to create a heuristic value for each edge
    total_distance = np.sum(distance_matrix)
    heuristic_values = edge_sums / total_distance
    
    return heuristic_values