import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Normalize the distance matrix to ensure that the sum of distances for any tour is less than or equal to the sum of all edges
    normalized_distance_matrix = distance_matrix / np.sum(distance_matrix)
    
    # Apply a minimum sum heuristic for edge selection
    min_sum_heuristic = np.min(distance_matrix, axis=0) + np.min(distance_matrix, axis=1)
    
    # Combine the normalized distance with the minimum sum heuristic
    combined_heuristic = normalized_distance_matrix + min_sum_heuristic
    
    # Return the heuristic matrix
    return combined_heuristic