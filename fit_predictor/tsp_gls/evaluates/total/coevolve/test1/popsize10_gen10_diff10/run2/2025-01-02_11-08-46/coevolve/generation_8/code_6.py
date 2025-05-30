import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the distance matrix is symmetric
    if not np.array_equal(distance_matrix, distance_matrix.T):
        raise ValueError("Distance matrix must be symmetric.")
    
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # Implement your advanced distance-based normalization techniques and minimum sum heuristic here
    # For the sake of this example, we'll just use a simple normalization by the maximum distance
    max_distance = np.max(distance_matrix)
    if max_distance == 0:
        raise ValueError("Distance matrix contains zero distances, which is not allowed.")
    
    # Normalize the distance matrix
    normalized_distance_matrix = distance_matrix / max_distance
    
    # Calculate the minimum sum of distances for each edge
    min_sum_distance = np.sum(normalized_distance_matrix, axis=0)
    
    # Assign the minimum sum as the heuristic value
    heuristic_matrix = min_sum_distance
    
    return heuristic_matrix