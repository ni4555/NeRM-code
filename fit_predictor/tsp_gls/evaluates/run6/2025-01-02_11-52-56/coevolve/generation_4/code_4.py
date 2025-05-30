import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the maximum distance from each node to any other node
    max_distances = np.max(distance_matrix, axis=1)
    
    # Calculate the heuristic as the average of the maximum distances
    # from each node to any other node
    heuristic_values = np.mean(max_distances)
    
    # Create a matrix of the same shape as the input matrix, with all values set to the heuristic
    heuristic_matrix = np.full(distance_matrix.shape, heuristic_values)
    
    return heuristic_matrix