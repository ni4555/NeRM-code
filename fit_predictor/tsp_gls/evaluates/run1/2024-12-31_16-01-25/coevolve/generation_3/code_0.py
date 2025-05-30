import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the Manhattan distance heuristic
    Manhattan_distance_heuristic = np.abs(np.diff(distance_matrix, axis=0)).sum(axis=1) + np.abs(np.diff(distance_matrix, axis=1)).sum(axis=1)
    
    # Use the distance matrix directly as a heuristic
    direct_distance_heuristic = distance_matrix.sum(axis=1)
    
    # Combine the two heuristics by taking the minimum of the two for each edge
    combined_heuristic = np.minimum(Manhattan_distance_heuristic, direct_distance_heuristic)
    
    return combined_heuristic