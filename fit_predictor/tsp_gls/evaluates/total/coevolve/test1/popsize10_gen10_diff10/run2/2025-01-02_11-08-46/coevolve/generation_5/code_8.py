import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the total distance for each edge
    total_distances = np.sum(distance_matrix, axis=0) + np.sum(distance_matrix, axis=1) - np.diag(distance_matrix)
    
    # Normalize the distances by the maximum distance
    max_distance = np.max(total_distances)
    if max_distance == 0:
        raise ValueError("Distance matrix contains only zeros, cannot compute heuristic values.")
    
    # Create the heuristic matrix
    heuristic_matrix = total_distances / max_distance
    
    return heuristic_matrix