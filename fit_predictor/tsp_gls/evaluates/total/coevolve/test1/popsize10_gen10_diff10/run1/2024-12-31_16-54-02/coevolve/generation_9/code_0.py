import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the minimum distance from each node to all other nodes
    min_distances = np.min(distance_matrix, axis=1)
    
    # Compute the heuristics as the difference between the minimum distance
    # from the current node to all other nodes and the actual edge distance
    heuristics = min_distances - distance_matrix
    
    return heuristics