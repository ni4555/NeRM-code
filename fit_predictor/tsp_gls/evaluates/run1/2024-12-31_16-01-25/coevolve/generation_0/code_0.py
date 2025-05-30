import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristics array with zeros
    heuristics = np.zeros_like(distance_matrix)
    
    # Calculate the heuristics for each edge
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:
                # Calculate the heuristic for the edge from i to j
                # This can be a simple heuristic, such as the minimum distance from j to any other node
                heuristics[i, j] = np.min(distance_matrix[j])
                
    return heuristics