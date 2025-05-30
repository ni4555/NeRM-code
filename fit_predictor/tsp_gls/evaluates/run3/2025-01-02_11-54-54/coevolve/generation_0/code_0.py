import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the same shape array to store heuristics
    heuristics = np.zeros_like(distance_matrix)
    
    # Set the heuristic for each edge based on the minimum distance to a node
    for i in range(len(distance_matrix)):
        for j in range(i+1, len(distance_matrix)):
            heuristics[i, j] = heuristics[j, i] = np.min(distance_matrix[i, :].max() + distance_matrix[:, j].min())
    
    return heuristics