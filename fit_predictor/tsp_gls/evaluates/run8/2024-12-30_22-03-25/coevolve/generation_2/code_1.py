import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristics matrix with zeros
    heuristics = np.zeros_like(distance_matrix)
    
    # Calculate the heuristic values for each edge
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:
                # Example heuristic: negative of the distance (the closer, the better)
                heuristics[i, j] = -distance_matrix[i, j]
            else:
                # No heuristic for self-loops
                heuristics[i, j] = float('inf')
    
    return heuristics