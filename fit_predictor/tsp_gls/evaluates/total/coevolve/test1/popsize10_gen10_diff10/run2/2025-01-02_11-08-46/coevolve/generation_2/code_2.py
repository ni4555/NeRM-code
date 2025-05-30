import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristics matrix with zeros
    heuristics_matrix = np.zeros_like(distance_matrix, dtype=float)
    
    # Compute the heuristics for each edge
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:
                # Calculate the heuristic based on the distance matrix
                # This is a simple example where we assume the heuristic is the negative distance
                heuristics_matrix[i][j] = -distance_matrix[i][j]
            else:
                # For the diagonal elements, which are not edges, we set a high value
                heuristics_matrix[i][j] = float('inf')
    
    return heuristics_matrix