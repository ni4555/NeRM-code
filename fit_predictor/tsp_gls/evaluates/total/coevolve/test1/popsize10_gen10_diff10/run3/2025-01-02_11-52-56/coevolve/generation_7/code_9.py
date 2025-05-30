import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the distance matrix is symmetric, and the diagonal is zero
    heuristic_matrix = np.zeros_like(distance_matrix)
    
    # Calculate the heuristic values
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:
                # Simple heuristic: the negative of the distance
                heuristic_matrix[i][j] = -distance_matrix[i][j]
    
    return heuristic_matrix