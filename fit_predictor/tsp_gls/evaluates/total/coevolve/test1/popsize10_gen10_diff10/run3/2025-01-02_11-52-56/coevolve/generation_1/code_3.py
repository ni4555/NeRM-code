import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = np.zeros_like(distance_matrix, dtype=np.float32)
    
    # Compute the heuristic value for each edge
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:
                # The heuristic is based on the distance divided by the average distance
                heuristic = distance_matrix[i, j] / np.mean(distance_matrix)
                heuristic_matrix[i, j] = heuristic
    
    return heuristic_matrix