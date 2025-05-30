import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the distance matrix is symmetric and the diagonal is filled with zeros
    n = distance_matrix.shape[0]
    heuristics_matrix = np.zeros_like(distance_matrix)

    # Example heuristic: Use the average distance of each edge as the heuristic value
    for i in range(n):
        for j in range(n):
            if i != j:
                heuristics_matrix[i, j] = np.mean(distance_matrix[i, :]) + np.mean(distance_matrix[:, j])
    
    return heuristics_matrix