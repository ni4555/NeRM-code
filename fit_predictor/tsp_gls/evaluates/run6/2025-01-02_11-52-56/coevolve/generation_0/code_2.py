import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Initialize an array with the same shape as the distance_matrix to store heuristics
    heuristics = np.zeros_like(distance_matrix)
    
    # Loop over the upper triangle of the distance matrix to fill in the heuristics
    # (i, j) and (j, i) should have the same heuristic since they are the same edge
    for i in range(distance_matrix.shape[0]):
        for j in range(i + 1, distance_matrix.shape[1]):
            # Calculate the heuristic as the negative of the distance, assuming the
            # lower the distance, the better the heuristic (since we want to minimize)
            heuristics[i, j] = -distance_matrix[i, j]
            heuristics[j, i] = heuristics[i, j]
    
    return heuristics