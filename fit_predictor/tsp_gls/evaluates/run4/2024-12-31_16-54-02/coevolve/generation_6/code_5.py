import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming that the distance_matrix is symmetric and that the diagonal is filled with zeros
    n = distance_matrix.shape[0]
    heuristics = np.zeros_like(distance_matrix)

    # Loop over all pairs of nodes except the first one (0-indexed), to avoid the starting node
    for i in range(1, n):
        for j in range(i+1, n):
            # Calculate the heuristic as the sum of distances from node i to node j and back
            heuristics[i][j] = heuristics[j][i] = distance_matrix[i][j] + distance_matrix[j][i]
    
    return heuristics