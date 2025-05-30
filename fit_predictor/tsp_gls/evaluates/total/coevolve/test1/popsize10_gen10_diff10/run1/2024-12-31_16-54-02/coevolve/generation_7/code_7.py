import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the distance_matrix is a symmetric matrix where distance_matrix[i][j] is the distance between node i and node j
    # and distance_matrix[i][i] is 0.
    
    # Create a matrix filled with ones, which will be the initial heuristic value for each edge
    heuristics = np.ones_like(distance_matrix)
    
    # For each pair of nodes (i, j) calculate the heuristic value as the distance divided by the maximum distance in the row or column
    for i in range(len(distance_matrix)):
        row_max = np.max(distance_matrix[i])
        col_max = np.max(distance_matrix[:, i])
        heuristics[i] = distance_matrix[i] / max(row_max, col_max)
    
    return heuristics