import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the distance_matrix is symmetric and has no zero diagonal
    n = distance_matrix.shape[0]
    heuristics_matrix = np.zeros_like(distance_matrix)

    # Calculate the maximum distance for each row and column
    max_row_distances = np.max(distance_matrix, axis=1)
    max_col_distances = np.max(distance_matrix, axis=0)

    # Apply the heuristic: the heuristic for an edge (i, j) is the maximum distance
    # from i to any other node minus the distance from i to j
    for i in range(n):
        for j in range(n):
            if i != j:
                heuristics_matrix[i][j] = max_row_distances[i] - distance_matrix[i][j]

    return heuristics_matrix