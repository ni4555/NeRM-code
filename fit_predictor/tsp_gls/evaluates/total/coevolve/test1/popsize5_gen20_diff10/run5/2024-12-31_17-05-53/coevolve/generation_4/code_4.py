import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Calculate the Manhattan distance for each edge
    # The Manhattan distance is the sum of the absolute differences in each dimension
    # Since the distance matrix is symmetric (distance[i][j] == distance[j][i]), we only need to compute half of it
    n = distance_matrix.shape[0]
    heuristics = np.zeros_like(distance_matrix)
    for i in range(n):
        for j in range(i + 1, n):  # start from i+1 to avoid duplicate edges
            # Calculate Manhattan distance for edge (i, j)
            manhattan_distance = np.sum(np.abs(distance_matrix[i] - distance_matrix[j]))
            # Assign the Manhattan distance as the heuristic for this edge
            heuristics[i, j] = heuristics[j, i] = manhattan_distance
    return heuristics