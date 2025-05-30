import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming distance_matrix is a symmetric matrix where distance_matrix[i][j] is the distance from node i to node j
    num_nodes = distance_matrix.shape[0]
    heuristics = np.zeros_like(distance_matrix)

    # Calculate the average distance for each edge
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            average_distance = np.mean(distance_matrix[i, :]) + np.mean(distance_matrix[j, :])
            heuristics[i, j] = average_distance
            heuristics[j, i] = average_distance

    return heuristics