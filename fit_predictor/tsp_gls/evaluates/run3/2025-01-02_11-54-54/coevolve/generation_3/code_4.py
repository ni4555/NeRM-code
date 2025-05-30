import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming that the distance matrix is symmetric and has the same size as the number of nodes
    # The heuristic function could be a simple function like the Manhattan distance from the origin
    # to each node, which is a common heuristic for TSP problems.
    # For simplicity, we'll use the sum of the row and column indices as a heuristic value,
    # as it is a simple heuristic without needing additional computations.

    # Calculate the Manhattan distance from the origin (0, 0) to each node
    heuristic_values = np.sum(np.column_stack((np.arange(distance_matrix.shape[0]), np.arange(distance_matrix.shape[1]))), axis=1)

    # We create a matrix of the same shape as the distance matrix, where each cell contains
    # the heuristic value for the corresponding edge.
    heuristic_matrix = np.zeros_like(distance_matrix)
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j:
                heuristic_matrix[i, j] = heuristic_values[i] + heuristic_values[j] - distance_matrix[i, j]

    return heuristic_matrix