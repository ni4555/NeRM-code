import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming the distance_matrix is symmetric and that it is square.
    n = distance_matrix.shape[0]
    # Find the central city index. This could be randomized, but for simplicity, we choose the middle one.
    central_city = n // 2

    # Initialize the heuristic matrix with zeros.
    heuristic_matrix = np.zeros_like(distance_matrix)

    # Compute the Manhattan distance from the central city to all other cities.
    for i in range(n):
        for j in range(n):
            if i != j:
                heuristic_matrix[i][j] = np.abs(i - central_city) + np.abs(j - central_city)

    return heuristic_matrix