import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # Assuming that the distance_matrix is symmetric and the diagonal elements are zeros
    # The heuristic is calculated as the minimum distance to any other vertex for each vertex
    n = distance_matrix.shape[0]
    heuristic_matrix = np.full(distance_matrix.shape, np.inf)

    for i in range(n):
        for j in range(n):
            if i != j:
                heuristic_matrix[i][j] = min(distance_matrix[i][:n], key=lambda x: x if x != 0 else np.inf)

    return heuristic_matrix