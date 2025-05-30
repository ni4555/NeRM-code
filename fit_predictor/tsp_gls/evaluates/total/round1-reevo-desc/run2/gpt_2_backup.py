import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    n = distance_matrix.shape[0]
    heuristics_matrix = np.zeros_like(distance_matrix)

    for i in range(n):
        for j in range(n):
            if i != j:
                # Calculate the heuristic by considering the local structure
                # and a global domain-specific penalty
                local_min = np.min(distance_matrix[i, distance_matrix[i] != float('inf')])
                global_min = np.min(distance_matrix[distance_matrix[:, i] != float('inf'), i])
                penalty = 1 if i == 0 or i == n - 1 else 0.5  # Higher penalty for edges to the start/end node
                heuristics_matrix[i, j] = (local_min + global_min + penalty * distance_matrix[i, j]) / 2
            else:
                heuristics_matrix[i, j] = float('inf')

    return heuristics_matrix
