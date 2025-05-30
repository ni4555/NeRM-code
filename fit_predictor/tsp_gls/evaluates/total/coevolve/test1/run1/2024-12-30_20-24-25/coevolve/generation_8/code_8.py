import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    # This is a simple heuristic that returns the inverse of the distance matrix
    # which means shorter distances (and thus better edges) will have lower values.
    return 1 / (distance_matrix + 1e-10)  # Adding a small constant to avoid division by zero

# Example usage:
# distance_matrix = np.array([[0, 2, 9, 10],
#                             [1, 0, 6, 4],
#                             [15, 7, 0, 8],
#                             [6, 3, 12, 0]])
# heuristics = heuristics_v2(distance_matrix)
# print(heuristics)